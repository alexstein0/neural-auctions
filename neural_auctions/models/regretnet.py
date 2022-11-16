"""
regretnet.py
Model classes for RegretNet Architecture
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

import torch
import torch.nn.init
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Model
from ..auction_utils import calculate_agent_util, get_misreports_from_bids_with_multiple_inits, tiled_misreport_util
from ..warmup import BaseWarmup


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ViewCut(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :-1, :]


class RegretNet(Model):
    def __init__(self,
                 n_agents,
                 n_items,
                 hidden_layer_size,
                 n_hidden_layers,
                 clamp_op=None,
                 activation="tanh",
                 separate=False):
        super().__init__()

        self.activation = activation
        if activation == "tanh":
            self.act = nn.Tanh
        else:
            self.act = nn.ReLU

        # this is for additive valuations only
        if clamp_op is None:
            self.clamp_op = lambda x: torch.clamp(x, 0, 1)
        else:
            self.clamp_op = clamp_op

        self.n_agents = n_agents
        self.n_items = n_items

        self.input_size = self.n_agents * self.n_items
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.separate = separate

        # outputs are agents (+dummy agent) per item, plus payments per agent
        self.allocations_size = (self.n_agents + 1) * self.n_items
        self.payments_size = self.n_agents

        # Set a_activation to softmax
        self.allocation_head = [nn.Linear(self.hidden_layer_size, self.allocations_size),
                                View((-1, self.n_agents + 1, self.n_items)),
                                ]
        self.view_cut = ViewCut()

        # Set p_activation to frac_sigmoid
        self.payment_head = [nn.Linear(self.hidden_layer_size, self.payments_size), nn.Sigmoid()]

        if self.separate:
            self.nn_model = nn.Sequential()
            self.payment_head = [nn.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                [l for i in range(self.n_hidden_layers)
                                 for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                self.payment_head

            self.payment_head = nn.Sequential(*self.payment_head)
            self.allocation_head = [nn.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                   [l for i in range(self.n_hidden_layers)
                                    for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                   self.allocation_head
            self.allocation_head = nn.Sequential(*self.allocation_head)
        else:
            self.nn_model = nn.Sequential(
                *([nn.Linear(self.input_size, self.hidden_layer_size), self.act()] +
                  [l for i in range(self.n_hidden_layers)
                   for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
            )
            self.allocation_head = nn.Sequential(*self.allocation_head)
            self.payment_head = nn.Sequential(*self.payment_head)

        self.batches_trained = 0
        self.epochs_trained = 0

    def forward(self, reports: torch.Tensor, sigma: float = 0.0) -> (torch.Tensor, torch.Tensor):
        # reports should be of size [batch_size, n_agents, n_items]
        # should be reshaped to [batch_size, n_agents * n_items]
        # output should be of size [batch_size, n_agents, n_items],
        x = reports.view(-1, self.n_agents * self.n_items)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)

        if sigma > 0.0:
            allocs = self._perturb_allocs(allocs, sigma)

        allocs = torch.nn.functional.softmax(allocs, dim=1)
        allocs = self.view_cut(allocs)

        # frac_sigmoid payment: multiply p = p_tilde * sum(alloc*bid)
        payments = self.payment_head(x) * torch.sum(allocs * reports, dim=2)

        return allocs, payments

    def train_loop(self,
                   train_loader: DataLoader,
                   model_cfg: DictConfig,
                   optimizer: Optimizer,
                   warmup_scheduler: BaseWarmup,
                   lr_scheduler: torch.optim.lr_scheduler,
                   device: str = "cpu") -> dict:
        self.train()

        loss_func = torch.nan
        regret_loss = torch.nan
        regret_max = torch.nan
        regret_quad = torch.nan
        payment_loss = torch.nan
        full_misreport_util = torch.Tensor().to(device)
        full_truthful_util = torch.Tensor().to(device)

        for i, (bids,) in enumerate(tqdm(train_loader, leave=False)):
            optimizer.zero_grad()
            bids = bids.to(device)
            truthful_allocs, truthful_payments = self.forward(bids)
            truthful_util = calculate_agent_util(bids, truthful_allocs, truthful_payments)
            misreports = get_misreports_from_bids_with_multiple_inits(self,
                                                                      bids,
                                                                      model_cfg.train_misreport_inits,
                                                                      model_cfg.train_misreport_iters,
                                                                      model_cfg.train_misreport_lr)
            if model_cfg.tile_misreports:
                misreport_util = tiled_misreport_util(misreports, bids, self)
                regret = torch.relu(misreport_util - truthful_util)
            else:
                misreport_allocs, misreport_payments = self.forward(misreports)
                misreport_util = calculate_agent_util(bids, misreport_allocs, misreport_payments)
                regret = torch.clamp_min(misreport_util - truthful_util, 0)

            full_truthful_util = torch.cat((full_truthful_util, truthful_util), dim=0)
            full_misreport_util = torch.cat((full_misreport_util, misreport_util), dim=0)

            regret_loss = regret.mean()
            if model_cfg.regret_quad_type == "square_first":
                regret_quad = (regret ** 2).mean()
            elif model_cfg.regret_quad_type == "mean_first":
                regret_quad = regret.mean() ** 2
            payment_loss = truthful_payments.sum(dim=1).mean()
            regret_max = regret.max().item()

            # Calculate loss
            loss_func = model_cfg.regret_weight * regret_loss \
                        + (model_cfg.rho / 2) * regret_quad \
                        - model_cfg.payment_weight * payment_loss

            # update model
            loss_func.backward()
            optimizer.step()

            # update various fancy multipliers
            if regret_loss >= model_cfg.regret_limit:
                if (self.batches_trained + 1) % model_cfg.lagr_update_iter_freq == 0:
                    with torch.no_grad():
                        model_cfg.regret_weight += model_cfg.rho * regret_loss.item()

                # this is because we may want to update rho as a function of iters not epochs
                if (model_cfg.rho_incr_iter_freq is not None) and \
                        (((self.batches_trained + 1) % model_cfg.rho_incr_iter_freq) == 0):
                    model_cfg.rho += model_cfg.rho_incr_amount

            self.batches_trained += 1

        if regret_loss >= model_cfg.regret_limit and (model_cfg.rho_incr_epoch_freq is not None) \
                and (((self.epochs_trained + 1) % model_cfg.rho_incr_epoch_freq) == 0):
            model_cfg.rho += model_cfg.rho_incr_amount
        self.epochs_trained += 1

        lr_scheduler.step()
        warmup_scheduler.dampen()
        util_difference = (full_misreport_util - full_truthful_util).mean().item()
        util_better = (torch.sum(full_misreport_util.ge(full_truthful_util)) / torch.numel(full_truthful_util)).item()

        output = {"train_loss": loss_func,
                  "regret_loss": regret_loss,
                  "regret_max": regret_max,
                  "regret_quad": regret_quad,
                  "payment_loss": payment_loss,
                  "regret_weight": model_cfg.regret_weight,
                  "util_difference": util_difference,
                  "util_better": util_better,
                  "rho": model_cfg.rho,
                  "payment_mult": model_cfg.payment_weight}
        return output
