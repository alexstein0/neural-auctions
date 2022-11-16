"""
model.py
Model classes for Neural Auctions
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

from abc import abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_auctions.auction_utils import calculate_agent_util, tiled_misreport_util, \
    get_misreports_from_bids_with_multiple_inits
from neural_auctions.warmup import BaseWarmup


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs: torch.Tensor, sigma: float = 0.0):
        pass

    @abstractmethod
    def train_loop(self,
                   train_loader: DataLoader,
                   model_cfg: DictConfig,
                   optimizer: Optimizer,
                   warmup_scheduler: BaseWarmup,
                   lr_scheduler: torch.optim.lr_scheduler,
                   device: str = "cpu") -> dict:
        pass

    def _perturb_allocs(self, allocs: torch.Tensor, sigma: float):
        """
        Add random noise to the allocations
        :param allocs: tensor of dimension [batch_size, n_agents + 1, n_items]
        :param sigma: float indicating the standard deviation of the noise
        :return: perturbed allocations of the same size as allocs
        """
        # get noise
        noise = torch.normal(0.0, sigma, size=allocs.size(), device=allocs.device)

        # add noise
        pert_allocs = allocs + noise

        return pert_allocs

    def test_loop(self,
                  dataloader: DataLoader,
                  misreport_iters: int,
                  misreport_lr: float,
                  misreport_inits: int,
                  sigma: float = 0,
                  device: str = "cpu") -> dict:
        self.eval()
        payments_total = 0
        regrets_total = 0
        num_training_data = 0
        regret_max = 0
        full_misreport_util = torch.Tensor().to(device)
        full_truthful_util = torch.Tensor().to(device)
        full_regret = torch.Tensor().to(device)

        for i, (bids,) in enumerate(tqdm(dataloader, leave=False)):
            bids = bids.to(device)
            truthful_allocs, truthful_payments = self.forward(bids, sigma)
            truthful_util = calculate_agent_util(bids, truthful_allocs, truthful_payments)

            misreports = get_misreports_from_bids_with_multiple_inits(self, bids, misreport_inits,
                                                                      misreport_iters, misreport_lr)
            misreport_util = tiled_misreport_util(misreports, bids, self)

            regret = torch.relu(misreport_util - truthful_util)
            regret_max = max(regret_max, regret.max().item())

            payments_total += truthful_payments.sum().item()
            regrets_total += regret.mean(dim=1).sum().item()
            num_training_data += bids.size(0)
            full_truthful_util = torch.cat((full_truthful_util, truthful_util), dim=0)
            full_misreport_util = torch.cat((full_misreport_util, misreport_util), dim=0)
            full_regret = torch.cat((full_regret, regret), dim=0)

        util_difference = (full_misreport_util - full_truthful_util).mean().item()
        util_better = (torch.sum(full_misreport_util.ge(full_truthful_util)) / torch.numel(full_truthful_util)).item()

        payments_mean = payments_total / num_training_data
        regret_mean = regrets_total / num_training_data
        result = {"payments_mean": payments_mean,
                  "regret_mean": regret_mean,
                  "regret_max": regret_max,
                  "util_difference": util_difference,
                  "util_better": util_better
                  }
        return result

    def test_loop_sigma(self,
                        dataloader: DataLoader,
                        misreport_iters: int,
                        misreport_lr: float,
                        misreport_inits: int,
                        util_num_samples: int,
                        sigma: float = 0,
                        device: str = "cpu") -> dict:
        self.eval()
        payments_total = 0
        regrets_total = 0
        num_training_data = 0
        regret_max = 0
        full_misreport_util = torch.Tensor().to(device)
        full_truthful_util = torch.Tensor().to(device)
        full_regret = torch.Tensor().to(device)

        num_samples = util_num_samples if sigma > 0 else 1

        for i, (bids,) in enumerate(tqdm(dataloader, leave=False)):
            bids = bids.to(device)
            n_batch, n_agents, n_items = bids.size()
            big_bids = bids.repeat(num_samples, 1, 1)

            truthful_allocs, truthful_payments = self.forward(big_bids, sigma)
            truthful_allocs = truthful_allocs.view(num_samples, n_batch, n_agents, n_items).mean(dim=0, keepdim=False)
            truthful_payments = truthful_payments.view(num_samples, n_batch, n_agents).mean(dim=0, keepdim=False)

            truthful_util = calculate_agent_util(bids, truthful_allocs, truthful_payments)

            misreports, misreport_util = get_misreports_from_bids_with_multiple_inits(self, bids, misreport_inits,
                                                                                      misreport_iters, misreport_lr,
                                                                                      sigma=sigma,
                                                                                      util_num_samples=num_samples,
                                                                                      return_agent_util=True)

            regret = torch.relu(misreport_util - truthful_util)
            regret_max = max(regret_max, regret.max().item())

            payments_total += truthful_payments.sum().item()
            regrets_total += regret.mean(dim=1).sum().item()
            num_training_data += bids.size(0)
            full_truthful_util = torch.cat((full_truthful_util, truthful_util), dim=0)
            full_misreport_util = torch.cat((full_misreport_util, misreport_util), dim=0)
            full_regret = torch.cat((full_regret, regret), dim=0)

        util_difference = (full_misreport_util - full_truthful_util).mean().item()
        util_better = (torch.sum(full_misreport_util.ge(full_truthful_util)) / torch.numel(full_truthful_util)).item()

        payments_mean = payments_total / num_training_data
        regret_mean = regrets_total / num_training_data
        result = {"payments_mean": payments_mean,
                  "regret_mean": regret_mean,
                  "regret_max": regret_max,
                  "util_difference": util_difference,
                  "util_better": util_better
                  }
        return result

    def inversion_loop(self,
                       sigma: float,
                       alpha: float,
                       optimizer: Optimizer,
                       true_payments: torch.Tensor,
                       true_allocs: torch.Tensor,
                       guessed_bids: torch.Tensor) -> dict:
        optimizer.zero_grad()
        n_items = torch.tensor(true_allocs.size(2))
        guessed_allocs, guessed_payments = self.forward(guessed_bids, sigma)
        allocation_loss = torch.norm(guessed_allocs - true_allocs) / torch.sqrt(n_items)
        payment_loss = torch.norm(guessed_payments - true_payments) / n_items
        privacy_loss = alpha * torch.norm(1 / 2 - guessed_bids, p=1)
        loss = allocation_loss + payment_loss + privacy_loss
        loss.backward()
        optimizer.step()
        output_dict = {"loss": loss,
                       "payment_loss": payment_loss,
                       "allocation_loss": allocation_loss}
        return output_dict
