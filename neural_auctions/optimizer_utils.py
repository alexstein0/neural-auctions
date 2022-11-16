"""
optimizer_utils.py
Optimizer related functions
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

from typing import Iterable

import torch
from icecream import ic
from omegaconf import DictConfig
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from .warmup import BaseWarmup, ExponentialWarmup


def get_optimizer(optimizer: str, lr: float, params: Iterable[torch.Tensor], wd: float = 2e-4) -> Optimizer:
    optimizer_name = optimizer.lower()
    if optimizer_name == "sgd":
        optimizer = SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"{ic.format()}: Optimizer choice of {optimizer_name} not yet implemented.")
    return optimizer


def get_optimizer_and_schedulers(optim_args: DictConfig,
                                 params: Iterable[torch.Tensor],
                                 state_dict: dict = None) -> (Optimizer,
                                                              BaseWarmup,
                                                              torch.optim.lr_scheduler,
                                                              int):
    epochs = optim_args.epochs
    lr_decay = optim_args.lr_decay
    lr_schedule = optim_args.lr_schedule
    lr_factor = optim_args.lr_factor
    warmup_period = optim_args.warmup_period
    optimizer = get_optimizer(optim_args.optimizer, optim_args.lr, params, optim_args.weight_decay)

    start_epoch = 0
    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=0)
        start_epoch = state_dict["epoch"] + 1
    else:
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=warmup_period)

    if lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_schedule,
                                   gamma=lr_factor, last_epoch=-1)
    elif lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {lr_decay} not yet implemented.")

    return optimizer, warmup_scheduler, lr_scheduler, start_epoch
