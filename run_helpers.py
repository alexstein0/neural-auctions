"""
run_helpers.py
Helper functions for all kinds of experiements
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

import json
import logging
import os
from collections import OrderedDict

import torch
from omegaconf import DictConfig, OmegaConf


def setup(cfg: DictConfig, caller: str) -> (str, logging.Logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info(f"{caller} running.")
    if cfg.exp_str is not None:
        log.info(f"Running experiment with parameters: {cfg.exp_str}")
    if cfg.hyp.random_seed is not None:
        torch.manual_seed(cfg.hyp.random_seed)
    log.info(OmegaConf.to_yaml(cfg))

    return device, log


def save_results(result: OrderedDict, file_prefix: str = None):
    file_name = "result" if not file_prefix else f"{file_prefix}_result"
    with open(os.path.join(f"{file_name}.json"), "w") as fp:
        json.dump(result, fp, indent=4)
