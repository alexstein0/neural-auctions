"""
utils.py
Utility functions
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

import logging

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from .models import RegretNet


def write(writer: SummaryWriter, tag: str, scalar: object, step: int):
    if writer is not None:
        writer.add_scalar(tag, scalar, step)


def get_model(model_cfg: DictConfig | None,
              state_dict: dict | None, device: str) -> (torch.nn.Module, dict, DictConfig):
    if state_dict is not None:
        model_cfg = state_dict["model_configs"]
    else:
        assert model_cfg is not None

    if model_cfg.name == "regretnet":
        model = RegretNet(model_cfg.n_agents, model_cfg.n_items,
                          hidden_layer_size=model_cfg.hidden_layer_size,
                          n_hidden_layers=model_cfg.n_hidden_layers,
                          activation=model_cfg.activation)
        model = model.to(device)
    else:
        raise NotImplementedError(f"Model by name {model_cfg.name} not yet implemented.")
    if state_dict is None:
        logging.info("No state dict provided")
        optimizer_state_dict = None
    else:
        logging.info("Loading from checkpoint...")
        model.load_state_dict(state_dict["model"])
        optimizer_state_dict = state_dict["optimizer"]

    return model, optimizer_state_dict, model_cfg


def get_state_dict(model_path: str, device: str) -> dict:
    logging.info(f"Loading state_dict from checkpoint {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    return state_dict
