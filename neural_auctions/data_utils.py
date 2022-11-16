"""
data_utils.py
Utility functions for data processing
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_dataloader(n_agents: int,
                   n_items: int,
                   num_examples: int,
                   device: str = "cpu",
                   batch_size: int = 512,
                   data_distribution: str = "uniform") -> DataLoader:
    item_ranges = preset_valuation_range(n_agents, n_items)
    data = generate_dataset_nxk(n_agents, n_items, num_examples, item_ranges, data_distribution)
    data = data.to(device)
    data = torch.utils.data.TensorDataset(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


def get_data_from_loader(model,
                         loader: DataLoader,
                         sigma: float,
                         device: str) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    payments = torch.Tensor().to(device)
    allocs = torch.Tensor().to(device)
    bids = torch.Tensor().to(device)

    for i, (batch_of_bids,) in enumerate(tqdm(loader, leave=False)):
        batch_of_bids = batch_of_bids.to(device)
        with torch.no_grad():
            batch_allocs, batch_payments = model.forward(batch_of_bids, sigma)
        # Record entire test data
        bids = torch.cat((bids, batch_of_bids), dim=0)
        payments = torch.cat((payments, batch_payments), dim=0)
        allocs = torch.cat((allocs, batch_allocs), dim=0)
    return bids, payments, allocs


def preset_valuation_range(n_agents: int, n_items: int) -> torch.Tensor:
    zeros = torch.zeros(n_agents, n_items)
    ones = torch.ones(n_agents, n_items)
    item_ranges = torch.stack((zeros, ones), dim=2).reshape(n_agents, n_items, 2)
    # item_ranges is a n_agents x n_items x 2 tensor where item_ranges[agent_i][item_j] = [lower_bound, upper_bound].
    assert item_ranges.shape == (n_agents, n_items, 2)
    return item_ranges


def generate_dataset_nxk(n_agents: int,
                         n_items: int,
                         num_examples: int,
                         item_ranges: torch.Tensor,
                         dist: str = "uniform"):
    # does not work for 2x1
    range_diff = item_ranges[:, :, 1] - item_ranges[:, :, 0]
    if dist == "normal":
        output = range_diff * torch.randn(num_examples, n_agents, n_items) + item_ranges[:, :, 0] + range_diff / 2
    elif dist == "zeros":
        output = torch.zeros(num_examples, n_agents, n_items)
    else:
        output = range_diff * torch.rand(num_examples, n_agents, n_items) + item_ranges[:, :, 0]
    return torch.clamp(output, item_ranges[:, :, 0], item_ranges[:, :, 1])


