"""
auction_utils.py
Functions for auction-realted calculations, like agent utility
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

import torch


def calculate_agent_util(valuations, allocs, payments):
    util_from_items = torch.sum(allocs * valuations, dim=2)
    return util_from_items - payments


def create_combined_misreports(misreports, valuations):
    n_agents = misreports.shape[1]
    n_items = misreports.shape[2]

    # mask might be a constant that could be allocated once outside
    mask = torch.zeros((misreports.shape[0], n_agents, n_agents, n_items), device=misreports.device)
    for i in range(n_agents):
        mask[:, i, i, :] = 1.0

    tiled_mis = misreports.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)
    tiled_true = valuations.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)
    return mask * tiled_mis + (1.0 - mask) * tiled_true


def tiled_misreport_util(current_misreports, current_valuations, model, sigma=0):
    n_agents = current_valuations.shape[1]
    n_items = current_valuations.shape[2]

    agent_idx = list(range(n_agents))
    tiled_misreports = create_combined_misreports(current_misreports, current_valuations)
    flatbatch_tiled_misreports = tiled_misreports.view(-1, n_agents, n_items)

    allocations, payments = model(flatbatch_tiled_misreports, sigma=sigma)

    reshaped_payments = payments.view(-1, n_agents, n_agents)
    reshaped_allocations = allocations.view(-1, n_agents, n_agents, n_items)

    # slice out or mask out agent's payments and allocations
    agent_payments = reshaped_payments[:, agent_idx, agent_idx]
    agent_allocations = reshaped_allocations[:, agent_idx, agent_idx, :]
    agent_utils = calculate_agent_util(current_valuations, agent_allocations, agent_payments)  # shape [-1, n_agents]
    return agent_utils


def get_misreports_from_bids_with_multiple_inits(model, bids, num_inits, misreport_iters, misreport_lr, sigma=0,
                                                 util_num_samples=1, return_agent_util=False):
    n_batch, n_agents, n_items = bids.size()
    big_bids = bids.repeat(num_inits, 1, 1)

    # if the model is noisy, we need to pass each initialization through the net num_samples times to average over
    # samples of the noise.
    num_samples = util_num_samples if sigma > 0 else 1

    current_misreports = torch.rand_like(big_bids)
    big_bids = big_bids.repeat(num_samples, 1, 1)

    for i in range(misreport_iters):
        current_misreports.requires_grad_(True)
        big_misreports = current_misreports.repeat(num_samples, 1, 1)
        model.zero_grad()
        agent_utils = tiled_misreport_util(big_misreports, big_bids, model, sigma=sigma)

        (misreports_grad,) = torch.autograd.grad(agent_utils.sum(), big_misreports)

        with torch.no_grad():
            # average over num_samples
            current_misreports += misreport_lr * misreports_grad.view(num_samples, n_batch * num_inits, n_agents,
                                                                      n_items).mean(dim=0, keepdim=False)
            current_misreports = model.clamp_op(current_misreports)

    agent_utils = agent_utils.view(num_samples, num_inits, n_batch, n_agents).mean(dim=0, keepdim=False)
    reshaped_misreports = current_misreports.view(num_inits, n_batch, n_agents, n_items)

    # get argmax of the util over the dimension that represents num_initis
    arg_max = agent_utils.argmax(dim=0, keepdim=True)

    # add dimension for items
    arg_max = arg_max[..., None]

    # repeat index to fill out the item dimension to the proper shape
    arg_max = arg_max.repeat(1, 1, 1, n_items)

    # use gather
    best_misreports = torch.gather(reshaped_misreports, 0, arg_max).squeeze(dim=0)

    if return_agent_util:
        return best_misreports, agent_utils.max(dim=0, keepdim=False)[0]
    else:
        return best_misreports


def myerson(n_agents, n_items):
    return n_items * ((-1 + 2 ** (-n_agents) + n_agents) / (1 + n_agents))


def get_random_recovery_rate(n_agents, n_items, privacy_tol):
    a = torch.rand((500000, int(n_agents), int(n_items)))
    b = torch.rand((500000, int(n_agents), int(n_items)))
    return 100.0 * (torch.abs(a - b) < privacy_tol).sum() / b.numel()


def get_recovery_rate(true_bids: torch.Tensor, guessed_bids: torch.Tensor, privacy_tol: float,
                      use_bidder_info: bool) -> float:
    if use_bidder_info:
        recovery_rate = 100.0 * (
                torch.abs(true_bids[:, 1:, :] - guessed_bids[:, 1:, :]) < privacy_tol).sum() / true_bids[:, 1:,
                                                                                               :].numel()
    else:
        recovery_rate = 100.0 * (torch.abs(true_bids - guessed_bids) < privacy_tol).sum() / true_bids.numel()
    return recovery_rate.item()
