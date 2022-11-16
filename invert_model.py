"""
invert_model.py
Invert saved models for measuring privacy
Developed collaborativley by
    Alex Stein, Avi Schwarzschild, and Michael Curry
    for "Protecting Bidder Information in Neural Auctions" project
August 2022
"""

import json
import logging
import os
import sys
from collections import OrderedDict

import hydra
import torch
from almost_unique_id import generate_id
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import neural_auctions as na
from run_helpers import setup, save_results
from test_model import test_model


@hydra.main(version_base="1.2", config_path="config", config_name="invert_model_config")
def main(cfg: DictConfig):
    device, log = setup(cfg, "invert_model.py main()")
    model_path = os.path.join(cfg.load.runtime_cwd, cfg.load.model_path)
    train_run_id = model_path.split("/")[-2]
    state_dict = na.get_state_dict(model_path, device)
    model, _, model_cfg = na.get_model(None, state_dict, device)

    writer = SummaryWriter(log_dir=f"tensorboard-{cfg.run_id}")

    ####################################################
    #               Dataset and Network and Optimizer
    inversion_loader = na.get_dataloader(model_cfg.n_agents,
                                         model_cfg.n_items,
                                         cfg.hyp.test_num_examples,
                                         device,
                                         cfg.hyp.test_batch_size,
                                         cfg.hyp.test_data_generating_distribution)
    loaders = {"test": inversion_loader}
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    log.info(f"This {model_cfg.name} has {pytorch_total_params / 1e3:0.3f} thousand parameters.")
    ####################################################

    results = invert_model(model, cfg, model_cfg, loaders, writer, device)
    results["train_run_id"] = train_run_id
    writer.flush()
    writer.close()

    test_results = test_model(model, cfg, model_cfg, loaders, device)
    results["test_results"] = test_results["test_results"]
    results["hyp"] = OmegaConf.to_container(cfg.hyp)
    results["model_path"] = model_path
    results["model_cfg"] = OmegaConf.to_container(model_cfg)
    save_results(results, train_run_id)
    log.info(json.dumps(results, indent=4))


def invert_model(model: na.models.Model,
                 cfg: DictConfig,
                 model_cfg: DictConfig,
                 loaders: dict,
                 writer: SummaryWriter = None,
                 device: str = "cpu") -> OrderedDict:
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("invert_model.py invert_model() running.")

    # gather data and initialize guesses
    true_bids, true_payments, true_allocs = na.get_data_from_loader(model, loaders["test"], cfg.hyp.sigma, device)

    # intialize the inversion with random data
    input_ranges = na.data_utils.preset_valuation_range(model_cfg.n_agents, model_cfg.n_items)
    guessed_bids = na.data_utils.generate_dataset_nxk(model_cfg.n_agents,
                                                      model_cfg.n_items,
                                                      true_allocs.size()[0],
                                                      input_ranges,
                                                      cfg.hyp.privacy_data_generating_distribution)
    guessed_bids = guessed_bids.to(true_allocs.device)
    guessed_bids.requires_grad = True
    if cfg.hyp.use_bidder_info:
        with torch.no_grad():
            guessed_bids[:, 0, :] = true_bids[:, 0, :]

    # using the true bids as the initialization (This can be done for debugging):
    # guessed_bids = true_bids.clone() + torch.rand_like(true_bids)*1
    # guessed_bids.requires_grad = True

    # detach allocs and payments from graph
    true_allocs = true_allocs.detach()
    true_payments = true_payments.detach()

    # get optimizer
    optimizer = na.get_optimizer(cfg.hyp.privacy_optimizer, cfg.hyp.privacy_lr, [guessed_bids], wd=0)

    # turn off gradient tracking for model params
    for p in model.parameters():
        p.requires_grad = False

    for invert_iter in range(cfg.hyp.privacy_iters):
        invert_metrics = model.inversion_loop(cfg.hyp.sigma,
                                              cfg.hyp.privacy_alpha,
                                              optimizer,
                                              true_payments,
                                              true_allocs,
                                              guessed_bids)

        invert_log_str = f"iteration: {invert_iter:5d}"
        for key, value in invert_metrics.items():
            invert_log_str += f" | {key}: {value:.4f}"

        with torch.no_grad():
            # todo: allow for other ranges here?
            guessed_bids[:] = guessed_bids.clamp(0, 1)
            if cfg.hyp.use_bidder_info:
                guessed_bids[:, 0, :] = true_bids[:, 0, :]
                guessed_bids[:, 0, :].grad = torch.zeros_like(guessed_bids[:, 0, :])

        if (invert_iter + 1) % cfg.hyp.privacy_val_period == 0:
            recovery_rate = na.get_recovery_rate(true_bids, guessed_bids, cfg.hyp.privacy_tol, cfg.hyp.use_bidder_info)
            invert_log_str += f" | recovery rate (eps={cfg.hyp.privacy_tol}): {recovery_rate:2.2f}" \
                              f" | MAE: {torch.mean(torch.abs(true_bids - guessed_bids)).item():2.6f}"
                              # f" | grad norm: {torch.norm(guessed_bids.grad):.4f}"
            log.info(invert_log_str)

            # TensorBoard loss writing
            for key, value in invert_metrics.items():
                na.write(writer, f"Inverting/{key}", value, invert_iter)
            na.write(writer, "Inverting/grad_norm", torch.norm(guessed_bids.grad), invert_iter)
            na.write(writer, f"Inverting/recovery_rate (eps={cfg.hyp.privacy_tol})", recovery_rate, invert_iter)

    torch.save(true_bids, "true_bids.pth")
    torch.save(guessed_bids, "guessed_bids.pth")

    # print one example
    guessed_allocs, guessed_payments = model(guessed_bids, cfg.hyp.sigma)
    log.info(f"\ntrue bids:\n {true_bids[1]}")
    log.info(f"\nguessed bids:\n {guessed_bids[1]}")
    log.info(f"\ntrue allocs:\n {true_allocs[1]}")
    log.info(f"\nguessed allocs:\n {guessed_allocs[1]}")
    log.info(f"\ntrue payments:\n {true_payments[1]}")
    log.info(f"\nguessed payments:\n {guessed_payments[1]}")

    # print error values
    log.info("Error measurements:")
    log.info(f"bids error MAE: {torch.mean(torch.abs(true_bids - guessed_bids)).item():.2f}")
    log.info(f"bids error inf-norm: {torch.norm(true_bids - guessed_bids, p=float('inf')).item():.2f}")
    log.info(f"allocs error inf-norm: {torch.norm(true_allocs - guessed_allocs, p=float('inf')).item():.2f}")
    log.info(f"payments error inf-norm: {torch.norm(true_payments - guessed_payments, p=float('inf')).item():.2f}")
    log.info(f"bids_recovery_rate (eps={cfg.hyp.privacy_tol}): {recovery_rate:2.2f}")

    results = OrderedDict([("epsilon", cfg.hyp.privacy_tol),
                          ("inversion_loss", invert_metrics["loss"].item()),
                          ("payment_mean", true_payments.sum(dim=1).mean(dim=0).item()),
                          ("privacy_tol", cfg.hyp.privacy_tol),
                          ("bids_mean_inversion_error", torch.mean(torch.abs(true_bids - guessed_bids)).item()),
                          ("bids_inf_norm_inversion_error",
                           torch.norm(true_bids - guessed_bids, p=float("inf")).item()),
                          ("bids_recovery_rate", recovery_rate),
                          ("sigma", cfg.hyp.sigma),
                          ("use_bidder_info", cfg.hyp.use_bidder_info)])
    return results


if __name__ == "__main__":
    run_id = generate_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
