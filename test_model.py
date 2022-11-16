"""
test_model.py
Test saved models
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
from almost_unique_id import generate_id
from omegaconf import DictConfig, OmegaConf

import neural_auctions as na
from run_helpers import setup, save_results


@hydra.main(version_base="1.2", config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    device, log = setup(cfg, "test_model.py main()")

    ####################################################
    #               Dataset and Network and Optimizer
    model_path = os.path.join(cfg.load.runtime_cwd, cfg.load.model_path)
    train_run_id = model_path.split("/")[-2]
    state_dict = na.get_state_dict(model_path, device)
    model, _, model_cfg = na.get_model(None, state_dict, device)

    test_loader = na.get_dataloader(model_cfg.n_agents, model_cfg.n_items, cfg.hyp.test_num_examples, device,
                                    cfg.hyp.test_batch_size, cfg.hyp.test_data_generating_distribution)
    loaders = {"test": test_loader}
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    log.info(f"This {model_cfg.name} has {pytorch_total_params / 1e3:0.3f} thousand parameters.")
    ####################################################

    results = test_model(model, cfg, model_cfg, loaders, device)
    results["model_path"] = model_path
    results["train_run_id"] = train_run_id
    results["hyp"] = OmegaConf.to_container(cfg.hyp)
    results["model_cfg"] = OmegaConf.to_container(model_cfg)
    save_results(results, train_run_id)
    log.info(json.dumps(results, indent=4))


def test_model(model: na.models.Model,
               cfg: DictConfig,
               model_cfg: DictConfig,
               loaders: dict,
               device: str) -> OrderedDict:
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("test_model.py test_model() running.")

    testing_metrics = model.test_loop_sigma(loaders["test"],
                                            cfg.hyp.test_misreport_iters,
                                            cfg.hyp.test_misreport_lr,
                                            cfg.hyp.test_misreport_inits,
                                            cfg.hyp.util_num_samples,
                                            cfg.hyp.sigma,
                                            device)

    test_log_str = f"test sigma: {cfg.hyp.sigma:.4f} | "
    for key, value in testing_metrics.items():
        test_log_str += f" | {key}: {value:.4f}"
    log.info(test_log_str)

    # save some accuracy stats (can be used without testing to discern which models trained)
    result = OrderedDict([("run_id", cfg.run_id),
                          ("model", model_cfg.name),
                          ("sigma", cfg.hyp.sigma),
                          ("test_results", testing_metrics)
                          ])

    return result


if __name__ == "__main__":
    run_id = generate_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
