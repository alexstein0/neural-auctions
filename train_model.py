"""
train_model.py
Train, test, and save models
August 2022
"""

import json
import logging
import sys
from collections import OrderedDict

import hydra
import numpy as np
import torch
from almost_unique_id import generate_id
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import neural_auctions as na
from run_helpers import setup, save_results


@hydra.main(version_base="1.2", config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    device, log = setup(cfg, "train_model.py main()")
    writer = SummaryWriter(log_dir=f"tensorboard-{cfg.run_id}-{cfg.exp_str}")

    ####################################################
    #               Dataset and Network and Optimizer
    train_loader = na.get_dataloader(cfg.model.n_agents, cfg.model.n_items, cfg.hyp.train_num_examples, device,
                                     cfg.hyp.train_batch_size, cfg.hyp.train_data_generating_distribution)
    test_loader = na.get_dataloader(cfg.model.n_agents, cfg.model.n_items, cfg.hyp.test_num_examples, device,
                                    cfg.hyp.test_batch_size, cfg.hyp.test_data_generating_distribution)
    loaders = {"train": train_loader, "test": test_loader}
    model, optimizer_state_dict, _ = na.get_model(cfg.model, None, device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    log.info(f"This {cfg.model.name} has {pytorch_total_params / 1e3:0.3f} thousand parameters.")
    ####################################################

    results = train_model(model, cfg, loaders, device, writer, optimizer_state_dict)
    results["hyp"] = OmegaConf.to_container(cfg.hyp)
    results["experiment_description"] = cfg.exp_str

    save_results(results)

    log.info(json.dumps(results, indent=4))
    writer.flush()
    writer.close()


def train_model(model: na.models.Model,
                cfg: DictConfig,
                loaders: dict,
                device: str,
                writer: SummaryWriter = None,
                optimizer_state_dict: dict = None) -> OrderedDict:
    optimizer, warmup_scheduler, lr_scheduler, start_epoch = na.get_optimizer_and_schedulers(cfg.hyp,
                                                                                             model.parameters(),
                                                                                             optimizer_state_dict)
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py train_model() running.")

    log.info(f"Training will start at epoch {start_epoch}.")
    log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
    # todo: look into stopping conditions that are specific to auction training (i.e. regret dependent)
    # highest_val_acc_so_far = -1
    best_so_far = False
    testing_metrics = {}

    for epoch in range(start_epoch, cfg.hyp.epochs):
        train_metrics = model.train_loop(loaders["train"], cfg.model, optimizer, warmup_scheduler, lr_scheduler, device)

        loss = train_metrics["train_loss"]
        train_log_str = f"epoch: {epoch:4d}"
        for key, value in train_metrics.items():
            train_log_str += f" | {key}: {value:.4f}"
            na.write(writer, f"Training/{key}", value, epoch)
        log.info(train_log_str)

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")
        # val_acc = na.test(net, [loaders["val"]], device)[0]
        # if val_acc > highest_val_acc_so_far:
        #     best_so_far = True
        #     highest_val_acc_so_far = val_acc
        # log.info(f"Training loss at epoch {epoch}: {loss:.4}")

        for i in range(len(optimizer.param_groups)):
            na.write(writer, f"Learning_rate/group{i}",
                     optimizer.param_groups[i]["lr"],
                     epoch)

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.hyp.test_val_period == 0 or epoch + 1 == cfg.hyp.epochs:
            # test accuracy
            testing_metrics = model.test_loop(loaders["test"],
                                              cfg.hyp.test_misreport_iters,
                                              cfg.hyp.test_misreport_lr,
                                              cfg.hyp.test_misreport_inits,
                                              cfg.hyp.sigma,
                                              device)
            test_log_str = f"epoch: {epoch:4d}"
            for key, value in testing_metrics.items():
                test_log_str += f" | {key}: {value:.4f}"
                na.write(writer, f"Accuracy/{key}", value, epoch)
            log.info(test_log_str)

        # check to see if we should save
        save_now = (epoch + 1) % cfg.hyp.save_period == 0 or \
                   (epoch + 1) == cfg.hyp.epochs or best_so_far
        if save_now:
            state = {"model": model.state_dict(),
                     "epoch": epoch,
                     "optimizer": optimizer.state_dict(),
                     "model_configs": cfg.model,
                     "experiment_description": cfg.exp_str}
            # out_str = f"model_best.pth" if best_so_far else f"model_{epoch}.pth"
            out_str = f"model_final.pth" if (epoch + 1) == cfg.hyp.epochs else f"model_{epoch}.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)

    # save some accuracy result (can be used without testing to discern which models trained)
    results = OrderedDict([("run_id", cfg.run_id),
                           ("model", cfg.model.name),
                           ("test_results", testing_metrics)
                           ])

    return results


if __name__ == "__main__":
    run_id = generate_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
