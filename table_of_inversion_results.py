"""
table_of_inversion_results.py
for making tables of results from multiple inversion runs
August 2022
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from neural_auctions import myerson, get_random_recovery_rate, get_recovery_rate


def get_row_in_df(results_dir, model_name: str, tol):
    with open(os.path.join(results_dir, f"{model_name}_result.json"), "r") as fh:
        result = json.load(fh)
    guessed_bid_f_name = os.path.join(results_dir, "guessed_bids.pth")
    true_bid_f_name = os.path.join(results_dir, "true_bids.pth")
    guessed_bids = torch.load(guessed_bid_f_name, map_location="cpu")
    true_bids = torch.load(true_bid_f_name, map_location="cpu")
    true_recover_rate = get_recovery_rate(true_bids, guessed_bids, tol, result["use_bidder_info"])
    result["n_agents"] = true_bids.size(1) #+ result["use_bidder_info"]
    result["n_items"] = true_bids.size(2)
    result["true_recover_rate"] = true_recover_rate
    result["random_recover_rate"] = get_random_recovery_rate(result["n_agents"], result["n_items"], tol).item()
    result["myerson"] = myerson(result["n_agents"], result["n_items"])
    result["count"] = 1
    result["name"] = results_dir.split("/")[-1]
    hyp = result["hyp"]
    result["privacy_lr"] = hyp["privacy_lr"]
    test_result = result["test_results"]
    result["misreport_iters"] = hyp["test_misreport_iters"]
    result["payment"] = test_result["payments_mean"]
    result["regret"] = test_result["regret_mean"]
    result["regret_max"] = test_result["regret_max"]
    model_cfg = result["model_cfg"]
    result["train_misr_inits"] = model_cfg["train_misreport_inits"]
    return result


def get_df(filepath, tol):
    df = pd.DataFrame()
    idx = 0
    for d_dir in glob.iglob(f"{filepath}/*/*/*_result.json", recursive=True):
        model_name = Path(d_dir).parent.name
        d_dir = str(Path(d_dir).parent.absolute())
        new_row_dict = get_row_in_df(d_dir, model_name, tol)
        parent_name = str(Path(Path(d_dir).parent).name)
        new_row_dict["parent_name"] = parent_name
        df = pd.concat([df, pd.DataFrame.from_dict({idx: new_row_dict}, orient="index")])
        idx += 1
    return df


def get_table(filepath, args):
    disp_max = args.max
    disp_min = args.min
    no_sem = args.no_sem
    pivot = args.pivot
    tol = args.privacy_tol
    pd.set_option("display.max_rows", None)
    df = get_df(filepath, tol)
    print(df.columns)
    df = df.sort_values(["n_agents", "n_items", "use_bidder_info", "sigma"])
    df = df[["name",
             "n_agents",
             "n_items",
             "privacy_lr",
             "train_misr_inits",
             "sigma",
             "use_bidder_info",
             "true_recover_rate",
             "random_recover_rate",
             "misreport_iters",
             "payment",
             "myerson",
             "regret",
             "regret_max",
             "bids_mean_inversion_error",
             "count",
             "parent_name"]]

    # index = ["n_agents", "n_items", "use_bidder_info", "train_misr_inits", "privacy_lr", "sigma"]
    index = ["n_agents", "n_items", "use_bidder_info", "sigma"]
    values = ["mean", "sem"] if not no_sem else ["mean"]
    if disp_max:
        values.append("max")
    if disp_min:
        values.append("min")

    if pivot:
        try:
            df = pd.pivot_table(df, index=index, aggfunc={"true_recover_rate": values,
                                                          "random_recover_rate": values,
                                                          "bids_mean_inversion_error": values,
                                                          "payment": values,
                                                          "regret": values,
                                                          "regret_max": values,
                                                          "count": "count"})
        except:
            raise KeyError("No data in the table. Check the path to test results.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values too table?")
    parser.add_argument("--no-sem", action="store_true", help="add min values too table?")
    parser.add_argument("--pivot", action="store_true", help="compute pivot table?")
    parser.add_argument("--privacy-tol", type=float, default=2e-2)

    args = parser.parse_args()
    table = get_table(args.filepath, args)
    table = table.round(4)
    # print(table.to_markdown())
    print(table)
    table.columns = table.columns.map("_".join)
    table.columns.name = None
    table = table.reset_index()
    table.to_csv(f"table{datetime.now().strftime('%m%d-%H.%M')}.csv")


if __name__ == "__main__":
    main()
