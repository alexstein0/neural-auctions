"""
table_of_training_results.py
for making tables of results from multiple training runs
August 2022
"""

import argparse
import glob
import json
import os

import pandas as pd
from omegaconf import OmegaConf

from neural_auctions import myerson


def get_df(filepath):
    pd.set_option("display.max_rows", None)
    df = pd.DataFrame()
    idx = 0
    for f_name in glob.iglob(f"{filepath}/**/result.json", recursive=True):
        with open(f_name, "r") as fp:
            result = json.load(fp)
            print(json.dumps(result, indent=4))

        folder = os.path.join(*f_name.split("/")[:-1])
        cfg = OmegaConf.load(f"{folder}/.hydra/config.yaml")
        result["n_agents"] = cfg.model.n_agents
        result["n_items"] = cfg.model.n_items
        result["initial_rho"] = cfg.model.rho
        result["initial_regret_weight"] = cfg.model.regret_weight
        result["train_misreport_iters"] = cfg.model.train_misreport_iters
        result["train_misreport_inits"] = cfg.model.train_misreport_inits
        result.update(cfg.hyp)
        test_result = result["test_results"]
        result["payment"] = test_result["payments_mean"]
        result["regret"] = test_result["regret_mean"]
        result["regret_max"] = test_result["regret_max"]
        result["myerson"] = myerson(result["n_agents"], result["n_items"])
        result["name"] = f_name.split("/")[-2]

        df = pd.concat([df, pd.DataFrame.from_dict({idx: result}, orient="index")])
        idx += 1
    return df


def get_table(filepath, disp_max, disp_min, no_sem, pivot):
    pd.set_option("display.max_rows", None)
    df = get_df(filepath)
    df["count"] = 1
    df = df.sort_values(["n_agents", "n_items"])
    df = df[["run_id",
             "n_agents",
             "n_items",
             "train_misreport_inits",
             "train_misreport_iters",
             "payment",
             "myerson",
             "regret",
             "regret_max",
             "count"]]

    index = ["n_agents", "n_items", "train_misreport_inits"]
    values = ["mean", "sem"] if not no_sem else ["mean"]
    if disp_max:
        values.append("max")
    if disp_min:
        values.append("min")

    if pivot:
        try:
            df = pd.pivot_table(df, index=index, aggfunc={"myerson": "mean",
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
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values too table?")
    parser.add_argument("--no-sem", action="store_true", help="add min values too table?")
    parser.add_argument("--pivot", action="store_true", help="compute pivot table?")

    args = parser.parse_args()
    table = get_table(args.filepath, args.max, args.min, args.no_sem, args.pivot)
    table = table.round(4)
    print(table.to_markdown())
    print(table)


if __name__ == "__main__":
    main()
