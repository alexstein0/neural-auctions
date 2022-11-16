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


def get_row_in_df(results_dir, model_name: str):
    with open(os.path.join(results_dir, f"{model_name}_result.json"), "r") as fh:
        result = json.load(fh)
    result["n_agents"] = result["model_cfg"]["n_agents"]
    result["n_items"] = result["model_cfg"]["n_items"]
    result["myerson"] = myerson(result["n_agents"], result["n_items"])
    result["count"] = 1
    result["name"] = results_dir.split("/")[-1]
    hyp = result["hyp"]
    test_result = result["test_results"]
    result["misreport_iters"] = hyp["test_misreport_iters"]
    result["misreport_inits"] = hyp["test_misreport_inits"]
    result["misreport_lr"] = hyp["test_misreport_lr"]
    result["test_payment"] = test_result["payments_mean"]
    result["test_regret_mean"] = test_result["regret_mean"]
    result["test_regret_max"] = test_result["regret_max"]
    result["test_util_difference"] = test_result["util_difference"]
    result["test_util_better"] = test_result["util_better"]
    return result


def get_df(filepath):
    df = pd.DataFrame()
    idx = 0
    for d_dir in glob.iglob(f"{filepath}*/*/*/*_result.json", recursive=True):
        model_name = Path(d_dir).parent.name
        d_dir = str(Path(d_dir).parent.absolute())
        new_row_dict = get_row_in_df(d_dir, model_name)
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
    pd.set_option("display.max_rows", None)
    df = get_df(filepath)
    df = df.sort_values(["n_agents", "n_items"])
    df = df[["name",
             "n_agents",
             "n_items",
             "misreport_inits",
             "misreport_iters",
             "misreport_lr",
             "myerson",
             "test_regret_mean",
             "test_regret_max",
             "test_payment",
             "test_util_difference",
             "test_util_better",
             "count",
             "parent_name"]]

    index = ["n_agents", "n_items", "misreport_inits", "misreport_iters", "misreport_lr"]
    values = ["mean", "sem"] if not no_sem else ["mean"]
    if disp_max:
        values.append("max")
    if disp_min:
        values.append("min")

    if pivot:
        try:
            df = pd.pivot_table(df, index=index, aggfunc={
                                                          "test_regret_mean": values,
                                                          "test_regret_max": values,
                                                          "test_payment": values,
                                                          "test_util_difference": values,
                                                          "test_util_better": values,
                                                          "count": "count"
                                                          }
                                )
        except:
            raise KeyError("No data in the table. Check the path to test results.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("filepath", type=str)
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values to table?")
    parser.add_argument("--no-sem", action="store_true", help="add sem values to table?")
    parser.add_argument("--pivot", action="store_true", help="compute pivot table?")

    args = parser.parse_args()
    table = get_table(args.filepath, args)
    table = table.round(4)
    # print(table.to_markdown())
    print(table)
    table.to_csv(f"table{datetime.now().strftime('%m%d-%H.%M')}.csv")


if __name__ == "__main__":
    main()
