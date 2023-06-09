#!/usr/bin/env python3

"""
This script keeps track of the best results obtained for each training iteration
of a given model and computes the average of the relevant statistics.

Author: Matteo Brivio [ matteo.brivio@student.uni-tuebingen.de ]
"""


import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate


def compute_stats(reports):
    """Given a list storing the best classification metrics for each training
    iteration over n epochs, computes average and standard deviation for accuracy,
    macro- and weighted-F1 values, respectively.

    Args:
        reports: a list of the best metrics for each training iteration.
    Return:
        A list of lists, storing average and standard deviation for accuracy,
        macro- and weighted-F1 values
    """
    accuracy, weigh_f1, macro_f1 = [], [], []
    
    for report in reports:
        accuracy.append(report.get("accuracy"))
        macro_f1.append(report.get("macro avg").get("f1-score"))
        weigh_f1.append(report.get("weighted avg").get("f1-score"))

    stats = [
        ["accuracy", np.mean(accuracy), np.std(accuracy)],
        ["weighted f1", np.mean(weigh_f1), np.std(weigh_f1)],
        ["macro f1", np.mean(macro_f1), np.std(macro_f1)]]

    return stats


def generate_final_report(args, evl_vals, epochs, reports, name, time_vals):
    """Generates a txt file storing the best classification metrics for each training
    iteration as well as average and standard deviation for accuracy, macro- and
    weighted-F1 values.

    Args:
        args: list of arguments passed to train.py.
        evl_vals: list of metrics for each training iteration of n epochs.
        epochs: number of epochs covered by each training iteration.
        reports: a list of the best metrics for each training iteration.
        name: name of the data-set being used.
        time_vals: list of training time values.
    Return:
        None
    """

    data_path = Path(name)
    report_path = Path(Path.cwd() / f"report_{data_path.parent.name}_{data_path.stem}_{args.batch_size}_{args.learn_rate}_{args.dropout_prob}.txt")

    with open(report_path, "a") as report_file:
        
        for iter, report in enumerate(reports, start=1):
            df_report = pd.DataFrame(report)
            df_report = df_report.transpose()
            report_file.write(f"ITERATION: {iter} | BEST REPORT OUT OF {epochs} EPOCHS\n")
            report_file.write(df_report.to_string()+"\n\n")
        
        # adds mean and std. deviation to the report
        report_file.write(f"\nAVERAGE AND STD. DEVIATION\n\n")
        stats = tabulate(compute_stats(reports), headers=["average", "std. deviation"])
        report_file.write(stats+"\n\n")
       
        for i, evl in enumerate(evl_vals):
            report_file.write(f"Iteration {i}, best f1: {evl.best_test_f1}\n")
            report_file.write(f"Iteration {i}, best acc: {evl.best_acc}\n")
            report_file.write(f"Iteration {i}, best epoch: {evl.best_test_epoch}\n")

        # adds training time for current iteration
        for t in time_vals: report_file.write(str(t)+"\n")
