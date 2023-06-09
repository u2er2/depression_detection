#!/usr/bin/env python3

"""
This script generates the k-fold cross-validation splits for the DV data-set.

Author: Matteo Brivio [ matteo.brivio@student.uni-tuebingen.de ]
"""


import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold


def get_args():
    parser = argparse.ArgumentParser(description="splits a csv data-set into k-folds")
    parser.add_argument("input_path", type=Path, help="path to the csv data-set")
    parser.add_argument("k", type=int, help="number of splits (must be at least 2)")
    parser.add_argument("output_path", type=Path, nargs="?", default=Path.cwd(),
        help="path to the directory storing the csv splits (default: current directory)")

    return parser.parse_args()


def split_into_kfolds(input_path, folds, output_path):
    """Partitions the given data-set into k subsamples (folds). Each fold is intended
    to be used once for testing while the remaining k - 1 folds are used for training.
    This results in k test-sets and k training-sets which are stored as csv files.

    Args:
        input_path: path to the csv data-set to be split.
        k: number of folds.
        output_path: path to the directory storing test- and training-sets.
    Return:
        None
    """
    df = pd.read_csv(input_path)
    kfold = KFold(n_splits=folds)
    current_fold = 1

    for train, test in kfold.split(df):
        train_slice = df.iloc[train, 1:]
        test_slice = df.iloc[test, 1:]
        train_slice.to_csv(Path(output_path / f"train_{current_fold}.csv"))
        test_slice.to_csv(Path(output_path / f"test_{current_fold}.csv"))
        current_fold += 1


def main():
    args = get_args()
    split_into_kfolds(args.input_path, args.k, args.output_path)


if __name__ == "__main__":
    main()
