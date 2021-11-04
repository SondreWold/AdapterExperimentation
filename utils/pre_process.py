from os import sep
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="The name of the dataset to split",
    )

    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="Split percentage (e.g 0,2)",
    )

    return parser.parse_args()


def main(args):
    SEED = 42
    np.random.seed(SEED)
    df = pd.read_csv(args.input, sep="\t")
    df_permutated = df.sample(frac=1, random_state=SEED)

    train_size = 1 - args.split
    train_end = int(len(df_permutated)*train_size)

    df_train = df_permutated[:train_end]
    df_test = df_permutated[train_end:]
    df_train.to_csv("cn_train.txt", sep="\t", index=False)
    df_test.to_csv("cn_validation.txt", sep="\t", index=False)


if __name__ == "__main__":
    main(parse_args())
