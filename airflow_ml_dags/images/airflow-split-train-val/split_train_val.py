import os

import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split_train_val")
@click.option("--input-dir")
@click.option("--train-data-dir")
@click.option("--val-data-dir")
def split_train_val(input_dir: str, train_data_dir: str, val_data_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(train_data_dir, "train_data.csv"), index=False)
    X_val.to_csv(os.path.join(val_data_dir, "val_data.csv"), index=False)
    y_train.to_csv(os.path.join(train_data_dir, "train_target.csv"), index=False)
    y_val.to_csv(os.path.join(val_data_dir, "val_target.csv"), index=False)


if __name__ == '__main__':
    split_train_val()
