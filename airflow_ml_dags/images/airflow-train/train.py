import os
import pickle

import pandas as pd
import click
from sklearn.ensemble import RandomForestClassifier


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    rfc = RandomForestClassifier()
    rfc.fit(data, target.values.ravel())

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "random_forest_classifier.pickle"), "wb") as model_file:
        pickle.dump(rfc, model_file)


if __name__ == '__main__':
    train()
