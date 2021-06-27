import os
import pickle

import pandas as pd
import click
from airflow.models import Variable


@click.command("predict")
@click.option("--input-dir")
@click.option("--path-to-model")
@click.option("--output-dir")
def predict(input_dir: str, output_dir: str, path_to_model: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    with open(path_to_model, "rb") as model_file:
        model = pickle.load(model_file)

    predicts = model.predict(data)
    predicts = pd.DataFrame(predicts)
    os.makedirs(output_dir, exist_ok=True)
    predicts.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
