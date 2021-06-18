import os
import pickle
import json

import pandas as pd
import click
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


@click.command("evaluate")
@click.option("--path-to-model")
@click.option("--path-to-val-data")
@click.option("--output-dir")
def evaluate(path_to_model: str, path_to_val_data: str, output_dir: str):
    data = pd.read_csv(os.path.join(path_to_val_data, "val_data.csv"))
    target = pd.read_csv(os.path.join(path_to_val_data, "val_target.csv")).values.ravel()

    with open(os.path.join(path_to_model, "random_forest_classifier.pickle"), "rb") as model_file:
        model = pickle.load(model_file)

    predicts = model.predict(data)
    metrics = {"f1_score": f1_score(y_true=target, y_pred=predicts, average="weighted"),
               "accuracy_score": accuracy_score(y_true=target, y_pred=predicts), }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == '__main__':
    evaluate()
