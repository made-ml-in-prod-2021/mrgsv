import os
from shutil import copyfile

import pandas as pd
import click
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    num_cols = data.columns[(data.dtypes == "float") | (data.dtypes == "int")].values
    ss = StandardScaler()
    data.loc[:, num_cols] = ss.fit_transform(data.loc[:, num_cols])

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    copyfile(os.path.join(input_dir, "target.csv"), os.path.join(output_dir, "target.csv"))


if __name__ == '__main__':
    preprocess()
