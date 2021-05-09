import os
import random
import logging
import sys

from typing import List
from yaml import safe_load

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multinomial


# test_logger = logging.getLogger(__name__)
# test_handler = logging.StreamHandler(sys.stdout)
# test_logger.setLevel(logging.INFO)
# test_logger.addHandler(test_handler)


def make_fake_dataset(path_to_dataset: str, path_to_sample_params: str, sample_size=50) -> pd.DataFrame:
    with open(path_to_sample_params, "r") as file:
        multinomial_params = safe_load(file)

    num_cols = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]
    cat_cols = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "target"
    ]
    df = pd.read_csv(path_to_dataset)
    random.seed(1234)

    def generate_multinomial_samples(params: dict, n: int = 10):
        counts = multinomial.rvs(n=n, p=list(params.values()), random_state=1234)
        samples = []
        for idx, val in enumerate(params):
            samples.extend([val for _ in range(counts[idx])])
        random.shuffle(samples)
        return samples

    def generate_numerical_samples(df_, n: int = 10):
        samples = []
        for col in num_cols:
            samples.append(random.choices(df_.loc[:, col].values, k=n))
        return samples

    def generate_random_samples_from_data(df_, cat_cols_, num_cols_, n=50):
        for col in cat_cols_:
            multinomial_params[col] = dict(df_.loc[:, col].value_counts() / (len(df_)))
        cat_samples = [generate_multinomial_samples(params, n=n) for params in multinomial_params.values()]
        num_samples = generate_numerical_samples(df_, n=n)
        num_samples.extend(cat_samples)
        return pd.DataFrame(np.array(num_samples).T, columns=num_cols_ + cat_cols_)

    out_samples = generate_random_samples_from_data(df, cat_cols, num_cols, sample_size)
    return out_samples


# @pytest.fixture
# def path_to_dataset():
#     return "data/raw/train.csv"
#
#
# @pytest.fixture
# def path_to_sample_params():
#     return "configs/sample_params.yaml"


@pytest.fixture
def dataset_path():
    curdir = os.path.dirname(__file__)
    # if not os.path.exists(os.path.join(curdir, "train_data_sample.csv")):
    df = make_fake_dataset("data/raw/train.csv", "tests/configs/sample_params.yaml")
    df.to_csv(os.path.join(curdir, "train_data_sample.csv"), index=False)
    # test_logger.info(f"fake dataset: {df.loc[:, 'target'].values}")
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture
def target_col():
    return "target"

@pytest.fixture
def transform_type():
    return "gaussian map"


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]


