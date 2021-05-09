from typing import List

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.data.make_dataset import read_data
from src.data_types import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer


@pytest.fixture
def feature_params(
        numerical_features: List[str],
        target_col: str,
        transform_type: str
) -> FeatureParams:
    params = FeatureParams(
        numerical_features=numerical_features,
        target_col=target_col,
        transform_type=transform_type,
    )
    return params


def test_make_features(
        feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()


def test_extract_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    target = extract_target(data, feature_params)
    assert_allclose(target, [0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0.,
                             1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1.,
                             1., 1., 0., 0., 0., 1.])
