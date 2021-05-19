import os
from typing import List

from py._path.local import LocalPath

from train_pipeline import run_train_pipeline
from src.data_types import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    numerical_features: List[str],
    target_col: str,
    transform_type: str,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            target_col=target_col,
            transform_type=transform_type,
        ),
        train_params=TrainingParams(model_type="RandomForestClassifier"),
    )
    real_model_path, metrics = run_train_pipeline(params)
    assert metrics["f1_score"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
