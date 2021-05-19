import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from ..data_types.train_params import TrainingParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnClassifierModel, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "f1_score": f1_score(y_true=target, y_pred=predicts),
        "roc_auc_score": roc_auc_score(y_true=target, y_score=predicts),
        "accuracy_score": accuracy_score(y_true=target, y_pred=predicts),
    }


def create_inference_pipeline(
    model: SklearnClassifierModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: SklearnClassifierModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
