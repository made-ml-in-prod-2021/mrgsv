import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.data_types.feature_params import FeatureParams


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline(transform_type: str) -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("standard scale", StandardScaler()) if transform_type == "standard scale"
            else ("gaussian map", PowerTransformer(standardize=True)),
        ],
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(params.transform_type),
                params.numerical_features,
            ),
        ],
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
