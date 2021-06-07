import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=13, max_items=13)]
    features: List[str]


class HeartDiseaseResponse(BaseModel):
    target: int


model: Optional[Pipeline] = None


def make_predict(
        data: List, features: List[str], model: Pipeline
) -> List[HeartDiseaseResponse]:
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)

    return [
        HeartDiseaseResponse(target=target) for target in predicts
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL",
                           default="../models/model_random_forest_classifier.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel):
    global model
    if request.features != FEATURE_ORDER:
        raise HTTPException(status_code=400, detail=f"Expected features in this order: {FEATURE_ORDER}")
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
