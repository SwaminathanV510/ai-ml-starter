from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd

app = FastAPI(title="Iris Classifier API", version="0.1.0")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def _load_bundle():
    return load("models/model.joblib")


@app.get("/")
def root():
    return {"status": "ok", "message": "Use /predict or /docs"}


@app.post("/predict")
def predict(features: IrisFeatures):
    bundle = _load_bundle()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    df = pd.DataFrame(
        [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]],
        columns=feature_names,
    )
    y = model.predict(df)[0]
    species = bundle["target_names"][y]
    return {"species": species}
