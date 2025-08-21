import argparse
from joblib import load
import pandas as pd


def predict(sample):
    bundle = load("models/model.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    df = pd.DataFrame([sample], columns=feature_names)
    pred = model.predict(df)[0]
    target_names = bundle["target_names"]
    return target_names[pred]


def cli():
    parser = argparse.ArgumentParser(description="Predict Iris species.")
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    args = parser.parse_args()

    sample = [
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,
    ]
    species = predict(sample)
    print(f"Predicted species: {species}")


if __name__ == "__main__":
    cli()
