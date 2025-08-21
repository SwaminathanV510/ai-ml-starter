from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(test_size: float = 0.2, random_state: int = 42):
    data = load_iris(as_frame=True)
    X = data.data  # DataFrame with feature names
    y = data.target  # Series of labels
    feature_names = list(X.columns)
    target_names = list(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (X_train, X_test, y_train, y_test, feature_names, target_names)
