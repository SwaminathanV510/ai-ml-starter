from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from pathlib import Path
from .data import load_data


def main():
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()

    clf = RandomForestClassifier(random_state=42, n_estimators=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.3f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=target_names))

    # Save model
    Path("models").mkdir(exist_ok=True, parents=True)
    dump(
        {
            "model": clf,
            "feature_names": feature_names,
            "target_names": target_names,
        },
        "models/model.joblib",
    )
    print("\nSaved model to models/model.joblib")


if __name__ == "__main__":
    main()
