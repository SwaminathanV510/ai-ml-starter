from src.data import load_data


def test_load_data_shapes():
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    # Basic sanity checks
    assert X_train.shape[1] == 4
    assert len(feature_names) == 4
    assert len(target_names) == 3
