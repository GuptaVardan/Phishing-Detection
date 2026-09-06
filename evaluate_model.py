from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "Artifacts"
FINAL_MODEL_DIR = ROOT / "final_model"


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def find_latest_file(pattern):
    files = list(ARTIFACTS_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


# The repository currently stores the model in final_model/.
model_path = find_latest_file("**/model.pkl") or FINAL_MODEL_DIR / "model.pkl"
preprocessor_path = (
    find_latest_file("**/preprocessor.pkl")
    or find_latest_file("**/preprocessing.pkl")
    or FINAL_MODEL_DIR / "preprocessor.pkl"
)

test_csv = find_latest_file("**/data_ingestion/ingested/test.csv")

if test_csv is not None:
    test_data = pd.read_csv(test_csv)

    if "Result" not in test_data.columns:
        raise ValueError(f"Target column 'Result' not found in {test_csv}")

    X_test = test_data.drop(columns=["Result"])
    y_test = test_data["Result"].replace(-1, 0).astype(int)

    model = load_pickle(model_path)
    preprocessor = load_pickle(preprocessor_path)

    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)

    dataset_description = str(test_csv)

else:
    test_npy = find_latest_file("**/data_transformation/transformed/test.npy")

    if test_npy is None:
        raise FileNotFoundError("No test.csv or test.npy found under Artifacts/")

    test_array = np.load(test_npy)

    X_test = test_array[:, :-1]
    y_test = test_array[:, -1].astype(int)

    model = load_pickle(model_path)
    y_pred = model.predict(X_test)

    dataset_description = str(test_npy)


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)

print(f"Model: {model_path}")
print(f"Test data: {dataset_description}")
print(f"Accuracy:  {accuracy:.10f}")
print(f"F1-score:  {f1:.10f}")
print(f"Precision: {precision:.10f}")