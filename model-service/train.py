from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


def main():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(iris.data, iris.target)

    model_bundle = {
        "model": model,
        "target_names": iris.target_names.tolist(),
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model_bundle, model_file)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
