import joblib
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_and_train(out_model="model.joblib", out_meta="model_meta.json"):
    X, y = make_classification(n_samples=5000, n_features=10, n_informative=6, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df.to_csv("data/synthetic.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df['label'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "features": feature_names}, out_model)

    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "training_rows": len(df),
        "has_pii": False,          # important field for model safety check
        "source": "synthetic",
        "created_by": "train.py"
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    generate_and_train()