import joblib
from pathlib import Path
def test_model_and_meta_exist():
    assert Path("model.joblib").exists()
    assert Path("model_meta.json").exists()

def test_model_predicts():
    data = joblib.load("model.joblib")
    model = data["model"]
    # simple smoke test
    import numpy as np
    arr = np.zeros((1, len(data["features"])))
    res = model.predict(arr)
    assert len(res) == 1