import json
import sys
from pathlib import Path


MODEL_PATH = "model.joblib"
METADATA_PATH = "model_meta.json"


MIN_TRAINING_ROWS = 100
MAX_MODEL_SIZE_MB = 50


def fail(msg: str, code: int = 1):
    print(f"[MODEL SAFETY CHECK FAILED] {msg}")
    sys.exit(code)


def run_check(model_path: str = MODEL_PATH, meta_path: str = METADATA_PATH):
    meta_p = Path(meta_path)
    model_p = Path(model_path)
    if not meta_p.exists():
        fail("model_meta.json missing")

    if not model_p.exists():
        fail("model.joblib missing")
    
    try:
        meta = json.loads(meta_p.read_text())
    except Exception as e:
        fail(f"Failed to parse metadata: {e}")

    if "has_pii" not in meta:
        fail("Metadata missing has_pii flag")

    if meta["has_pii"]:
        fail("PII present as indicated by model metadata")

    if "n_features" not in meta or "feature_names" not in meta:
        fail("Metadata missing feature definition")

    if meta["n_features"] != len(meta["feature_names"]):
        fail("There is a mismatch in feature count between n_features and feature_names")
 
    if meta.get("training_rows", 0) < MIN_TRAINING_ROWS: 
        fail("Insufficient training data rows")
   

    size_mb = model_p.stat().st_size / (1024*1024)
    if size_mb > MAX_MODEL_SIZE_MB:
        fail(f"Model is too large ({size_mb:.1f} MB)")

    print("Model safety checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run_check())