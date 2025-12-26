import json
import os
import joblib

MODEL_PATH = "model.joblib"
METADATA_PATH = "model_meta.json"


class ModelLoadError(RuntimeError):
    """Raised when the model or metadata cannot be loaded safely."""
    pass


def load_metadata(path: str = METADATA_PATH) -> dict:
    """
    Load model metadata from disk.

    This metadata is used to validate compatibility
    between the trained model and inference inputs.
    """
    if not os.path.exists(path):
        raise ModelLoadError(f"metadata file not found: {path}")

    try:
        with open(path, "r") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ModelLoadError(f"failed to load metadata: {e}") from e

    if "n_features" not in metadata:
        raise ModelLoadError("metadata missing required field: n_features")

    return metadata


def load_model(path: str = MODEL_PATH):
    """
    Load the trained model artefact from disk.
    """
    if not os.path.exists(path):
        raise ModelLoadError(f"model file not found: {path}")

    try:
        model = joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"failed to load model: {e}") from e

    if not hasattr(model, "predict"):
        raise ModelLoadError("loaded object does not implement predict()")

    return model


def validate_model_compatibility(model, metadata: dict) -> None:
    """
    Perform basic safety checks before allowing inference.
    """
    n_features = metadata.get("n_features")

    if not isinstance(n_features, int) or n_features <= 0:
        raise ModelLoadError("invalid n_features value in metadata")

    # Optional defensive check for common sklearn models
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != n_features:
            raise ModelLoadError(
                "model feature count does not match metadata"
            )


def load_model_and_metadata():
    """
    Single entry point used by the API and tests.
    """
    metadata = load_metadata()
    model = load_model()
    validate_model_compatibility(model, metadata)

    return model, metadata