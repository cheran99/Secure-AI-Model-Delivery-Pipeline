from pydantic import BaseModel, conlist

class PredictRequest(BaseModel):
    features: conlist(float, min_items=10, max_items=10)  # ensure expected length