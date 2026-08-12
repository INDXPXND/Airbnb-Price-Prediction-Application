from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.constants import (
    AMENITIES,
    RATING_MEDIAN,
    bucket_checkin,
    bucket_checkout,
    bucket_country,
    bucket_region,
)

MODEL_PATH = Path(__file__).parent / "models" / "cbr.jbl"
model = joblib.load(MODEL_PATH)

# Columns the model was trained on natively as categorical (see notebook cell 35/39).
CAT_COLS = ["country", "checkin", "checkout", "region"]

# Exact column order/names the model expects.
FEATURE_ORDER = list(model.feature_names_)

app = FastAPI(title="Airbnb price prediction API")


class PredictionRequest(BaseModel):
    country: str
    region: str
    bathroom_type: str

    bathrooms: int = Field(ge=0)
    beds: int = Field(ge=0)
    guests: int = Field(ge=1)
    bedrooms: int = Field(ge=0)

    toiles: int = Field(ge=0)

    studio: bool

    checkin: str
    checkout: str

    rating: float | None = Field(default=None, ge=0, le=5)

    reviews: int = Field(ge=0)

    amenities: list[str]             # human-readable labels from constants.AMENITIES


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/amenities")
async def get_amenities():
    """Lets the frontend fetch the exact amenity list the model supports."""
    return {"amenities": [a["display"] for a in AMENITIES]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if request.bathroom_type not in ("Shared", "Private", "Toilet only"):
        raise HTTPException(status_code=422, detail="bathroom_type must be Shared, Private or Toilet only")

    no_rating = request.rating is None
    rating = request.rating if request.rating is not None else RATING_MEDIAN

    # Reproduce the exact top-N -> "Other" bucketing the model was trained
    # with (see constants.py docstring for why this matters).
    country = bucket_country(request.country)
    region = bucket_region(request.country, request.region)
    checkin = bucket_checkin(request.checkin)
    checkout = bucket_checkout(request.checkout)

    row = {
        "rating": rating,
        "reviews": request.reviews,
        "country": country,
        "bathrooms": request.bathrooms,
        "beds": request.beds,
        "guests": request.guests,
        "toiles": request.toiles,
        "bedrooms": request.bedrooms,
        "studios": int(request.studio),
        "checkin": checkin,
        "checkout": checkout,
        "amenities_count": len(request.amenities),
        "region": region,
        "bathroom_shared": int(request.bathroom_type == "Shared"),
        "bathroom_private": int(request.bathroom_type == "Private"),
        "toilet_only": int(request.bathroom_type == "Toilet only"),
        "no_rating": int(no_rating),
    }

    selected = set(request.amenities)
    for amenity in AMENITIES:
        row[amenity["column"]] = int(amenity["display"] in selected)

    data = pd.DataFrame([row])
    # Guarantee every column the model needs exists, in the right order.
    data = data.reindex(columns=FEATURE_ORDER, fill_value=0)
    for col in CAT_COLS:
        data[col] = data[col].astype(str)

    prediction = np.expm1(model.predict(data))

    return {"prediction": float(prediction[0])}