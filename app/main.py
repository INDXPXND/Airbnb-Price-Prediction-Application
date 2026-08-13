import sys
from pathlib import Path

# `streamlit run app/main.py` puts this file's own directory (app/) on
# sys.path, not the project root -- so the sibling `common/` package isn't
# found unless PYTHONPATH is set. Docker sets PYTHONPATH explicitly
# (see Dockerfile.streamlit), but a plain local run doesn't, so we add the
# project root here too, making this work regardless of cwd or PYTHONPATH.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
import streamlit as st
import os


from common.constants import (
    AMENITIES,
    BATHROOM_TYPES,
    TOP_CHECKIN,
    TOP_CHECKOUT,
    TOP_COUNTRIES,
    TOP_REGIONS_BY_COUNTRY,
)

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Airbnb price prediction", page_icon="🏠")

@st.cache_data
def load_data():
    return pd.read_csv("datasets/airbnb_updated.csv")


df = load_data()

st.title("Airbnb apartment price prediction")
st.caption("Fill in the listing details and get an estimated nightly price.")

st.caption("Top 20 countries: " + ", ".join(TOP_COUNTRIES))


col1, col2 = st.columns(2)
with col1:

    country = st.selectbox("Country", sorted(df["country"].unique()))
    if country in TOP_COUNTRIES:
        st.caption(f"✅ {country} has its own learned effect on price.")
    else:
        st.caption(f'ℹ️ {country} isn\'t one of the model\'s top 20 countries — it will be treated as "Other".')

    bathroom_type = st.selectbox("Bathroom", BATHROOM_TYPES)

    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=int(df["bathrooms"].max()), value=1, step=1)

    guests = st.number_input("Guests", min_value=1, max_value=int(df["guests"].max()), value=2, step=1)

with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=int(df["bedrooms"].max()), value=1, step=1)

    beds = st.number_input("Beds", min_value=0, max_value=int(df["beds"].max()), value=1, step=1)

    studio = st.checkbox("Studio apartment")

with st.expander("Additional details (optional)"):
    # The model only learned individual statistics for 30 regions, 5 check-in
    # windows and 5 check-out times (see constants.py). Everything else is
    # bucketed to "Other" at prediction time, so — unlike before — we only
    # offer the values that actually change the prediction, plus "Other".
    region_choices = sorted(TOP_REGIONS_BY_COUNTRY.get(country, []))
    if region_choices:
        region = st.selectbox(
            "Region",
            region_choices + ["Other"],
            help='Only regions the model learned individually are listed; anything else counts as "Other".',
        )
    else:
        region = "Other"
        st.selectbox("Region", ["Other"], disabled=True,
                     help=f"No individually-modeled region for {country} — region will be \"Other\".")

    checkin_choices = TOP_CHECKIN + ["Other"]

    checkin = st.selectbox("Check-in", checkin_choices, index=checkin_choices.index("Flexible"))

    checkout_choices = TOP_CHECKOUT + ["Other"]
    checkout = st.selectbox("Check-out", checkout_choices)

    toiles = st.number_input("Toilets", min_value=0, max_value=int(df["toiles"].max()), value=0, step=1)

    reviews = st.number_input("Number of reviews", min_value=0, max_value=int(df["reviews"].max()), value=0, step=1)

    has_rating = st.checkbox("Listing already has a rating")
    rating = st.slider("Rating", 0.0, 5.0, 4.8, 0.01) if has_rating else None

amenities = st.multiselect("Amenities", [a["display"] for a in AMENITIES])

errors = []
if not (0 <= bathrooms <= df["bathrooms"].max()):
    errors.append(f"Bathrooms must be between 0 and {int(df['bathrooms'].max())} (got {bathrooms}).")
if not (1 <= guests <= df["guests"].max()):
    errors.append(f"Guests must be between 1 and {int(df['guests'].max())} (got {guests}).")
if not (0 <= bedrooms <= df["bedrooms"].max()):
    errors.append(f"Bedrooms must be between 0 and {int(df['bedrooms'].max())} (got {bedrooms}).")
if not (0 <= beds <= df["beds"].max()):
    errors.append(f"Beds must be between 0 and {int(df['beds'].max())} (got {beds}).")
if not (0 <= toiles <= df["toiles"].max()):
    errors.append(f"Toilets must be between 0 and {int(df['toiles'].max())} (got {toiles}).")
if reviews < 0:
    errors.append(f"Number of reviews can't be negative (got {reviews}).")

for error in errors:
    st.error(error)

if st.button("Predict", disabled=bool(errors)):
    payload = {
        "country": country,
        "region": region,
        "bathroom_type": bathroom_type,
        "bathrooms": bathrooms,
        "beds": beds,
        "guests": guests,
        "bedrooms": bedrooms,
        "toiles": toiles,
        "studio": studio,
        "checkin": checkin,
        "checkout": checkout,
        "rating": rating,
        "reviews": reviews,
        "amenities": amenities,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()["prediction"]
        st.success(f"Estimated price: **${prediction:,.2f}** per night")
    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the prediction API. Start it first with:\n\n"
            "`uvicorn api:app --reload`"
        )
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned an error: {exc}")
    except (KeyError, ValueError):
        st.error("Unexpected response from the API.")