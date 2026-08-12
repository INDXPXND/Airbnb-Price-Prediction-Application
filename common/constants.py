"""
Shared constants for the Airbnb price prediction app.

The list below mirrors the model's (cbr.jbl) "top amenity" dummy columns
(see Airbnb.ipynb, cell 35: `AMENITIES_TOP_N = 30`). Each entry maps a
human-readable label (shown in the Streamlit UI) to the exact feature-column
name the model expects.

Only 7 of those 30 amenity columns are kept here. Checking
model.feature_importances_ shows a sharp drop-off after these — things like
Wifi, Kitchen, TV or a washing machine barely move the prediction (each
<0.3% importance, vs. e.g. ~2.3% for Pool) because almost every listing has
them, so they carry little signal. Showing 30 checkboxes when only ~7 of
them actually change the price is misleading, so the form only offers the
ones that matter:

    amenity_pool                        2.26%
    amenity_private_pool                1.45%
    amenity_air_conditioning            1.42%
    amenity_pets_allowed                1.38%
    amenity_breakfast                   0.55%
    amenity_free_parking_on_premises    0.55%
    amenity_smoking_allowed             0.43%
    -- next one (amenity_indoor_fireplace) drops to 0.35%, and it keeps
       falling from there -> everything past this line is noise.
"""

AMENITIES = [
    {"display": "Pool", "column": "amenity_pool"},
    {"display": "Private pool", "column": "amenity_private_pool"},
    {"display": "Air conditioning", "column": "amenity_air_conditioning"},
    {"display": "Pets allowed", "column": "amenity_pets_allowed"},
    {"display": "Breakfast", "column": "amenity_breakfast"},
    {"display": "Free parking on premises", "column": "amenity_free_parking_on_premises"},
    {"display": "Smoking allowed", "column": "amenity_smoking_allowed"},
]

# Fallback used when the listing has no rating yet (same approach as training:
# the notebook fills missing ratings with the training-set median and keeps a
# separate `no_rating` flag column).
RATING_MEDIAN = 4.89

BATHROOM_TYPES = ["Shared", "Private", "Toilet only"]

# --------------------------------------------------------------------------
# Categorical bucketing (must mirror notebook cell 35 EXACTLY).
#
# During training, `country`, `region`, `checkin` and `checkout` were reduced
# to their top-N most frequent values (computed on X_train only, using the
# same train_test_split(test_size=0.3, random_state=42) as the notebook);
# every other value was replaced with the literal string "Other" BEFORE
# CatBoost ever saw the data. CatBoost then learned real, meaningful
# statistics for "Other" as its own category.
#
# If we instead feed the model some raw value it never saw during training
# (e.g. a country outside this list), CatBoost treats it as a genuinely
# unknown category and falls back to an internal default — which is NOT the
# same as "Other" and, worse, is IDENTICAL for every unseen value regardless
# of what it actually is. That's why two different "rare" countries used to
# produce the exact same prediction. To predict correctly we must replicate
# the same bucketing ourselves: anything not in these lists -> "Other".
# --------------------------------------------------------------------------

TOP_COUNTRIES = [
    "India", "Italy", "Greece", "Thailand", "Turkey", "France", "Morocco",
    "Georgia", "United Kingdom", "Japan", "Sri Lanka", "Taiwan", "Indonesia",
    "Germany", "Cuba", "South Korea", "Philippines", "Norway",
    "United States", "Croatia",
]

TOP_CHECKIN = ["After 3 00 pm", "After 2 00 pm", "Flexible", "Unknown", "After 12 00 pm"]

TOP_CHECKOUT = ["11 00 am", "10 00 am", "12 00 pm", "Unknown", "9 00 am"]

# The 30 top regions overall, grouped by the country they actually belong to
# in the dataset (exact, case-sensitive strings — required for an exact match
# against what CatBoost learned). Any region not listed here for the chosen
# country is bucketed to "Other", even if the country itself is one of the
# top 20.
TOP_REGIONS_BY_COUNTRY = {
    "Turkey": ["Antalya", "Muğla", "Sakarya"],
    "Georgia": ["Gudauri", "Mtskheta-Mtianeti"],
    "Thailand": ["Chang Wat Phuket", "Phuket"],
    "India": [
        "Delhi", "Goa", "Haryana", "Himachal Pradesh", "Kerala",
        "Maharashtra", "Rajasthan", "Uttar Pradesh", "Uttarakhand",
    ],
    "Philippines": ["Calabarzon"],
    "Indonesia": ["Bali"],
    "Greece": ["Bali", "Egeo"],
    "Italy": ["Italy", "Sicilia", "Sicily", "Toscana"],
    "Sri Lanka": ["Southern Province"],
    "Norway": ["Nordland"],
    "France": ["Provence-Alpes-Côte d'Azur"],
    "Morocco": ["Marrakesh-Safi"],
    "United Kingdom": ["England"],
    "Jordan": ["Aqaba Governorate"],
    "Cuba": ["Sancti Spíritus"],
}


def bucket_country(country: str) -> str:
    return country if country in TOP_COUNTRIES else "Other"


def bucket_region(country: str, region: str) -> str:
    allowed = TOP_REGIONS_BY_COUNTRY.get(country, [])
    return region if region in allowed else "Other"


def bucket_checkin(checkin: str) -> str:
    return checkin if checkin in TOP_CHECKIN else "Other"


def bucket_checkout(checkout: str) -> str:
    return checkout if checkout in TOP_CHECKOUT else "Other"
