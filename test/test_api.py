import pytest
from fastapi.testclient import TestClient

from api import app
from common.constants import AMENITIES, TOP_CHECKIN, TOP_CHECKOUT, TOP_COUNTRIES, TOP_REGIONS_BY_COUNTRY

client = TestClient(app)

BASE_PAYLOAD = {
    "country": "India",
    "region": "Delhi",
    "bathroom_type": "Private",
    "bathrooms": 1,
    "beds": 1,
    "guests": 2,
    "bedrooms": 1,
    "toiles": 0,
    "studio": False,
    "checkin": "Flexible",
    "checkout": "Unknown",
    "rating": None,
    "reviews": 0,
    "amenities": [],
}


def payload(**overrides):
    return {**BASE_PAYLOAD, **overrides}


def predict(**overrides):
    return client.post("/predict", json=payload(**overrides))


# ---------------------------------------------------------------------------
# Health / metadata endpoints
# ---------------------------------------------------------------------------

class TestHealthAndMeta:
    def test_root_ok(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_amenities_endpoint_matches_constants(self):
        r = client.get("/amenities")
        assert r.status_code == 200
        assert r.json() == {"amenities": [a["display"] for a in AMENITIES]}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    def test_predict_returns_positive_float(self):
        r = predict()
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["prediction"], float)
        assert body["prediction"] > 0

    def test_predict_is_deterministic(self):
        assert predict().json() == predict().json()

    def test_rating_none_does_not_error(self):
        assert predict(rating=None).status_code == 200

    def test_missing_rating_diverges_from_explicit_low_rating(self):
        # rating=None -> RATING_MEDIAN (~4.9) + no_rating=1
        # An explicit low rating should noticeably change the prediction.
        p_none = predict(rating=None).json()["prediction"]
        p_low = predict(rating=1.0).json()["prediction"]
        assert p_none != p_low


# ---------------------------------------------------------------------------
# Input validation (pydantic Field constraints)
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("field", ["bathrooms", "beds", "bedrooms", "toiles", "reviews"])
    def test_negative_numeric_fields_rejected(self, field):
        assert predict(**{field: -1}).status_code == 422

    def test_guests_zero_rejected(self):
        # guests has ge=1, not ge=0
        assert predict(guests=0).status_code == 422

    def test_guests_negative_rejected(self):
        assert predict(guests=-1).status_code == 422

    @pytest.mark.parametrize("rating", [-0.1, 5.1, 100])
    def test_rating_out_of_range_rejected(self, rating):
        assert predict(rating=rating).status_code == 422

    @pytest.mark.parametrize("rating", [0, 5, 2.5])
    def test_rating_in_range_accepted(self, rating):
        assert predict(rating=rating).status_code == 200

    def test_invalid_bathroom_type_rejected(self):
        r = predict(bathroom_type="Luxury")
        assert r.status_code == 422

    @pytest.mark.parametrize("bathroom_type", ["Shared", "Private", "Toilet only"])
    def test_valid_bathroom_types_accepted(self, bathroom_type):
        assert predict(bathroom_type=bathroom_type).status_code == 200

    def test_missing_required_field_rejected(self):
        bad = payload()
        del bad["country"]
        assert client.post("/predict", json=bad).status_code == 422

    def test_wrong_type_rejected(self):
        assert predict(bathrooms="a lot").status_code == 422


# ---------------------------------------------------------------------------
# "Other" bucketing -- regression tests for the bug where unseen categories
# were passed to the model raw instead of being mapped to the literal
# "Other" category the model was actually trained on.
# ---------------------------------------------------------------------------

class TestCategoryBucketing:
    def test_unseen_countries_match_explicit_other(self):
        p_other = predict(country="Other").json()["prediction"]
        p_a = predict(country="Nowhereland").json()["prediction"]
        p_b = predict(country="Fiji").json()["prediction"]
        assert p_a == p_other
        assert p_b == p_other

    def test_two_different_unseen_countries_agree_with_each_other(self):
        # This is the exact bug we hit: two different never-seen countries
        # must land on the same (correct) "Other" prediction.
        assert predict(country="Nepal").json() == predict(country="Poland").json()

    def test_top_country_differs_from_other(self):
        top_country = TOP_COUNTRIES[0]
        p_top = predict(country=top_country, region="Other").json()["prediction"]
        p_other = predict(country="Other", region="Other").json()["prediction"]
        assert p_top != p_other

    def test_region_bucketed_per_country(self):
        country, regions = next((c, r) for c, r in TOP_REGIONS_BY_COUNTRY.items() if r)
        top_region = regions[0]
        p_top_region = predict(country=country, region=top_region).json()["prediction"]
        p_other_region = predict(country=country, region="Other").json()["prediction"]
        assert p_top_region != p_other_region

    def test_region_only_counts_for_its_own_country(self):
        # A region in the top-30 list for country A must bucket to "Other"
        # when submitted together with a different country B.
        for country, regions in TOP_REGIONS_BY_COUNTRY.items():
            for region in regions:
                other_country = next(c for c in TOP_COUNTRIES if c != country)
                if region not in TOP_REGIONS_BY_COUNTRY.get(other_country, []):
                    p_wrong = predict(country=other_country, region=region).json()["prediction"]
                    p_other = predict(country=other_country, region="Other").json()["prediction"]
                    assert p_wrong == p_other
                    return
        pytest.skip("No suitable country/region pair found for this test")

    def test_unseen_checkin_matches_other(self):
        p_a = predict(checkin="Whenever works for you").json()["prediction"]
        p_other = predict(checkin="Other").json()["prediction"]
        assert p_a == p_other

    def test_unseen_checkout_matches_other(self):
        p_a = predict(checkout="3pm sharp").json()["prediction"]
        p_other = predict(checkout="Other").json()["prediction"]
        assert p_a == p_other

    def test_top_checkin_differs_from_other(self):
        top_checkin = TOP_CHECKIN[0]
        p_top = predict(checkin=top_checkin).json()["prediction"]
        p_other = predict(checkin="Other").json()["prediction"]
        assert p_top != p_other

    def test_top_checkout_differs_from_other(self):
        top_checkout = TOP_CHECKOUT[0]
        p_top = predict(checkout=top_checkout).json()["prediction"]
        p_other = predict(checkout="Other").json()["prediction"]
        assert p_top != p_other


# ---------------------------------------------------------------------------
# Amenities
# ---------------------------------------------------------------------------

class TestAmenities:
    def test_all_amenities_changes_prediction_vs_none(self):
        p_none = predict(amenities=[]).json()["prediction"]
        display_names = [a["display"] for a in AMENITIES]
        p_all = predict(amenities=display_names).json()["prediction"]
        assert p_all != p_none

    def test_pool_alone_increases_price(self):
        # Pool has the highest feature importance among amenities -- adding
        # it (all else equal) should raise the predicted price.
        p_none = predict(amenities=[]).json()["prediction"]
        p_pool = predict(amenities=["Pool"]).json()["prediction"]
        assert p_pool > p_none

    def test_unknown_amenity_label_does_not_error(self):
        # Labels that don't match any known amenity are simply ignored for
        # the amenity_* columns, but still count toward amenities_count.
        r = predict(amenities=["Definitely not a real amenity"])
        assert r.status_code == 200