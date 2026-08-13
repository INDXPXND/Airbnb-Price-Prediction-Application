"""
Tests for app/main.py using Streamlit's AppTest framework.

main.py reads "datasets/airbnb_updated.csv" as a relative path, so these
tests chdir to the project root before running the script (matching how it's
actually launched: `streamlit run app/main.py` from the project root, or
WORKDIR /code in Dockerfile.streamlit).
"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from common.constants import AMENITIES, TOP_COUNTRIES

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "app" / "main.py"


@pytest.fixture
def app_test():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        at = AppTest.from_file(str(MAIN_PY))
        at.run(timeout=30)
        yield at
    finally:
        os.chdir(cwd)


def get_number_input(at, label):
    return next(ni for ni in at.number_input if ni.label == label)


def get_selectbox(at, label):
    return next(sb for sb in at.selectbox if sb.label == label)


def get_button(at, label):
    return next(b for b in at.button if b.label == label)


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

class TestLoadsCleanly:
    def test_no_exception_on_load(self, app_test):
        assert not app_test.exception

    def test_title_present(self, app_test):
        assert app_test.title[0].value == "Airbnb apartment price prediction"

    def test_predict_button_present(self, app_test):
        assert get_button(app_test, "Predict")

    def test_top_20_countries_caption_present(self, app_test):
        captions = [c.value for c in app_test.caption]
        assert any("Top 20 countries" in c for c in captions)
        for country in TOP_COUNTRIES:
            assert any(country in c for c in captions)


class TestExpectedWidgets:
    def test_selectboxes_present(self, app_test):
        labels = {sb.label for sb in app_test.selectbox}
        assert {"Country", "Bathroom", "Check-in", "Check-out"}.issubset(labels)

    def test_amenities_multiselect_matches_constants(self, app_test):
        ms = app_test.multiselect[0]
        assert set(ms.options) == {a["display"] for a in AMENITIES}


# ---------------------------------------------------------------------------
# Numeric input bounds
#
# The +/- steppers (and pasting an out-of-range value) should be capped by
# min_value/max_value on every number_input -- relying only on the
# post-hoc `errors` checks still lets someone spam the +/- buttons to an
# absurd number before the (correct, but late) error shows up.
# ---------------------------------------------------------------------------

class TestNumericInputBounds:
    @pytest.mark.parametrize(
        "label", ["Bathrooms", "Guests", "Bedrooms", "Beds", "Toilets", "Number of reviews"]
    )
    def test_has_min_value(self, app_test, label):
        ni = get_number_input(app_test, label)
        assert ni.min is not None, f'"{label}" has no min_value set -- it can be pushed negative with the "-" button.'

    @pytest.mark.parametrize(
        "label", ["Bathrooms", "Guests", "Bedrooms", "Beds", "Toilets"]
    )
    def test_has_max_value(self, app_test, label):
        ni = get_number_input(app_test, label)
        assert ni.max is not None, f'"{label}" has no max_value set -- it can be pushed to an absurd value with the "+" button.'

    def test_guests_min_is_one(self, app_test):
        ni = get_number_input(app_test, "Guests")
        if ni.min is not None:
            assert ni.min >= 1


# ---------------------------------------------------------------------------
# Validation surfaces errors and blocks prediction
# ---------------------------------------------------------------------------

class TestValidationSurfacesErrors:
    def test_default_state_has_no_errors(self, app_test):
        assert len(app_test.error) == 0

    def test_default_state_predict_enabled(self, app_test):
        assert not get_button(app_test, "Predict").disabled

    def test_bathrooms_cannot_go_negative(self, app_test):
        # With min_value=0 set, the widget itself refuses the attempted -5 --
        # the value stays within bounds and Predict remains usable.
        get_number_input(app_test, "Bathrooms").set_value(-5)
        app_test.run(timeout=30)
        bathrooms = get_number_input(app_test, "Bathrooms")
        assert bathrooms.value >= 0
        assert not get_button(app_test, "Predict").disabled

    def test_reviews_cannot_go_negative(self, app_test):
        get_number_input(app_test, "Number of reviews").set_value(-1)
        app_test.run(timeout=30)
        reviews = get_number_input(app_test, "Number of reviews")
        assert reviews.value >= 0
        assert not get_button(app_test, "Predict").disabled


# ---------------------------------------------------------------------------
# Country / region UX: correctly reflects what the model can distinguish
# ---------------------------------------------------------------------------

class TestCountryRegionHints:
    def test_top_country_shows_positive_caption(self, app_test):
        top_country = TOP_COUNTRIES[0]
        get_selectbox(app_test, "Country").set_value(top_country)
        app_test.run(timeout=30)
        captions = [c.value for c in app_test.caption]
        assert any("✅" in c and top_country in c for c in captions)

    def test_non_top_country_shows_other_caption(self, app_test):
        non_top = next(
            c for c in app_test.selectbox[0].options if c not in TOP_COUNTRIES
        )
        get_selectbox(app_test, "Country").set_value(non_top)
        app_test.run(timeout=30)
        captions = [c.value for c in app_test.caption]
        assert any("ℹ️" in c and "Other" in c for c in captions)


# ---------------------------------------------------------------------------
# End-to-end: submit the form and get a prediction back, with the FastAPI
# app mounted in-process (no real network / no separate uvicorn process
# needed).
# ---------------------------------------------------------------------------

class TestPredictEndToEnd:
    def test_predict_flow_returns_price(self, app_test, monkeypatch):
        from fastapi.testclient import TestClient

        from api import app as fastapi_app

        client = TestClient(fastapi_app)

        def fake_post(url, json=None, timeout=None):
            return client.post("/predict", json=json)

        monkeypatch.setattr("requests.post", fake_post)

        get_button(app_test, "Predict").click()
        app_test.run(timeout=30)

        assert not app_test.error
        assert app_test.success
        assert "Estimated price" in app_test.success[0].value

    def test_predict_flow_reports_api_down(self, app_test, monkeypatch):
        import requests

        def raise_connection_error(url, json=None, timeout=None):
            raise requests.exceptions.ConnectionError("no server")

        monkeypatch.setattr("requests.post", raise_connection_error)

        get_button(app_test, "Predict").click()
        app_test.run(timeout=30)

        assert not app_test.success
        assert any("Can't reach the prediction API" in e.value for e in app_test.error)