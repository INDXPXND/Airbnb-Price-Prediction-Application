"""
Tests for common/constants.py: the amenity list and the top-N bucketing
helpers that api.py relies on to reproduce the model's training-time
category handling.
"""
from common.constants import (
    AMENITIES,
    BATHROOM_TYPES,
    RATING_MEDIAN,
    TOP_CHECKIN,
    TOP_CHECKOUT,
    TOP_COUNTRIES,
    TOP_REGIONS_BY_COUNTRY,
    bucket_checkin,
    bucket_checkout,
    bucket_country,
    bucket_region,
)


class TestAmenitiesList:
    def test_seven_amenities(self):
        # Trimmed down from the model's full 30 amenity_* columns to the 7
        # that actually carry meaningful feature importance.
        assert len(AMENITIES) == 7

    def test_unique_display_names(self):
        names = [a["display"] for a in AMENITIES]
        assert len(names) == len(set(names))

    def test_unique_columns(self):
        cols = [a["column"] for a in AMENITIES]
        assert len(cols) == len(set(cols))

    def test_columns_are_prefixed(self):
        for a in AMENITIES:
            assert a["column"].startswith("amenity_")

    def test_entries_have_display_and_column(self):
        for a in AMENITIES:
            assert "display" in a and "column" in a
            assert a["display"] and a["column"]


class TestBathroomTypes:
    def test_three_types(self):
        assert set(BATHROOM_TYPES) == {"Shared", "Private", "Toilet only"}


class TestRatingMedian:
    def test_within_valid_range(self):
        assert 0 <= RATING_MEDIAN <= 5


class TestTopListSizes:
    def test_top_countries(self):
        assert len(TOP_COUNTRIES) == 20
        assert len(set(TOP_COUNTRIES)) == 20

    def test_top_checkin(self):
        assert len(TOP_CHECKIN) == 5
        assert len(set(TOP_CHECKIN)) == 5

    def test_top_checkout(self):
        assert len(TOP_CHECKOUT) == 5
        assert len(set(TOP_CHECKOUT)) == 5

    def test_top_regions_total_at_most_30(self):
        all_regions = {r for regions in TOP_REGIONS_BY_COUNTRY.values() for r in regions}
        assert len(all_regions) <= 30


class TestBucketCountry:
    def test_top_countries_pass_through_unchanged(self):
        for country in TOP_COUNTRIES:
            assert bucket_country(country) == country

    def test_unseen_country_becomes_other(self):
        assert bucket_country("Definitely Not A Country") == "Other"

    def test_other_passes_through(self):
        assert bucket_country("Other") == "Other"

    def test_case_sensitive(self):
        # Bucketing must match the exact strings the model was trained on --
        # a differently-cased value is a different (unseen) category.
        lowered = TOP_COUNTRIES[0].lower()
        if lowered != TOP_COUNTRIES[0]:
            assert bucket_country(lowered) == "Other"


class TestBucketRegion:
    def test_valid_region_for_its_own_country(self):
        for country, regions in TOP_REGIONS_BY_COUNTRY.items():
            for region in regions:
                assert bucket_region(country, region) == region

    def test_region_invalid_for_a_different_country(self):
        country, regions = next((c, r) for c, r in TOP_REGIONS_BY_COUNTRY.items() if r)
        region = regions[0]
        wrong_country = next(c for c in TOP_COUNTRIES if c != country)
        if region not in TOP_REGIONS_BY_COUNTRY.get(wrong_country, []):
            assert bucket_region(wrong_country, region) == "Other"

    def test_unknown_country_and_region(self):
        assert bucket_region("Nowhereland", "Anywhere") == "Other"

    def test_known_country_unknown_region(self):
        country = next(iter(TOP_REGIONS_BY_COUNTRY))
        assert bucket_region(country, "Some Random Village") == "Other"


class TestBucketCheckinCheckout:
    def test_top_checkin_pass_through(self):
        for value in TOP_CHECKIN:
            assert bucket_checkin(value) == value

    def test_unseen_checkin_becomes_other(self):
        assert bucket_checkin("Whenever you like") == "Other"

    def test_top_checkout_pass_through(self):
        for value in TOP_CHECKOUT:
            assert bucket_checkout(value) == value

    def test_unseen_checkout_becomes_other(self):
        assert bucket_checkout("Whenever you like") == "Other"