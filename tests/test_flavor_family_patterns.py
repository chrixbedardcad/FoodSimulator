from food_api import GameData


def _load_data() -> GameData:
    return GameData.from_json()


def test_single_taste_varied_family_matches_matrix():
    data = _load_data()
    ingredients = [
        data.ingredients["Tomato"],
        data.ingredients["Yogurt"],
        data.ingredients["PickledCucumber"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Harmony Roll"
    assert round(outcome.dish_multiplier, 2) == 4.2


def test_same_taste_mixed_families_still_score_base_value():
    data = _load_data()
    ingredients = [
        data.ingredients["Tomato"],
        data.ingredients["Lemon"],
        data.ingredients["Yogurt"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is None
    assert outcome.family_pattern == "balanced"
    assert outcome.flavor_pattern == "all_same"
    assert round(outcome.dish_multiplier, 2) == 1.0


def test_balanced_five_all_different_unlocks_mosaic_feast():
    data = _load_data()
    ingredients = [
        data.ingredients["Onion"],
        data.ingredients["Bacon"],
        data.ingredients["Tomato"],
        data.ingredients["Fish"],
        data.ingredients["Garlic"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Mosaic Feast"
    assert outcome.family_pattern == "balanced"
    assert outcome.flavor_pattern == "all_different"
    assert round(outcome.dish_multiplier, 2) == 6.5
