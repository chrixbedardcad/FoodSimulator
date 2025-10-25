import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from food_api import GameData


def _load_data() -> GameData:
    return GameData.from_json()


def test_unique_ingredients_have_no_penalty():
    data = _load_data()
    names = [
        "Honey",
        "Beef",
        "Seaweed",
        "Capers",
        "PickledCucumber",
    ]
    ingredients = [data.ingredients[name] for name in names]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 50


def test_global_duplicate_penalty_applies_multiplier():
    data = _load_data()
    ingredients = [
        data.ingredients["OliveOil"],
        data.ingredients["OliveOil"],
        data.ingredients["Egg"],
        data.ingredients["Tomato"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 43


def test_per_card_duplicate_penalty_scores_extra_copies_lower():
    data = _load_data()
    base_rules = dict(getattr(data, "rules", {}) or {})
    penalty_config = dict(base_rules.get("duplicate_ingredient_penalty", {}))
    penalty_config["application"] = "per_card"
    base_rules["duplicate_ingredient_penalty"] = penalty_config
    data.rules = base_rules

    ingredients = [
        data.ingredients["OliveOil"],
        data.ingredients["OliveOil"],
        data.ingredients["Honey"],
        data.ingredients["Onion"],
        data.ingredients["Capers"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 44
