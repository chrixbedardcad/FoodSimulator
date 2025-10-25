import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from food_api import GameData, Ingredient


def _load_data() -> GameData:
    return GameData.from_json()


def _sample_cards() -> dict[str, Ingredient]:
    return {
        "Tomato": Ingredient(
            name="Tomato",
            ingredient_id="test.tomato",
            taste="Sour",
            Value=22,
            family="Fruit",
        ),
        "Basil": Ingredient(
            name="Basil",
            ingredient_id="test.basil",
            taste="Sweet",
            Value=10,
            family="Herb",
        ),
        "Mozzarella": Ingredient(
            name="Mozzarella",
            ingredient_id="test.mozzarella",
            taste="Sweet",
            Value=18,
            family="Dairy",
        ),
        "Onion": Ingredient(
            name="Onion",
            ingredient_id="test.onion",
            taste="Sweet",
            Value=12,
            family="Vegetable",
        ),
    }


def test_unique_ingredients_have_no_penalty():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [cards["Tomato"], cards["Basil"], cards["Mozzarella"]]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 50


def test_global_duplicate_penalty_applies_multiplier():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [
        cards["Tomato"],
        cards["Tomato"],
        cards["Basil"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 43


def test_duplicate_penalty_alert_mentions_penalty():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [
        cards["Tomato"],
        cards["Tomato"],
        cards["Basil"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert any("too much Tomato" in alert for alert in outcome.alerts)
    assert any("penalty -20%" in alert for alert in outcome.alerts)


def test_global_duplicate_penalty_handles_multiple_types():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [
        cards["Tomato"],
        cards["Tomato"],
        cards["Onion"],
        cards["Onion"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 44


def test_per_card_duplicate_penalty_scores_extra_copies_lower():
    data = _load_data()
    base_rules = dict(getattr(data, "rules", {}) or {})
    penalty_config = dict(base_rules.get("duplicate_ingredient_penalty", {}))
    penalty_config["application"] = "per_card"
    base_rules["duplicate_ingredient_penalty"] = penalty_config
    data.rules = base_rules

    cards = _sample_cards()
    ingredients = [
        cards["Tomato"],
        cards["Tomato"],
        cards["Basil"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 50


def test_duplicate_alert_triggers_for_third_copy():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [
        cards["Tomato"],
        cards["Tomato"],
        cards["Tomato"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert any("over taste" in alert for alert in outcome.alerts)
