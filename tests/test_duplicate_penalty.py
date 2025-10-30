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
            display_name="Tomato",
        ),
        "Basil": Ingredient(
            name="Basil",
            ingredient_id="test.basil",
            taste="Sweet",
            Value=10,
            family="Herb",
            display_name="Basil",
        ),
        "Mozzarella": Ingredient(
            name="Mozzarella",
            ingredient_id="test.mozzarella",
            taste="Sweet",
            Value=18,
            family="Dairy",
            display_name="Mozzarella",
        ),
        "Onion": Ingredient(
            name="Onion",
            ingredient_id="test.onion",
            taste="Sweet",
            Value=12,
            family="Vegetable",
            display_name="Onion",
        ),
    }


def test_unique_ingredients_have_no_penalty():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [cards["Tomato"], cards["Basil"], cards["Mozzarella"]]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 50


def test_three_unique_families_same_taste_match_tasteful():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [cards["Basil"], cards["Mozzarella"], cards["Onion"]]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Tasteful"
    assert outcome.flavor_pattern == "all_same"


def test_global_duplicate_penalty_applies_multiplier():
    data = _load_data()
    cards = _sample_cards()
    ingredients = [
        cards["Tomato"],
        cards["Tomato"],
        cards["Basil"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert int(round(outcome.dish_value)) == 27


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
    assert any("penalty -50%" in alert for alert in outcome.alerts)


def test_duplicate_alert_uses_display_name():
    data = _load_data()
    pickle = data.ingredients["PickledCucumber"]
    tomato = data.ingredients["Tomato"]

    outcome = data.evaluate_dish([pickle, pickle, tomato])

    assert pickle.display_name == "Pickled Cucumber"
    assert any("Pickled Cucumber" in alert for alert in outcome.alerts)


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

    assert int(round(outcome.dish_value)) == 17


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

    assert int(round(outcome.dish_value)) == 43


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


def test_duplicate_penalty_reduces_unmatched_all_same_flavor():
    data = _load_data()
    rice = data.ingredients["Rice"]
    pasta = data.ingredients["Pasta"]
    seaweed = data.ingredients["Seaweed"]

    ingredients = [pasta, rice, rice, seaweed]

    outcome = data.evaluate_dish(ingredients)

    assert round(outcome.dish_multiplier, 2) == 1.0
    assert int(round(outcome.dish_value)) == int(round(outcome.base_value))
    assert any("too much Rice" in alert for alert in outcome.alerts)
