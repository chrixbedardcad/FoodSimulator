from food_api import GameData


def _load_data() -> GameData:
    return GameData.from_json()


def test_all_same_taste_all_different_families_scores_tasteful():
    data = _load_data()
    ingredients = [
        data.ingredients["Truffle"],
        data.ingredients["Pasta"],
        data.ingredients["Beef"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Tasteful"
    assert outcome.family_pattern == "mixed"
    assert outcome.flavor_pattern == "all_same"
    assert round(outcome.dish_multiplier, 2) == 2.0


def test_all_same_taste_mixed_families_still_scores_tasteful():
    data = _load_data()
    ingredients = [
        data.ingredients["Tomato"],
        data.ingredients["Lemon"],
        data.ingredients["Yogurt"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Tasteful"
    assert outcome.family_pattern == "mixed"
    assert outcome.flavor_pattern == "all_same"
    assert round(outcome.dish_multiplier, 2) == 2.0


def test_all_same_family_all_same_taste_scores_fusion():
    data = _load_data()
    ingredients = [
        data.ingredients["Truffle"],
        data.ingredients["Mushroom"],
        data.ingredients["Seaweed"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Fusion"
    assert outcome.family_pattern == "all_same"
    assert outcome.flavor_pattern == "all_same"
    assert round(outcome.dish_multiplier, 2) == 6.0


def test_all_different_family_and_flavor_scores_rich():
    data = _load_data()
    ingredients = [
        data.ingredients["Truffle"],
        data.ingredients["Egg"],
        data.ingredients["Tomato"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Rich"
    assert outcome.family_pattern == "mixed"
    assert outcome.flavor_pattern == "mixed"
    assert round(outcome.dish_multiplier, 2) == 3.0


def test_same_family_flavor_mixed_scores_harmony():
    data = _load_data()
    ingredients = [
        data.ingredients["Truffle"],
        data.ingredients["Mushroom"],
        data.ingredients["Basil"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Harmony"
    assert outcome.family_pattern == "all_same"
    assert outcome.flavor_pattern == "mixed"
    assert round(outcome.dish_multiplier, 2) == 2.0


def test_same_family_all_different_flavors_still_scores_harmony():
    data = _load_data()
    ingredients = [
        data.ingredients["Truffle"],
        data.ingredients["Basil"],
        data.ingredients["Onion"],
    ]

    outcome = data.evaluate_dish(ingredients)

    assert outcome.entry is not None
    assert outcome.entry.name == "Harmony"
    assert outcome.family_pattern == "all_same"
    assert outcome.flavor_pattern == "mixed"
    assert round(outcome.dish_multiplier, 2) == 2.0
