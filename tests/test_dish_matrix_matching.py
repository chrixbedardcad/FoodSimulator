import os
import sys
from itertools import combinations

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from food_api import GameData


def _find_unique_family_and_taste_quad(data: GameData):
    ingredients = data.ingredients.values()
    for combo in combinations(ingredients, 4):
        families = {ingredient.family for ingredient in combo}
        tastes = {ingredient.taste for ingredient in combo}
        if len(families) == 4 and len(tastes) == 4:
            return combo
    raise AssertionError("Could not find a 4-card combo with unique families and tastes")


def test_rich_matches_all_different_tastes():
    data = GameData.from_json()
    combo = _find_unique_family_and_taste_quad(data)

    outcome = data.evaluate_dish(combo)

    assert outcome.entry is not None
    assert outcome.entry.name == "Rich"
    expected_multiplier = outcome.entry.multiplier
    assert outcome.dish_multiplier == pytest.approx(expected_multiplier)

    base_value = sum(ingredient.Value for ingredient in combo)
    assert outcome.dish_value == pytest.approx(base_value * expected_multiplier)
