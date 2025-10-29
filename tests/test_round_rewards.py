import random

import pytest

pytest.importorskip("PIL")

from food_api import GameData
from food_desktop import GameSession


@pytest.fixture
def game_data() -> GameData:
    return GameData.from_json()


def test_bonus_choices_include_full_catalog(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    basket_ingredients = {name for name, _ in game_data.baskets[basket_name]}
    all_ingredients = set(game_data.ingredients.keys())
    outside_basket = all_ingredients - basket_ingredients
    assert outside_basket  # sanity check that catalog is larger than the basket

    session = GameSession(
        game_data,
        basket_name=basket_name,
        chefs=[],
        rounds=1,
        hand_size=3,
        pick_size=3,
        deck_size=10,
        bias=0.0,
        rng=random.Random(1234),
    )

    picks = session._choose_bonus_ingredients(count=len(all_ingredients))
    pick_names = {ingredient.name for ingredient in picks}

    assert outside_basket <= pick_names
