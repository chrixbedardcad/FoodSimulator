from collections import Counter
import random

import pytest

from food_api import GameData
from food_desktop import GameSession


@pytest.fixture
def game_data() -> GameData:
    return GameData.from_json()


def _ingredient_id_from_name(data: GameData, name: str) -> str:
    ingredient = data.ingredients.get(name)
    if not ingredient:
        return ""
    return getattr(ingredient, "ingredient_id", "")


def test_get_pantry_card_ids_reflects_remaining_cards(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    session = GameSession(
        game_data,
        basket_name=basket_name,
        chefs=[],
        rounds=1,
        hand_size=3,
        pick_size=3,
        deck_size=12,
        rng=random.Random(9876),
    )

    def expected_counts() -> Counter[str]:
        counts: Counter[str] = Counter()
        for card in session.get_hand():
            ingredient = card.ingredient
            ingredient_id = getattr(ingredient, "ingredient_id", "")
            if not ingredient_id:
                ingredient_id = _ingredient_id_from_name(game_data, ingredient.name)
            if ingredient_id:
                counts[ingredient_id] += 1
        for ingredient in session.get_remaining_deck():
            ingredient_id = getattr(ingredient, "ingredient_id", "")
            if not ingredient_id:
                ingredient_id = _ingredient_id_from_name(game_data, ingredient.name)
            if ingredient_id:
                counts[ingredient_id] += 1
        return counts

    assert Counter(session.get_pantry_card_ids()) == expected_counts()

    if session.get_hand():
        session.hand.pop()
    if session.deck:
        session.deck.pop()

    assert Counter(session.get_pantry_card_ids()) == expected_counts()
