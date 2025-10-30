import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from food_api import GameData, Ingredient, build_market_deck
from food_desktop import GameSession
from rotting_round import IngredientCard


def _make_game_data(counts: Dict[str, int]) -> GameData:
    ingredients = {
        name: Ingredient(
            name=name,
            ingredient_id=f"{name.lower()}-id",
            taste="Sweet",
            Value=1,
            family="Test",
            display_name=name,
        )
        for name in counts
    }
    basket = [(name, copies) for name, copies in counts.items()]
    return GameData(
        ingredients=ingredients,
        recipes=[],
        chefs=[],
        seasonings=[],
        baskets={"Test": basket},
        taste_matrix={},
        dish_matrix=[],
        rules={},
    )


def test_build_market_deck_draws_uniform_unique_choices() -> None:
    counts = {"Honey": 10, "Apple": 1, "Salt": 1, "Pepper": 1, "Lime": 1}
    data = _make_game_data(counts)
    deck = build_market_deck(
        data,
        "Test",
        chefs=[],
        deck_size=sum(counts.values()),
        bias=1.0,
    )

    deck_names = [ingredient.name for ingredient in deck]
    assert len(deck_names) == sum(counts.values())
    assert Counter(deck_names) == Counter(counts)

    expected_unique_prefix = min(len(counts), len(deck_names))
    assert len(set(deck_names[:expected_unique_prefix])) == expected_unique_prefix


def test_rebalance_deck_preserves_uniform_draws() -> None:
    counts = {"Honey": 6, "Apple": 1, "Salt": 1, "Pepper": 1}
    data = _make_game_data(counts)

    session = GameSession(
        data,
        basket_name="Test",
        chefs=[],
        rounds=1,
        hand_size=1,
        pick_size=1,
        deck_size=0,
    )

    cards = [
        IngredientCard(ingredient=data.ingredients[name])
        for name, copies in counts.items()
        for _ in range(copies)
    ]

    balanced = session._rebalance_deck(cards)

    assert {id(card) for card in balanced} == {id(card) for card in cards}

    deck_names = [card.ingredient.name for card in balanced]
    assert Counter(deck_names) == Counter(counts)

    expected_unique_prefix = min(len(counts), len(deck_names))
    assert len(set(deck_names[:expected_unique_prefix])) == expected_unique_prefix
