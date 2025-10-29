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


def test_reward_ingredient_persists_across_rounds(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    bonus_ingredient = next(iter(game_data.ingredients.values()))

    session = GameSession(
        game_data,
        basket_name=basket_name,
        chefs=[],
        rounds=3,
        hand_size=3,
        pick_size=3,
        deck_size=9,
        bias=0.0,
        rng=random.Random(2024),
    )

    session.hand.clear()
    session.deck.clear()
    session._enter_round_summary()
    assert session.awaiting_new_round()

    session.begin_next_round_from_empty_basket(bonus_ingredient)

    def total_bonus_cards() -> int:
        in_hand = sum(
            1
            for card in session.get_hand()
            if card.ingredient.name == bonus_ingredient.name
        )
        in_deck = sum(
            1
            for ingredient in session.get_remaining_deck()
            if ingredient.name == bonus_ingredient.name
        )
        return in_hand + in_deck

    assert total_bonus_cards() >= 1

    session.hand.clear()
    session.deck.clear()
    session._enter_round_summary()
    assert session.awaiting_new_round()

    session.begin_next_round_after_reward()

    assert total_bonus_cards() >= 1
