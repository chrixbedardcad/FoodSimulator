from collections import Counter
import random

import pytest

from food_api import GameData, build_market_deck
from food_desktop import GameSession
from rotting_round import IngredientCard


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


def test_leftover_cards_carry_into_next_round(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    session = GameSession(
        game_data,
        basket_name=basket_name,
        chefs=[],
        rounds=2,
        hand_size=3,
        pick_size=3,
        deck_size=6,
        rng=random.Random(4321),
    )

    leftover_cards = list(session.hand)
    session.deck.clear()
    session._awaiting_basket_reset = True

    session.begin_next_round_after_reward()

    total_cards = len(session.hand) + len(session.deck)
    assert total_cards == session.deck_size + len(leftover_cards)

    current_ids = {id(card) for card in session.hand + session.deck}
    for card in leftover_cards:
        assert id(card) in current_ids


def test_market_deck_priors_new_ingredients(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    rng = random.Random(2468)
    deck = build_market_deck(
        game_data,
        basket_name=basket_name,
        chefs=[],
        deck_size=40,
        rng=rng,
    )

    unique_names = {
        ingredient_name
        for ingredient_name, _ in game_data.baskets[basket_name]
        if game_data.ingredients.get(ingredient_name)
    }

    expected_unique_prefix = min(len(unique_names), len(deck))
    seen: set[str] = set()
    for index in range(expected_unique_prefix):
        name = deck[index].name
        assert name not in seen
        seen.add(name)


def test_rebalance_deck_spreads_duplicates(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    session = GameSession(
        game_data,
        basket_name=basket_name,
        chefs=[],
        rounds=1,
        hand_size=3,
        pick_size=3,
        deck_size=12,
        rng=random.Random(1357),
    )

    sample_cards: list[IngredientCard] = []
    ingredient_names = list(game_data.ingredients.keys())[:5]
    for index, name in enumerate(ingredient_names):
        ingredient = game_data.ingredients[name]
        copies = 4 if index == 0 else 1
        for _ in range(copies):
            sample_cards.append(IngredientCard(ingredient=ingredient))

    rebalanced = session._rebalance_deck(sample_cards)
    unique_names = {card.ingredient.name for card in sample_cards}
    expected_unique_prefix = min(len(unique_names), len(rebalanced))
    seen: set[str] = set()
    for index in range(expected_unique_prefix):
        name = rebalanced[index].ingredient.name
        assert name not in seen
        seen.add(name)
