from __future__ import annotations

import pytest

from food_api import GameData
from rotting_round import RottingRound, rot_circles


@pytest.fixture
def game_data() -> GameData:
    return GameData.from_json()


def make_round(data: GameData, names: list[str], hand_size: int) -> RottingRound:
    ingredients = [data.ingredients[name] for name in names]
    return RottingRound(data, ingredients, hand_size=hand_size)


def test_turns_increment_only_in_hand(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Tomato", "Mushroom", "Onion"], hand_size=3)

    assert [card.turns_in_hand for card in round_state.cards_in_hand()] == [0, 0, 0]

    round_state.end_turn_decay()
    assert [card.turns_in_hand for card in round_state.cards_in_hand()] == [1, 1, 1]

    round_state.end_turn_decay()
    assert [card.turns_in_hand for card in round_state.cards_in_hand()] == [2, 2, 2]


def test_leaving_hand_preserves_turns_in_hand(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Tomato", "Egg", "Mushroom"], hand_size=3)

    round_state.end_turn_decay()
    assert {card.ingredient.name: card.turns_in_hand for card in round_state.cards_in_hand()} == {
        "Tomato": 1,
        "Egg": 1,
        "Mushroom": 1,
    }

    assert not round_state.play_attempt([0, 1])

    turns = {card.ingredient.name: card.turns_in_hand for card in round_state.cards_in_hand()}
    assert turns["Tomato"] == 2
    assert turns["Egg"] == 2
    assert turns["Mushroom"] == 1


def test_card_rots_when_turn_limit_reached(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Basil"], hand_size=1)

    round_state.end_turn_decay()
    assert round_state.hand[0] is not None
    assert not round_state.hand[0].is_rotten

    round_state.end_turn_decay()
    assert round_state.hand[0] is not None
    assert round_state.hand[0].is_rotten
    assert round_state.lost


def test_invalid_cook_returns_cards_and_preserves_decay(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Tomato", "Egg", "Mushroom"], hand_size=3)

    round_state.end_turn_decay()
    assert all(card.turns_in_hand == 1 for card in round_state.cards_in_hand())

    original_validator = game_data.is_valid_dish
    game_data.is_valid_dish = lambda _ingredients: False  # type: ignore[assignment]
    try:
        assert not round_state.play_attempt([0, 1, 2])
    finally:
        game_data.is_valid_dish = original_validator

    assert all(card.turns_in_hand == 2 for card in round_state.cards_in_hand())


def test_round_ends_when_basket_empty(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Tomato"], hand_size=1)
    assert round_state.is_round_over()


def test_loss_when_all_hand_slots_rotten(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Basil", "Basil"], hand_size=2)

    round_state.end_turn_decay()
    assert not round_state.lost

    round_state.end_turn_decay()
    assert round_state.lost


def test_loss_when_only_one_fresh_card_remains(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Basil", "Honey"], hand_size=2)

    round_state.end_turn_decay()
    assert not round_state.lost

    round_state.end_turn_decay()
    assert round_state.lost


def test_rot_circles_reflects_state(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Basil"], hand_size=1)
    card = round_state.hand[0]
    assert card is not None

    info = rot_circles(card)
    assert info == {
        "total": game_data.ingredients["Basil"].rotten_turns,
        "filled": 0,
        "cells": ["empty"] * game_data.ingredients["Basil"].rotten_turns,
        "is_rotten": False,
    }

    round_state.end_turn_decay()
    info = rot_circles(card)
    assert info["filled"] == 1
    assert info["cells"][0] == "filled"
    assert not info["is_rotten"]

    round_state.end_turn_decay()
    info = rot_circles(card)
    assert info["filled"] == info["total"]
    assert all(cell == "filled" for cell in info["cells"])
    assert info["is_rotten"]


def test_non_perishables_never_rot(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Honey"], hand_size=1)

    for _ in range(10):
        round_state.end_turn_decay()

    card = round_state.hand[0]
    assert card is not None
    assert not card.is_rotten
    assert card.turns_in_hand == 10
