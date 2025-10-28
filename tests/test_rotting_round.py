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
    round_state = make_round(game_data, ["Basil", "Basil", "Basil"], hand_size=3)

    round_state.end_turn_decay()
    assert not round_state.lost

    round_state.end_turn_decay()
    assert round_state.lost


def test_loss_when_only_one_fresh_card_remains(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Basil", "Basil", "Honey"], hand_size=3)

    round_state.end_turn_decay()
    assert not round_state.lost

    round_state.end_turn_decay()
    assert round_state.lost


def test_loss_when_not_enough_fresh_cards_to_cook(game_data: GameData) -> None:
    round_state = make_round(
        game_data,
        ["Basil", "Basil", "Honey", "Honey"],
        hand_size=4,
    )

    round_state.end_turn_decay()
    assert not round_state.lost

    round_state.end_turn_decay()
    assert round_state.lost


def test_loss_checked_after_play_attempt(game_data: GameData) -> None:
    round_state = make_round(
        game_data,
        ["Basil", "Basil", "Honey", "Honey"],
        hand_size=3,
    )

    # Mark the first two cards as rotten so only one fresh card remains once drawn.
    for card in round_state.hand[:2]:
        assert card is not None
        card.is_rotten = True

    assert not round_state.lost

    # Attempt to play with the available indices. The rotten cards are ignored, so the
    # cook fails, the cards are returned, and the hand is refilled.
    round_state.play_attempt([0, 1, 2])

    assert round_state.lost


def test_loss_requires_low_fresh_and_no_recipe(game_data: GameData) -> None:
    round_state = make_round(game_data, ["Tomato", "Egg", "Mushroom"], hand_size=3)

    original_validator = game_data.is_valid_dish
    game_data.is_valid_dish = lambda _ingredients: False  # type: ignore[assignment]
    try:
        round_state._update_loss_state()
        assert not round_state.lost

        for card in round_state.hand[:2]:
            assert card is not None
            card.is_rotten = True

        round_state._update_loss_state()
        assert round_state.lost
    finally:
        game_data.is_valid_dish = original_validator


def test_prep_one_once_per_turn(game_data: GameData) -> None:
    round_state = make_round(
        game_data,
        ["Tomato", "Egg", "Mushroom", "Basil", "Rice"],
        hand_size=3,
    )

    first_card = round_state.hand[0]
    assert first_card is not None
    original_turns = first_card.turns_in_hand

    assert round_state.prep_one(0)
    assert round_state.prep_used_this_turn

    drawn_card = round_state.hand[0]
    assert drawn_card is not None
    assert drawn_card.ingredient.name != first_card.ingredient.name

    moved_card = round_state.basket[-1]
    assert moved_card.ingredient.name == first_card.ingredient.name
    assert moved_card.turns_in_hand == original_turns

    assert not round_state.prep_one(1)

    round_state.end_turn_decay()
    assert not round_state.prep_used_this_turn
    assert round_state.prep_one(0)


def test_compost_removes_rotten_and_respects_cooldown(game_data: GameData) -> None:
    round_state = make_round(
        game_data,
        ["Basil", "Basil", "Honey", "Pasta", "Rice", "Tomato"],
        hand_size=5,
    )

    round_state.end_turn_decay()
    round_state.end_turn_decay()

    rotten_card = round_state.hand[0]
    assert rotten_card is not None and rotten_card.is_rotten
    assert not round_state.lost

    assert round_state.compost(0)
    assert round_state.compost_cd == 3
    assert round_state.hand[0] is not None

    assert not round_state.compost(0)

    for _ in range(3):
        round_state.end_turn_decay()
    assert round_state.compost_cd == 0

    # Manually rot a card to ensure compost can trigger again once cooldown ends.
    next_card = round_state.hand[0]
    assert next_card is not None
    next_card.is_rotten = True

    assert round_state.compost(0)


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
