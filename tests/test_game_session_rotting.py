from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from food_api import GameData
from food_desktop import GameSession, InvalidDishSelection
from rotting_round import IngredientCard


@pytest.fixture
def game_data() -> GameData:
    return GameData.from_json()


def make_session(data: GameData, names: list[str]) -> GameSession:
    basket_name = next(iter(data.baskets))
    hand_size = max(len(names), 3)
    pick_size = min(5, hand_size)
    session = GameSession(
        data,
        basket_name=basket_name,
        chefs=[],
        rounds=1,
        cooks_per_round=1,
        hand_size=hand_size,
        pick_size=pick_size,
        deck_size=max(10, hand_size),
        bias=0.0,
    )
    session.hand = [IngredientCard(ingredient=data.ingredients[name]) for name in names]
    return session


def test_decay_marks_cards_rotten_and_finishes_run(game_data: GameData) -> None:
    session = make_session(game_data, ["Basil", "Basil", "Basil"])

    session._apply_end_turn_decay()
    assert [card.turns_in_hand for card in session.hand] == [1, 1, 1]
    session.consume_events()

    session._apply_end_turn_decay()
    assert all(card.is_rotten for card in session.hand)
    assert session.finished

    events = session.consume_events()
    assert any("has gone rotten" in message for message in events)
    assert any("run is over" in message for message in events)


def test_nonperishable_cards_never_rot(game_data: GameData) -> None:
    session = make_session(game_data, ["Honey", "Honey", "Honey"])

    for _ in range(5):
        session._apply_end_turn_decay()

    assert all(not card.is_rotten for card in session.hand)
    assert not session.finished


def test_discarding_rotten_card_not_allowed(game_data: GameData) -> None:
    session = make_session(game_data, ["Basil", "Tomato", "Egg"])
    rotten_card = session.hand[0]
    rotten_card.is_rotten = True
    rotten_card.turns_in_hand = max(rotten_card.ingredient.rotten_turns, 0)

    with pytest.raises(ValueError):
        session.discard_indices([0])


def test_invalid_cook_returns_cards_and_increments_decay(game_data: GameData) -> None:
    """Invalid dishes send the ingredients back to the basket and advance rot."""

    session = make_session(game_data, ["Truffle", "Egg", "Basil"])
    original_cards = list(session.hand)

    with pytest.raises(InvalidDishSelection) as exc_info:
        session.play_turn([0, 1, 2])

    assert "do not form a dish" in str(exc_info.value)
    assert session.turn_number == 0
    assert len(session.hand) == len(original_cards)

    for card in original_cards:
        assert card.turns_in_hand == 1

    locations = session.hand + list(session.deck)
    for card in original_cards:
        assert card in locations

    events = session.consume_events()
    assert any("returned to the basket" in message for message in events)

    ingredients = getattr(exc_info.value, "ingredients", ())
    assert ingredients
    assert len(ingredients) == len(original_cards)
    assert all(card.ingredient == ingredient for card, ingredient in zip(original_cards, ingredients))


def test_all_same_ingredient_selection_rejected(game_data: GameData) -> None:
    session = make_session(game_data, ["Basil", "Basil", "Basil"])
    original_cards = list(session.hand)

    with pytest.raises(InvalidDishSelection) as exc_info:
        session.play_turn([0, 1, 2])

    exception = exc_info.value
    assert exception.primary_ingredient is not None
    assert exception.primary_ingredient.name == "Basil"
    assert "do not form a dish" in str(exception)

    # Cards should be returned to the basket and gain a rot turn.
    assert session.turn_number == 0
    locations = session.hand + list(session.deck)
    for card in original_cards:
        assert card.turns_in_hand == 1
        assert card in locations

    assert exception.ingredients
    assert len(exception.ingredients) == len(original_cards)
    assert all(ingredient.name == "Basil" for ingredient in exception.ingredients)


def test_invalid_selection_ages_remaining_cards(game_data: GameData) -> None:
    session = make_session(game_data, ["Truffle", "Egg", "Basil", "Tomato"])
    lingering_card = session.hand[-1]

    with pytest.raises(InvalidDishSelection):
        session.play_turn([0, 1, 2])

    assert lingering_card in session.hand
    assert lingering_card.turns_in_hand == 1


def test_basket_cards_preserve_decay_state(game_data: GameData) -> None:
    session = make_session(game_data, ["Honey", "Honey", "Honey"])

    basil_card = IngredientCard(ingredient=game_data.ingredients["Basil"])
    basil_card.turns_in_hand = 1
    session.deck.append(basil_card)

    for _ in range(3):
        session._apply_end_turn_decay()

    assert basil_card.turns_in_hand == 1
    assert not basil_card.is_rotten
