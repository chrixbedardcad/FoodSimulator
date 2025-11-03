from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from food_api import GameData
from food_desktop import GameSession
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
        hand_size=hand_size,
        pick_size=pick_size,
        deck_size=max(10, hand_size),
        bias=0.0,
    )
    session.hand = [IngredientCard(ingredient=data.ingredients[name]) for name in names]
    return session


def test_rotten_cards_ruin_dish_and_finish_run_when_empty(game_data: GameData) -> None:
    recipe = game_data.recipes[0]
    names = list(recipe.trio)
    session = make_session(game_data, names)
    session.deck.clear()
    session.consume_events()

    for card in session.hand:
        card.is_rotten = True

    assert not session.finished

    outcome = session.play_turn([0, 1, 2])

    total_value = sum(game_data.ingredients[name].Value for name in names)
    expected_penalty = -(total_value * len(names))

    assert outcome.final_score == expected_penalty
    assert outcome.recipe_name is None
    assert outcome.ruined
    assert session.finished

    events = session.consume_events()
    assert any("Rotten ingredients spoiled the dish" in message for message in events)


def test_nonperishable_cards_never_rot(game_data: GameData) -> None:
    session = make_session(game_data, ["Honey", "Honey", "Honey"])

    for _ in range(5):
        session._apply_end_turn_decay()

    assert all(not card.is_rotten for card in session.hand)
    assert not session.finished


def test_returning_rotten_card_not_allowed(game_data: GameData) -> None:
    session = make_session(game_data, ["Basil", "Tomato", "Egg"])
    rotten_card = session.hand[0]
    rotten_card.is_rotten = True
    rotten_card.turns_in_hand = max(rotten_card.ingredient.rotten_turns, 0)

    with pytest.raises(ValueError):
        session.return_indices([0])


def test_cook_without_recipe_scores_total_value(game_data: GameData) -> None:
    session = make_session(game_data, ["Truffle", "Egg", "Basil"])
    original_cards = list(session.hand)
    total_value = sum(card.ingredient.Value for card in original_cards)

    outcome = session.play_turn([0, 1, 2])

    assert outcome.recipe_name is None
    assert outcome.dish_name is None
    assert outcome.final_score == total_value
    assert session.turn_number == 1
    assert len(session.hand) == session.hand_size

    events = session.consume_events()
    assert all("returned to the basket" not in message for message in events)


def test_all_same_ingredient_selection_scores_value(game_data: GameData) -> None:
    session = make_session(game_data, ["Basil", "Basil", "Basil"])
    cards = list(session.hand)
    total_value = sum(card.ingredient.Value for card in cards)

    outcome = session.play_turn([0, 1, 2])

    assert outcome.recipe_name is None
    assert outcome.dish_name is None
    assert outcome.final_score == total_value
    assert session.turn_number == 1


def test_invalid_selection_ages_remaining_cards(game_data: GameData) -> None:
    session = make_session(game_data, ["Truffle", "Egg", "Basil", "Tomato"])
    lingering_card = session.hand[-1]

    outcome = session.play_turn([0, 1, 2])

    assert outcome.final_score > 0
    assert lingering_card in session.hand
    assert lingering_card.turns_in_hand == 1



def test_return_action_counts_as_turn(game_data: GameData) -> None:
    session = make_session(game_data, ["Tomato", "Basil", "Basil", "Mushroom"])
    returned_card = session.hand[0]
    lingering_card = session.hand[1]
    extra_card = session.hand[2]

    returned_card.turns_in_hand = 2
    lingering_card.turns_in_hand = 1
    extra_card.turns_in_hand = 0

    removed, _ = session.return_indices([0])

    assert removed[0].name == returned_card.ingredient.name
    assert returned_card.turns_in_hand == 2
    assert not returned_card.is_rotten
    assert lingering_card.turns_in_hand == 2
    assert lingering_card.is_rotten
    assert extra_card.turns_in_hand == 1


def test_basket_cards_preserve_decay_state(game_data: GameData) -> None:
    session = make_session(game_data, ["Honey", "Honey", "Honey"])

    basil_card = IngredientCard(ingredient=game_data.ingredients["Basil"])
    basil_card.turns_in_hand = 2
    session.hand[0] = basil_card

    removed, _ = session.return_indices([0])

    assert removed[0].name == "Basil"
    assert basil_card.turns_in_hand == 2
    assert not basil_card.is_rotten


def test_cleanup_must_be_confirmed_before_next_round(game_data: GameData) -> None:
    basket_name = next(iter(game_data.baskets))
    session = GameSession(
        game_data,
        basket_name=basket_name,
        chefs=[],
        rounds=2,
        hand_size=2,
        pick_size=2,
        deck_size=0,
        bias=0.0,
    )

    session.consume_events()

    fresh_card = IngredientCard(ingredient=game_data.ingredients["Tomato"])
    rotten_hand = IngredientCard(ingredient=game_data.ingredients["Basil"])
    rotten_hand.is_rotten = True
    rotten_hand.turns_in_hand = max(rotten_hand.ingredient.rotten_turns, 0)
    rotten_deck = IngredientCard(ingredient=game_data.ingredients["Egg"])
    rotten_deck.is_rotten = True
    rotten_deck.turns_in_hand = max(rotten_deck.ingredient.rotten_turns, 0)

    session.hand = [fresh_card, rotten_hand]
    session.deck = [rotten_deck]

    session._enter_round_summary()

    assert session.needs_cleanup_confirmation()
    pending = {ingredient.name for ingredient in session.pending_cleanup_ingredients()}
    assert pending == {"Basil", "Egg"}

    with pytest.raises(RuntimeError):
        session.begin_next_round_after_reward()

    removed = session.acknowledge_cleanup()
    assert {ingredient.name for ingredient in removed} == {"Basil", "Egg"}
    assert not session.needs_cleanup_confirmation()

    events = session.consume_events()
    assert any("Removed rotten" in message for message in events)

    session.begin_next_round_after_reward()

    assert all(not card.is_rotten for card in session.hand)
    assert all(not card.is_rotten for card in session.deck)
