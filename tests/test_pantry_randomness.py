import random
from typing import Dict, List, Sequence, Tuple

from food_api import GameData, Ingredient, build_market_deck
from food_desktop import GameSession
from rotting_round import IngredientCard


class TrackingRandom(random.Random):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.choice_history: List[Tuple[str, ...]] = []

    def choice(self, seq: Sequence[str]):  # type: ignore[override]
        snapshot = tuple(seq)
        self.choice_history.append(snapshot)
        return super().choice(seq)


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


def _assert_choice_history_matches_counts(
    history: Sequence[Tuple[str, ...]],
    deck_names: Sequence[str],
    initial_counts: Dict[str, int],
) -> None:
    counts = dict(initial_counts)
    assert len(history) == len(deck_names)
    for snapshot, name in zip(history, deck_names):
        assert tuple(counts.keys()) == snapshot
        assert name in counts
        counts[name] -= 1
        if counts[name] <= 0:
            del counts[name]


def test_build_market_deck_draws_uniform_unique_choices() -> None:
    counts = {"Honey": 10, "Apple": 1, "Salt": 1, "Pepper": 1, "Lime": 1}
    data = _make_game_data(counts)
    rng = TrackingRandom(1234)

    deck = build_market_deck(
        data,
        "Test",
        chefs=[],
        deck_size=sum(counts.values()),
        bias=1.0,
        rng=rng,
    )

    deck_names = [ingredient.name for ingredient in deck]
    assert len(deck_names) == sum(counts.values())
    _assert_choice_history_matches_counts(rng.choice_history, deck_names, counts)


def test_rebalance_deck_preserves_uniform_draws() -> None:
    counts = {"Honey": 6, "Apple": 1, "Salt": 1, "Pepper": 1}
    data = _make_game_data(counts)
    rng = TrackingRandom(4321)

    session = GameSession(
        data,
        basket_name="Test",
        chefs=[],
        rounds=1,
        hand_size=1,
        pick_size=1,
        deck_size=0,
        rng=rng,
    )

    rng.choice_history.clear()

    cards = [
        IngredientCard(ingredient=data.ingredients[name])
        for name, copies in counts.items()
        for _ in range(copies)
    ]

    balanced = session._rebalance_deck(cards)

    assert {id(card) for card in balanced} == {id(card) for card in cards}

    deck_names = [card.ingredient.name for card in balanced]
    _assert_choice_history_matches_counts(rng.choice_history, deck_names, counts)
