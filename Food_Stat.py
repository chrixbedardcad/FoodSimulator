"""Generate statistical report for Food Simulator dish outcomes.

This script simulates ingredient draws and computes the chance of
producing every entry in the dish matrix under configurable
conditions. The simulation can use a specific basket and chef roster, or
fall back to a general pool of all ingredients.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
from collections import Counter
from typing import Iterable, List, Optional, Sequence

from food_api import (
    Chef,
    DEFAULT_HAND_SIZE,
    DEFAULT_PICK_SIZE,
    GameData,
    Ingredient,
    build_market_deck,
    draw_cook,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate dish chances and export CSV")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of simulated draws to perform (default: 10000)",
    )
    parser.add_argument(
        "--basket",
        type=str,
        default=None,
        help="Name of the basket to use for the ingredient deck",
    )
    parser.add_argument(
        "--chefs",
        type=str,
        nargs="*",
        default=(),
        help="Chef names to activate during the simulation",
    )
    parser.add_argument(
        "--hand-size",
        type=int,
        default=DEFAULT_HAND_SIZE,
        help="Hand size to simulate (default: %(default)s)",
    )
    parser.add_argument(
        "--pick-size",
        type=int,
        default=DEFAULT_PICK_SIZE,
        help="Maximum number of ingredients selected each draw (default: %(default)s)",
    )
    parser.add_argument(
        "--deck-size",
        type=int,
        default=100,
        help="Number of cards in the simulated deck (default: 100)",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=1.0,
        help="Bias multiplier for key ingredients when using a basket deck",
    )
    parser.add_argument(
        "--guarantee-prob",
        type=float,
        default=0.0,
        help="Probability that a key ingredient is forced into a pick",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("reports", "food_stats.csv"),
        help="Path to the CSV report (default: reports/food_stats.csv)",
    )
    parser.add_argument(
        "--ingredients-json",
        type=str,
        default=None,
        help="Override path to ingredients.json",
    )
    parser.add_argument(
        "--recipes-json",
        type=str,
        default=None,
        help="Override path to recipes.json",
    )
    parser.add_argument(
        "--chefs-json",
        type=str,
        default=None,
        help="Override path to chefs.json",
    )
    parser.add_argument(
        "--taste-json",
        type=str,
        default=None,
        help="Override path to taste_matrix.json",
    )
    parser.add_argument(
        "--baskets-json",
        type=str,
        default=None,
        help="Override path to basket.json",
    )
    parser.add_argument(
        "--dish-matrix-json",
        type=str,
        default=None,
        help="Override path to dish_matrix.json",
    )
    return parser.parse_args()


def load_game_data(args: argparse.Namespace) -> GameData:
    kwargs = {}
    if args.ingredients_json:
        kwargs["ingredients_path"] = args.ingredients_json
    if args.recipes_json:
        kwargs["recipes_path"] = args.recipes_json
    if args.chefs_json:
        kwargs["chefs_path"] = args.chefs_json
    if args.taste_json:
        kwargs["taste_path"] = args.taste_json
    if args.baskets_json:
        kwargs["baskets_path"] = args.baskets_json
    if args.dish_matrix_json:
        kwargs["dish_matrix_path"] = args.dish_matrix_json
    return GameData.from_json(**kwargs)


def resolve_chefs(data: GameData, chef_names: Iterable[str]) -> List[Chef]:
    lookup = {chef.name: chef for chef in data.chefs}
    resolved: List[Chef] = []
    for name in chef_names:
        chef = lookup.get(name)
        if chef:
            resolved.append(chef)
    return resolved


def build_general_deck(
    data: GameData, deck_size: int, rng: random.Random
) -> List[Ingredient]:
    pool = list(data.ingredients.values())
    if not pool:
        return []
    deck: List[Ingredient] = []
    while len(deck) < deck_size:
        take = min(deck_size - len(deck), len(pool))
        deck.extend(rng.sample(pool, take))
    rng.shuffle(deck)
    return deck


def _resolve_basket_name(data: GameData, basket_name: str) -> str:
    """Return the canonical basket name matching ``basket_name``.

    Basket names in ``basket.json`` are capitalized (e.g. ``"Asian"``), but the
    command line argument is easy to provide in a different case.  To make the
    tool friendlier, resolve the name case-insensitively and fall back to the
    original error if no match exists.
    """

    lookup = {name.lower(): name for name in data.baskets}
    resolved = lookup.get(basket_name.lower())
    if resolved:
        return resolved
    raise ValueError(
        "Unknown basket: {}. Available baskets: {}".format(
            basket_name, ", ".join(sorted(data.baskets)) or "<none>"
        )
    )


def build_deck(
    data: GameData,
    basket_name: Optional[str],
    chefs: Sequence[Chef],
    deck_size: int,
    bias: float,
    rng: random.Random,
) -> List[Ingredient]:
    if basket_name:
        basket_name = _resolve_basket_name(data, basket_name)
        return build_market_deck(
            data,
            basket_name,
            chefs,
            deck_size=deck_size,
            bias=bias,
            rng=rng,
        )
    return build_general_deck(data, deck_size, rng)


def simulate(
    data: GameData,
    basket_name: Optional[str],
    chefs: Sequence[Chef],
    iterations: int,
    hand_size: int,
    pick_size: int,
    deck_size: int,
    bias: float,
    guarantee_prob: float,
    rng: random.Random,
) -> Counter[int | str]:
    if iterations <= 0 or pick_size <= 0:
        return Counter()

    counts: Counter[int | str] = Counter()
    denom_by_size: dict[int, int] = {}

    size_set = sorted(
        {e.min_ingredients for e in data.dish_matrix if e.min_ingredients <= pick_size}
    )
    if not size_set:
        return counts

    deck = build_deck(data, basket_name, chefs, deck_size, bias, rng)
    hand: List[Ingredient] = []

    for _ in range(iterations):
        if len(deck) < hand_size:
            deck = build_deck(data, basket_name, chefs, deck_size, bias, rng)
            hand = []

        picks, hand, deck = draw_cook(
            data,
            hand,
            deck,
            chefs,
            guarantee_prob=guarantee_prob,
            hand_size=hand_size,
            pick_size=pick_size,
            rng=rng,
        )
        if not picks:
            continue

        for k in size_set:
            denom_by_size[k] = denom_by_size.get(k, 0) + 1
            if k > len(picks):
                continue

            matched_this_hand: set[int] = set()
            for combo in itertools.combinations(picks, k):
                outcome = data.evaluate_dish(combo)
                if outcome.entry:
                    matched_this_hand.add(outcome.entry.id)

            for entry_id in matched_this_hand:
                counts[entry_id] += 1

    for k, denom in denom_by_size.items():
        counts[f"__denom_{k}__"] = denom
    return counts


def ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def write_report(
    data: GameData,
    counts: Counter,
    output_path: str,
    hand_size: int,
    pick_size: int,
    basket_name: Optional[str],
    chefs: Sequence[Chef],
) -> None:
    ensure_output_dir(output_path)
    fieldnames = [
        "id",
        "name",
        "min_ingredients",
        "max_ingredients",
        "family_pattern",
        "flavor_pattern",
        "multiplier",
        "tier",
        "chance",
        "hand_size",
        "pick_size",
        "basket",
        "chefs",
        "description",
        "occurrences",
        "original_chance",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data.dish_matrix:
            count = counts.get(entry.id, 0)
            k = entry.min_ingredients
            denom = counts.get(f"__denom_{k}__", 0)
            chance = (count / denom) if denom else 0.0
            writer.writerow(
                {
                    "id": entry.id,
                    "name": entry.name,
                    "min_ingredients": entry.min_ingredients,
                    "max_ingredients": entry.max_ingredients,
                    "family_pattern": entry.family_pattern,
                    "flavor_pattern": entry.flavor_pattern,
                    "multiplier": entry.multiplier,
                    "tier": entry.tier,
                    "chance": round(chance, 6),
                    "hand_size": hand_size,
                    "pick_size": pick_size,
                    "basket": basket_name or "",
                    "chefs": ", ".join(chef.name for chef in chefs),
                    "description": entry.description,
                    "occurrences": count,
                    "original_chance": entry.chance,
                }
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data = load_game_data(args)
    chefs = resolve_chefs(data, args.chefs)

    counts = simulate(
        data,
        args.basket,
        chefs,
        args.iterations,
        args.hand_size,
        args.pick_size,
        args.deck_size,
        args.bias,
        args.guarantee_prob,
        rng,
    )
    write_report(
        data,
        counts,
        args.output,
        args.hand_size,
        args.pick_size,
        args.basket,
        chefs,
    )
    print(f"Report saved to {args.output}.")


if __name__ == "__main__":
    main()
