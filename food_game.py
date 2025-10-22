"""Interactive single-player Food Deck test harness.

This script offers a minimal text interface that mirrors the mechanics from
``food_simulator.py`` so that designers can explore turn-by-turn choices before
implementing a richer front end (e.g., in Unreal Engine).

Key features:
* Shares the JSON-driven data sets (ingredients, recipes, chefs, themes).
* Lets the user select chefs and market themes.
* Deals five-card hands, allows picking any trio, and immediately shows score
  breakdowns and recipe hits.
* Supports running multiple short sessions in a row to compare outcomes.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Sequence

from food_simulator import (
    Chef,
    Ingredient,
    build_market_deck,
    chef_key_ingredients,
    initialize_data,
    trio_score,
    which_recipe,
    CHEFS,
    THEMES,
)

# Default gameplay knobs – tweak freely for experimentation.
HAND_SIZE = 5
TRIO_SIZE = 3
DEFAULT_TURNS = 5
DEFAULT_DECK_SIZE = 60
DEFAULT_BIAS = 2.7


def _prompt_selection(prompt: str, options: Sequence[str]) -> str:
    """Display numbered options and return the chosen value."""
    indexed = list(enumerate(options, start=1))
    for idx, value in indexed:
        print(f"  {idx:2d}. {value}")
    while True:
        raw = input(f"{prompt} (number or name, blank for random): ").strip()
        if not raw:
            choice = random.choice(options)
            print(f"  -> Randomly selected: {choice}\n")
            return choice
        if raw.isdigit():
            pos = int(raw)
            if 1 <= pos <= len(options):
                print()
                return options[pos - 1]
        # allow direct name lookup (case-insensitive)
        matches = [value for value in options if value.lower() == raw.lower()]
        if matches:
            print()
            return matches[0]
        print("Invalid selection. Please try again.")


def choose_theme() -> str:
    theme_names = sorted(THEMES.keys())
    print("\n=== Choose a market theme ===")
    return _prompt_selection("Theme", theme_names)


def choose_chef() -> Chef:
    chef_names = [chef.name for chef in CHEFS]
    print("\n=== Choose a chef ===")
    selected_name = _prompt_selection("Chef", chef_names)
    for chef in CHEFS:
        if chef.name == selected_name:
            return chef
    raise RuntimeError("Selected chef not found; data set may be inconsistent.")


def prompt_turn_count() -> int:
    while True:
        raw = input(
            f"How many turns this run? (default {DEFAULT_TURNS}, blank to accept): "
        ).strip()
        if not raw:
            return DEFAULT_TURNS
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Please enter a positive integer.")


def prompt_run_count() -> int:
    while True:
        raw = input("How many runs would you like to play in a row? (default 1): ").strip()
        if not raw:
            return 1
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Please enter a positive integer.")


def describe_ingredient(ingredient: Ingredient, chef_key_set: Iterable[str]) -> str:
    star = "*" if ingredient.name in chef_key_set else " "
    return f"{star}{ingredient.name} (Taste: {ingredient.taste}, Chips: {ingredient.chips})"


def display_hand(hand: Sequence[Ingredient], chef: Chef) -> None:
    print("\nYour hand:")
    key_set = chef_key_ingredients(chef)
    for idx, ingredient in enumerate(hand, start=1):
        print(f"  {idx}. {describe_ingredient(ingredient, key_set)}")
    print("  * indicates an ingredient that appears in the chef's signature recipes.")


def prompt_trio(hand: Sequence[Ingredient]) -> List[Ingredient] | None:
    while True:
        raw = input(
            "Select three cards by number (e.g., 1 3 4). Enter 'q' to end the run: "
        ).strip()
        if raw.lower() == "q":
            return None
        tokens = [token for token in raw.replace(",", " ").split(" ") if token]
        if len(tokens) != TRIO_SIZE:
            print(f"Please choose exactly {TRIO_SIZE} distinct cards.")
            continue
        try:
            picks = [int(token) for token in tokens]
        except ValueError:
            print("Selections must be numeric indices.")
            continue
        if any(p < 1 or p > len(hand) for p in picks):
            print("One or more selections are out of range.")
            continue
        if len(set(picks)) != TRIO_SIZE:
            print("Please avoid duplicate selections.")
            continue
        return [hand[p - 1] for p in picks]


def score_trio(selected: Sequence[Ingredient], chef: Chef) -> int:
    score, chips, taste_sum, multiplier = trio_score(list(selected))
    recipe_name = which_recipe(list(selected))
    chef_hits = sum(1 for ing in selected if ing.name in chef_key_ingredients(chef))
    print("\n--- Trio Result ---")
    for ing in selected:
        print(f"  {ing.name} (Taste: {ing.taste}, Chips: {ing.chips})")
    print(f"Total chips: {chips}")
    print(f"Taste synergy sum: {taste_sum}")
    print(f"Multiplier applied: x{multiplier}")
    print(f"Score gained: {score}")
    if recipe_name:
        print(f"Recipe completed: {recipe_name}")
    else:
        print("No recipe completed this turn.")
    print(f"Chef key ingredients used: {chef_hits}/{TRIO_SIZE}\n")
    return score


def play_single_run(theme_name: str, chef: Chef, turns: int) -> int:
    theme_pool = THEMES[theme_name]
    deck = build_market_deck(
        theme_pool, chef, deck_size=DEFAULT_DECK_SIZE, bias=DEFAULT_BIAS
    )
    random.shuffle(deck)
    total_score = 0

    for turn in range(1, turns + 1):
        if len(deck) < HAND_SIZE:
            deck = build_market_deck(
                theme_pool, chef, deck_size=DEFAULT_DECK_SIZE, bias=DEFAULT_BIAS
            )
            random.shuffle(deck)
            print("\n-- Deck refreshed --")

        hand = [deck.pop() for _ in range(HAND_SIZE)]
        print(f"\n=== Turn {turn}/{turns} ===")
        display_hand(hand, chef)
        selected = prompt_trio(hand)
        if selected is None:
            print("Run ended early by player choice.\n")
            break
        gained = score_trio(selected, chef)
        total_score += gained
    return total_score


def main() -> None:
    print("""
===============================================
 Food Deck Mini Game (terminal prototype)
-----------------------------------------------
 * Data shared with food_simulator.py (JSON-driven).
 * Pick a chef and theme, draw 5 ingredients per turn,
   and choose any 3 to cook a trio.
 * Scores are computed using the same trio scoring model.
 * Enter 'q' during a turn to stop the current run early.
===============================================
""")

    initialize_data()

    while True:
        theme = choose_theme()
        chef = choose_chef()
        turns = prompt_turn_count()
        runs = prompt_run_count()
        seed_raw = input("Optional RNG seed (blank for random): ").strip()
        if seed_raw:
            random.seed(int(seed_raw))
        else:
            random.seed()

        for run_index in range(1, runs + 1):
            print(f"\n>>> Starting run {run_index}/{runs} with Chef {chef.name} in {theme} <<<")
            score = play_single_run(theme, chef, turns)
            print(f"Final score for run {run_index}: {score}\n")

        again = input("Play another configuration? (y/N): ").strip().lower()
        if again not in {"y", "yes"}:
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
