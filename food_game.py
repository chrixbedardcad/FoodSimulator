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
from typing import Collection, Iterable, List, Mapping, Sequence, Tuple

from food_api import Chef, Ingredient, RULES_VERSION, GameData, build_market_deck
from seed_utils import resolve_seed

DATA = GameData.from_json()

# Default gameplay knobs – tweak freely for experimentation.
HAND_SIZE = 5
TRIO_SIZE = 3
DEFAULT_TURNS = 5
DEFAULT_DECK_SIZE = 60
DEFAULT_BIAS = 2.7


def _prompt_selection(prompt: str, options: Sequence[str], rng: random.Random) -> str:
    """Display numbered options and return the chosen value."""
    indexed = list(enumerate(options, start=1))
    for idx, value in indexed:
        print(f"  {idx:2d}. {value}")
    while True:
        raw = input(f"{prompt} (number or name, blank for random): ").strip()
        if not raw:
            choice = rng.choice(options)
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


def choose_theme(rng: random.Random) -> str:
    theme_names = sorted(DATA.themes.keys())
    print("\n=== Choose a market theme ===")
    return _prompt_selection("Theme", theme_names, rng)


def choose_chef(rng: random.Random) -> Chef:
    chef_names = [chef.name for chef in DATA.chefs]
    print("\n=== Choose a chef ===")
    selected_name = _prompt_selection("Chef", chef_names, rng)
    for chef in DATA.chefs:
        if chef.name == selected_name:
            return chef
    raise RuntimeError("Selected chef not found; data set may be inconsistent.")


def choose_additional_chef(
    existing_names: Iterable[str], rng: random.Random
) -> Chef | None:
    """Prompt the user to add a new chef, excluding already selected ones."""

    taken = {name.lower() for name in existing_names}
    available = [chef.name for chef in DATA.chefs if chef.name.lower() not in taken]
    if not available:
        print("No additional chefs available to add.")
        return None

    print("\n=== Add a new chef to your lineup ===")
    selected_name = _prompt_selection("New Chef", available, rng)
    for chef in DATA.chefs:
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


def prompt_seed() -> Tuple[int, random.Random]:
    """Prompt the user for a reproducible RNG seed."""

    while True:
        seed_raw = input("Optional RNG seed (blank for random): ").strip()
        if not seed_raw:
            return resolve_seed(None)
        try:
            typed_seed = int(seed_raw)
        except ValueError:
            print("Please enter a valid integer seed.")
            continue
        return resolve_seed(typed_seed)


def _chef_marker(chef: Chef) -> str:
    """Return a single-letter marker representing the chef."""

    parts = [part for part in chef.name.split() if part]
    for part in parts:
        if part.lower() != "chef":
            return part[0].upper()
    return parts[0][0].upper() if parts else "?"


def describe_ingredient(
    ingredient: Ingredient,
    chefs: Sequence[Chef],
    chef_key_map: Mapping[str, Collection[str]],
) -> str:
    markers = [
        _chef_marker(chef)
        for chef in chefs
        if ingredient.name in chef_key_map.get(chef.name, ())
    ]
    marker_text = f"(*{''.join(markers)})" if markers else ""
    prefix = f"{marker_text} " if marker_text else ""
    return f"{prefix}{ingredient.name} (Taste: {ingredient.taste}, Chips: {ingredient.chips})"


def display_hand(hand: Sequence[Ingredient], chefs: Sequence[Chef]) -> None:
    print("\nYour hand:")
    chef_key_map = {chef.name: DATA.chef_key_ingredients(chef) for chef in chefs}
    for idx, ingredient in enumerate(hand, start=1):
        print(f"  {idx}. {describe_ingredient(ingredient, chefs, chef_key_map)}")
    chef_names = ", ".join(chef.name for chef in chefs)
    print(
        "  (*X) next to an ingredient indicates it appears in an active chef's signature recipes."
    )
    if chefs:
        legend = ", ".join(f"(*{_chef_marker(chef)}) {chef.name}" for chef in chefs)
        print(f"    Legend: {legend}")
    print(f"    Active chefs: {chef_names}")


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


def _chef_recipe_multiplier(chef: Chef, recipe_name: str | None) -> float:
    if not recipe_name:
        return 1.0
    multipliers = chef.perks.get("recipe_multipliers", {})
    try:
        return float(multipliers.get(recipe_name, 1.0))
    except (TypeError, ValueError):
        return 1.0


def _team_recipe_multiplier(
    chefs: Sequence[Chef], recipe_name: str | None
) -> Tuple[float, List[Tuple[str, float]]]:
    if not recipe_name:
        return 1.0, []
    contributions: List[Tuple[str, float]] = []
    total = 1.0
    for chef in chefs:
        multiplier = _chef_recipe_multiplier(chef, recipe_name)
        total *= multiplier
        if multiplier != 1.0:
            contributions.append((chef.name, multiplier))
    return total, contributions


def score_trio(selected: Sequence[Ingredient], chefs: Sequence[Chef]) -> int:
    base_score, chips, taste_sum, taste_multiplier = DATA.trio_score(list(selected))
    recipe_name = DATA.which_recipe(list(selected))
    recipe_multiplier, contributions = _team_recipe_multiplier(chefs, recipe_name)
    total_multiplier = taste_multiplier * recipe_multiplier
    final_score = int(round(base_score * recipe_multiplier))
    key_set = DATA.chefs_key_ingredients(chefs)
    chef_hits = sum(1 for ing in selected if ing.name in key_set)

    print("\n--- Trio Result ---")
    for ing in selected:
        print(f"  {ing.name} (Taste: {ing.taste}, Chips: {ing.chips})")
    print(f"Total chips: {chips}")
    print(f"Taste synergy sum: {taste_sum}")
    print(f"Taste multiplier: x{taste_multiplier}")
    if recipe_name:
        print(f"Recipe completed: {recipe_name}")
    else:
        print("No recipe completed this turn.")
    if contributions:
        breakdown = ", ".join(f"{name}: x{mult:.2f}" for name, mult in contributions)
        print(f"Recipe multiplier: x{recipe_multiplier:.2f} ({breakdown})")
    else:
        print(f"Recipe multiplier: x{recipe_multiplier:.2f}")
    print(f"Total multiplier: x{total_multiplier:.2f}")
    print(f"Score gained: {final_score} (base score before recipe bonus: {base_score})")
    active_names = ", ".join(chef.name for chef in chefs)
    print(f"Chef key ingredients used: {chef_hits}/{TRIO_SIZE} (Active: {active_names})\n")
    return final_score


def play_single_run(
    theme_name: str, chefs: Sequence[Chef], turns: int, rng: random.Random
) -> int:
    total_score = 0
    hand: List[Ingredient] = []
    if not chefs:
        raise ValueError("At least one chef must be active for a run.")
    deck = build_market_deck(
        DATA,
        theme_name,
        chefs,
        deck_size=DEFAULT_DECK_SIZE,
        bias=DEFAULT_BIAS,
        rng=rng,
    )
    rng.shuffle(deck)
    print("\nActive chefs this run: " + ", ".join(chef.name for chef in chefs))
    for turn in range(1, turns + 1):
        needed = HAND_SIZE - len(hand)
        if needed > 0:
            if len(deck) < needed:
                deck = build_market_deck(
                    DATA,
                    theme_name,
                    chefs,
                    deck_size=DEFAULT_DECK_SIZE,
                    bias=DEFAULT_BIAS,
                    rng=rng,
                )
                rng.shuffle(deck)
                print("\n-- Deck refreshed for the team --")
            hand.extend(deck.pop() for _ in range(needed))
        print(f"\n=== Turn {turn}/{turns} ===")
        display_hand(hand, chefs)
        selected = prompt_trio(hand)
        if selected is None:
            print("Run ended early by player choice.\n")
            break
        gained = score_trio(selected, chefs)
        total_score += gained
        print(f"Cumulative score: {total_score}\n")
        for ingredient in selected:
            hand.remove(ingredient)
    return total_score


def main() -> None:
    print(
        f"""
===============================================
 Food Deck Mini Game (terminal prototype)
 Version: {RULES_VERSION}
-----------------------------------------------
 * Data shared with food_simulator.py (JSON-driven).
 * Pick a chef and theme, draw 5 ingredients per turn,
   and choose any 3 to cook a trio.
 * Scores are computed using the same trio scoring model.
 * Enter 'q' during a turn to stop the current run early.
===============================================
"""
    )

    while True:
        seed_value, rng = prompt_seed()
        print(
            f"\nUsing RNG seed: {seed_value}\n"
            "  (Pass this seed to food_simulator.py with --seed to reproduce runs.)\n"
        )

        theme = choose_theme(rng)
        chef = choose_chef(rng)
        turns = prompt_turn_count()
        runs = prompt_run_count()
        roster: List[Chef] = [chef]

        for run_index in range(1, runs + 1):
            chef_names = ", ".join(c.name for c in roster)
            print(
                f"\n>>> Starting run {run_index}/{runs} with Chefs {chef_names} in {theme} <<<"
            )
            score = play_single_run(theme, roster, turns, rng)
            print(f"Final score for run {run_index}: {score}\n")

            if len(roster) < len(DATA.chefs):
                add_choice = input(
                    "Add a new chef to your lineup for the next run? (y/N): "
                ).strip().lower()
                if add_choice in {"y", "yes"}:
                    new_chef = choose_additional_chef([c.name for c in roster], rng)
                    if new_chef and new_chef not in roster:
                        roster.append(new_chef)
                        print(
                            f"Chef {new_chef.name} has joined your team for future runs."
                        )

        again = input("Play another configuration? (y/N): ").strip().lower()
        if again not in {"y", "yes"}:
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
