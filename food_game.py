"""Interactive single-player Food Deck test harness.

This script offers a minimal text interface that mirrors the mechanics from
``food_simulator.py`` so that designers can explore turn-by-turn choices before
implementing a richer front end (e.g., in Unreal Engine).

Key features:
* Shares the JSON-driven data sets (ingredients, recipes, chefs, themes).
* Lets the user select chefs and market themes.
* Uses the same round → cook structure as the automated simulator so a "run"
  represents a full game composed of multiple rounds, each with several cooks.
* Deals configurable hand sizes (default eight cards), allows picking any trio size,
  and immediately shows score breakdowns and recipe hits.
* Supports running multiple short sessions in a row to compare outcomes.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Collection, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from family_icons import get_family_icon
from food_api import (
    Chef,
    Ingredient,
    RULES_VERSION,
    GameData,
    SimulationConfig,
    build_market_deck,
)
from taste_icons import get_taste_icon
from seed_utils import resolve_seed

DATA = GameData.from_json()

# Default gameplay knobs – shared with the simulator for consistent semantics.
DEFAULT_CONFIG = SimulationConfig()
DEFAULT_HAND_SIZE = DEFAULT_CONFIG.hand_size
DEFAULT_PICK_SIZE = DEFAULT_CONFIG.pick_size
DEFAULT_ROUNDS = DEFAULT_CONFIG.rounds
DEFAULT_COOKS_PER_ROUND = DEFAULT_CONFIG.cooks
DEFAULT_DECK_SIZE = DEFAULT_CONFIG.deck_size
DEFAULT_BIAS = DEFAULT_CONFIG.bias
DEFAULT_MAX_CHEFS = DEFAULT_CONFIG.active_chefs


def format_multiplier(multiplier: float) -> str:
    rounded = round(multiplier)
    if abs(multiplier - rounded) < 1e-9:
        return f"x{int(rounded)}"
    return f"x{multiplier:.2f}"


def describe_taste_and_family(ingredient: Ingredient) -> str:
    """Return a combined taste/family string with optional icons."""

    taste_icon = get_taste_icon(ingredient.taste)
    family_icon = get_family_icon(ingredient.family)

    taste_part = (
        f"Taste: {taste_icon} {ingredient.taste}"
        if taste_icon
        else f"Taste: {ingredient.taste}"
    )
    family_part = (
        f"Family: {family_icon} {ingredient.family}"
        if family_icon
        else f"Family: {ingredient.family}"
    )
    return f"{taste_part}, {family_part}"


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


def prompt_round_count() -> int:
    while True:
        raw = input(
            f"How many rounds per run? (default {DEFAULT_ROUNDS}, blank to accept): "
        ).strip()
        if not raw:
            return DEFAULT_ROUNDS
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Please enter a positive integer.")


def prompt_cooks_per_round() -> int:
    while True:
        raw = input(
            "How many cooks in each round? "
            f"(default {DEFAULT_COOKS_PER_ROUND}, blank to accept): "
        ).strip()
        if not raw:
            return DEFAULT_COOKS_PER_ROUND
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


def prompt_hand_size() -> int:
    while True:
        raw = input(
            "How many cards per hand? "
            f"(default {DEFAULT_HAND_SIZE}, blank to accept): "
        ).strip()
        if not raw:
            if DEFAULT_HAND_SIZE <= 0:
                print("Hand size must be a positive integer.")
                continue
            return DEFAULT_HAND_SIZE
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Please enter a positive integer.")


def prompt_pick_size(hand_size: int) -> int:
    while True:
        raw = input(
            "How many cards should be selected each turn? "
            f"(default {DEFAULT_PICK_SIZE}, blank to accept): "
        ).strip()
        if not raw:
            pick_size = DEFAULT_PICK_SIZE
            if pick_size > hand_size:
                print(
                    "Default pick size exceeds the chosen hand size; "
                    f"using {hand_size} instead."
                )
                pick_size = hand_size
        elif raw.isdigit() and int(raw) > 0:
            pick_size = int(raw)
        else:
            print("Please enter a positive integer.")
            continue

        if pick_size > hand_size:
            print("Pick size cannot exceed the hand size. Please choose a smaller number.")
            continue
        return pick_size


def prompt_max_chefs() -> int:
    while True:
        raw = input(
            f"What is the maximum number of chefs you can recruit? "
            f"(default {DEFAULT_MAX_CHEFS}, blank to accept): "
        ).strip()
        if not raw:
            return max(1, DEFAULT_MAX_CHEFS)
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

def describe_ingredient(
    ingredient: Ingredient,
    chefs: Sequence[Chef],
    chef_key_map: Mapping[str, Collection[str]],
    cookbook_ingredients: Collection[str],
) -> List[str]:
    chef_names = [
        chef.name
        for chef in chefs
        if ingredient.name in chef_key_map.get(chef.name, ())
    ]
    book = " 📖" if ingredient.name in cookbook_ingredients else ""
    taste_family = describe_taste_and_family(ingredient)
    lines = [f"{ingredient.name}{book} ({taste_family}, Value: {ingredient.Value})"]
    if chef_names:
        first, *rest = chef_names
        lines.append(f"Chef Key: {first}")
        lines.extend(f"           {name}" for name in rest)
    recipes = list(DATA.recipes_using_ingredient(ingredient.name))
    if recipes:
        lines.append("Recipes: " + ", ".join(recipes))
    else:
        lines.append("Recipes: (none)")
    return lines


def display_hand(
    hand: Sequence[Ingredient], chefs: Sequence[Chef], cookbook: Mapping[str, Sequence[str]]
) -> None:
    print("\nYour hand:")
    chef_key_map = {chef.name: DATA.chef_key_ingredients(chef) for chef in chefs}
    cookbook_ingredients = {
        ingredient
        for combo in cookbook.values()
        for ingredient in combo
    }
    for idx, ingredient in enumerate(hand, start=1):
        description = describe_ingredient(
            ingredient, chefs, chef_key_map, cookbook_ingredients
        )
        if not description:
            continue
        first, *rest = description
        print(f"  {idx}. {first}")
        for line in rest:
            print(f"       {line}")
    chef_names = ", ".join(chef.name for chef in chefs)
    if chefs:
        print(
            "    Chef Key lines list which active chefs feature the ingredient in a signature recipe."
        )
    print(f"    Active chefs: {chef_names or 'None'}")


def print_cookbook(
    cookbook: Mapping[str, Sequence[str]],
    counts: Mapping[str, int] | None = None,
    title: str = "Cookbook",
    chefs: Sequence[Chef] | None = None,
) -> None:
    print(f"\n=== {title} ===")
    if not cookbook:
        print("  No recipes discovered yet.")
        return
    for recipe_name in sorted(cookbook):
        ingredients = ", ".join(cookbook[recipe_name])
        count_text = ""
        if counts:
            cooked = counts.get(recipe_name, 0)
            if cooked:
                times = "time" if cooked == 1 else "times"
                count_text = f" (cooked {cooked} {times})"
        multiplier = DATA.recipe_multiplier(
            recipe_name,
            chefs=chefs,
            times_cooked=counts.get(recipe_name, 0) if counts else 0,
        )
        print(
            f"  {recipe_name}: {ingredients} — multiplier {format_multiplier(multiplier)}{count_text}"
        )


def prompt_trio(hand: Sequence[Ingredient], pick_size: int) -> List[Ingredient] | None:
    while True:
        raw = input(
            f"Select {pick_size} cards by number (e.g., 1 3 4). Enter 'q' to end the run: "
        ).strip()
        if raw.lower() == "q":
            return None
        tokens = [token for token in raw.replace(",", " ").split(" ") if token]
        if len(tokens) != pick_size:
            print(f"Please choose exactly {pick_size} distinct cards.")
            continue
        try:
            picks = [int(token) for token in tokens]
        except ValueError:
            print("Selections must be numeric indices.")
            continue
        if any(p < 1 or p > len(hand) for p in picks):
            print("One or more selections are out of range.")
            continue
        if len(set(picks)) != pick_size:
            print("Please avoid duplicate selections.")
            continue
        return [hand[p - 1] for p in picks]


def score_trio(
    selected: Sequence[Ingredient],
    chefs: Sequence[Chef],
    pick_size: int,
    cookbook: MutableMapping[str, Tuple[str, ...]],
    recipe_counts: MutableMapping[str, int],
) -> int:
    Value = sum(ingredient.Value for ingredient in selected)
    recipe_name = DATA.which_recipe(list(selected))
    times_cooked_before = recipe_counts.get(recipe_name, 0) if recipe_name else 0
    recipe_multiplier = DATA.recipe_multiplier(
        recipe_name,
        chefs=chefs,
        times_cooked=times_cooked_before,
    )
    final_score = int(round(Value * recipe_multiplier))
    key_set = DATA.chefs_key_ingredients(chefs)
    chef_hits = sum(1 for ing in selected if ing.name in key_set)

    discovered = False
    personal_discovery = False
    chef_has_recipe = False
    total_cooked = 0
    if recipe_name:
        combo = tuple(sorted(ingredient.name for ingredient in selected))
        chef_has_recipe = any(recipe_name in chef.recipe_names for chef in chefs)
        if recipe_name not in cookbook:
            cookbook[recipe_name] = combo
            discovered = True
            personal_discovery = not chef_has_recipe
        recipe_counts[recipe_name] = times_cooked_before + 1
        total_cooked = recipe_counts[recipe_name]

    print("\n--- Trio Result ---")
    for ing in selected:
        taste_family = describe_taste_and_family(ing)
        print(f"  {ing.name} ({taste_family}, Value: {ing.Value})")
    print(f"Total Value: {Value}")
    if recipe_name:
        print(f"Recipe completed: {recipe_name}")
        print(f"Recipe multiplier: x{recipe_multiplier:.2f}")
        if discovered:
            if personal_discovery:
                base = DATA.recipe_by_name.get(recipe_name)
                base_text = (
                    f" Base multiplier recorded as {format_multiplier(base.base_multiplier)}."
                    if base
                    else ""
                )
                print(
                    "  -> Personal discovery! Recipe added to your cookbook." + base_text
                )
            else:
                print("  -> New recipe added to your cookbook!")
        if total_cooked:
            times = "time" if total_cooked == 1 else "times"
            print(f"Cooked {recipe_name} {total_cooked} {times} so far.")
    else:
        print("No recipe completed this turn.")
    print(f"Score gained: {final_score} (base Value: {Value})")
    active_names = ", ".join(chef.name for chef in chefs) if chefs else "None"
    print(
        f"Chef key ingredients used: {chef_hits}/{pick_size} (Active: {active_names})\n"
    )
    return final_score


def play_single_run(
    theme_name: str,
    chefs: List[Chef],
    rounds: int,
    cooks_per_round: int,
    hand_size: int,
    pick_size: int,
    max_chefs: int,
    rng: random.Random,
) -> Tuple[int, Dict[str, Tuple[str, ...]], Counter[str]]:
    if rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if cooks_per_round <= 0:
        raise ValueError("cooks_per_round must be a positive integer")
    if hand_size <= 0:
        raise ValueError("hand_size must be a positive integer")
    if pick_size <= 0:
        raise ValueError("pick_size must be a positive integer")
    if pick_size > hand_size:
        raise ValueError("pick_size cannot exceed hand_size")

    if max_chefs <= 0:
        raise ValueError("max_chefs must be a positive integer")
    if len(chefs) > max_chefs:
        raise ValueError("Initial chef roster exceeds the configured maximum.")

    total_score = 0
    cookbook: Dict[str, Tuple[str, ...]] = {}
    recipe_counts: Counter[str] = Counter()
    active_names = ", ".join(chef.name for chef in chefs) or "None"
    print("\nActive chefs this run: " + active_names)
    print(f"Maximum chefs allowed: {max_chefs}")

    total_turns = rounds * cooks_per_round
    turn_number = 0

    for round_index in range(1, rounds + 1):
        deck = build_market_deck(
            DATA,
            theme_name,
            chefs,
            deck_size=DEFAULT_DECK_SIZE,
            bias=DEFAULT_BIAS,
            rng=rng,
        )
        rng.shuffle(deck)
        hand: List[Ingredient] = []
        print(f"\n=== Round {round_index}/{rounds} ===")

        for cook_index in range(1, cooks_per_round + 1):
            turn_number += 1
            needed = hand_size - len(hand)
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

            print(
                f"\nTurn {turn_number}/{total_turns} "
                f"(Cook {cook_index}/{cooks_per_round})"
            )
            display_hand(hand, chefs, cookbook)
            selected = prompt_trio(hand, pick_size)
            if selected is None:
                print("Run ended early by player choice.\n")
                return total_score, cookbook, recipe_counts

            gained = score_trio(selected, chefs, pick_size, cookbook, recipe_counts)
            total_score += gained
            print(f"Cumulative score: {total_score}\n")
            for ingredient in selected:
                hand.remove(ingredient)

        if (
            round_index < rounds
            and len(chefs) < max_chefs
            and len(chefs) < len(DATA.chefs)
        ):
            add_choice = input(
                "Recruit a new chef before the next round? (y/N): "
            ).strip().lower()
            if add_choice in {"y", "yes"}:
                new_chef = choose_additional_chef([c.name for c in chefs], rng)
                if new_chef and all(c.name != new_chef.name for c in chefs):
                    chefs.append(new_chef)
                    print(
                        f"Chef {new_chef.name} joins your lineup! "
                        "Future draws will favor their signature ingredients. "
                        f"Roster {len(chefs)}/{max_chefs}."
                    )

    return total_score, cookbook, recipe_counts


def main() -> None:
    print(
        f"""
===============================================
 Food Deck Mini Game (terminal prototype)
 Version: {RULES_VERSION}
-----------------------------------------------
 * Data shared with food_simulator.py (JSON-driven).
 * Pick a chef and theme, configure rounds, cooks per round, hand size, and
   how many ingredients to keep each cook.
 * Scores are computed using the same trio scoring model.
 * Enter 'q' during a cook to stop the current run early.
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
        roster: List[Chef] = []
        start_with_chef = input(
            "Begin with a chef? (Y/n, blank for yes): "
        ).strip().lower()
        if start_with_chef not in {"n", "no"}:
            chef = choose_chef(rng)
            roster.append(chef)
        rounds = prompt_round_count()
        cooks_per_round = prompt_cooks_per_round()
        hand_size = prompt_hand_size()
        pick_size = prompt_pick_size(hand_size)
        max_chefs = prompt_max_chefs()
        runs = prompt_run_count()
        total_turns = rounds * cooks_per_round
        overall_cookbook: Dict[str, Tuple[str, ...]] = {}

        if len(roster) > max_chefs:
            print(
                f"Reducing initial roster to the first {max_chefs} chefs to respect the limit."
            )
            del roster[max_chefs:]

        for run_index in range(1, runs + 1):
            chef_names = ", ".join(c.name for c in roster) or "None"
            print(
                f"\n>>> Starting run {run_index}/{runs} with Chefs {chef_names} in {theme} "
                f"({rounds} rounds × {cooks_per_round} cooks, hand {hand_size}, pick {pick_size}) "
                f"[{total_turns} total turns, max chefs {max_chefs}] <<<"
            )
            score, run_cookbook, run_counts = play_single_run(
                theme,
                roster,
                rounds,
                cooks_per_round,
                hand_size,
                pick_size,
                max_chefs,
                rng,
            )
            print(f"Final score for run {run_index}: {score}\n")

            if run_cookbook:
                print_cookbook(
                    run_cookbook,
                    run_counts,
                    title=f"Run {run_index} Cookbook",
                    chefs=roster,
                )
            else:
                print("No recipes discovered this run.")

            for name, combo in run_cookbook.items():
                overall_cookbook.setdefault(name, combo)

            if len(roster) < max_chefs and len(roster) < len(DATA.chefs):
                add_choice = input(
                    "Add a new chef to your lineup for the next run? (y/N): "
                ).strip().lower()
                if add_choice in {"y", "yes"}:
                    new_chef = choose_additional_chef([c.name for c in roster], rng)
                    if new_chef and new_chef not in roster:
                        roster.append(new_chef)
                        print(
                            f"Chef {new_chef.name} has joined your team for future runs. "
                            f"Roster {len(roster)}/{max_chefs}."
                        )

        if overall_cookbook:
            print_cookbook(overall_cookbook, title="Combined Cookbook (all runs)")
        else:
            print("No recipes were discovered across these runs.")

        again = input("Play another configuration? (y/N): ").strip().lower()
        if again not in {"y", "yes"}:
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
