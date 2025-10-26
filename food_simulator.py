from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from datetime import datetime
from typing import Dict, Iterator, List, Tuple

from food_api import (
    RULES_VERSION,
    GameData,
    SimulationConfig,
    ensure_dir,
    format_report_header,
    simulate_many,
)
from seed_utils import resolve_seed


PRIMARY_SUMMARY_KEYS: Tuple[str, ...] = (
    "nbplay",
    "runs_per_play",
    "runs",
    "theme",
    "seed",
    "rounds",
    "cooks_per_round",
    "turns_per_run",
)

MULTIPLIER_SUMMARY_KEYS = {"avg_recipe_multiplier"}


def iter_summary_items(summary: Dict[str, object]) -> Iterator[Tuple[str, object]]:
    """Yield summary entries in a stable order for console and reports."""

    seen: set[str] = set()
    for key in PRIMARY_SUMMARY_KEYS:
        if key in summary and key not in MULTIPLIER_SUMMARY_KEYS:
            seen.add(key)
            yield key, summary[key]

    for key, value in summary.items():
        if key in MULTIPLIER_SUMMARY_KEYS or key in seen:
            continue
        seen.add(key)
        yield key, value


def write_report_files(
    out_dir: str,
    summary: Dict[str, object],
    ingredient_totals: Counter,
    taste_totals: Counter,
    recipe_totals: Counter,
    scores: List[float],
) -> Dict[str, str]:
    ensure_dir(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = summary.get("seed")
    theme = summary.get("theme")
    plays = summary.get("nbplay")
    runs = summary.get("runs")
    base_name = f"report_{theme}_plays{plays}_runs{runs}_seed{seed}_{timestamp}"

    txt_path = os.path.join(out_dir, base_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(format_report_header())
        handle.write("\n")
        multiplier_keys = {"avg_recipe_multiplier"}
        for key, value in iter_summary_items(summary):
            if key in multiplier_keys:
                continue
            handle.write(f"{key}: {format_summary_value(value)}\n")

        multiplier_sections = [
            ("avg_recipe_multiplier", "Recipe multiplier"),
        ]
        wrote_multiplier_header = False
        for key, label in multiplier_sections:
            if key not in summary:
                continue
            if not wrote_multiplier_header:
                handle.write("\nAverage Multipliers:\n")
                wrote_multiplier_header = True
            handle.write(f"{label}: {format_summary_value(summary[key])}\n")

        handle.write("\nTop Ingredients (usage):\n")
        for name, count in ingredient_totals.most_common(20):
            handle.write(f"{name},{count}\n")
        handle.write("\nTaste Mix:\n")
        for taste, count in taste_totals.items():
            handle.write(f"{taste},{count}\n")
        handle.write("\nMost Cooked Recipes:\n")
        for recipe, count in recipe_totals.most_common(20):
            handle.write(f"{recipe},{count}\n")

    csv_ing = os.path.join(out_dir, base_name + "_ingredients.csv")
    with open(csv_ing, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Ingredient", "UseCount"])
        for name, count in ingredient_totals.most_common():
            writer.writerow([name, count])

    csv_taste = os.path.join(out_dir, base_name + "_tastes.csv")
    with open(csv_taste, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Taste", "Count"])
        for taste, count in taste_totals.items():
            writer.writerow([taste, count])

    csv_recipes = os.path.join(out_dir, base_name + "_recipes.csv")
    with open(csv_recipes, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Recipe", "TimesCooked"])
        for recipe, count in recipe_totals.most_common():
            writer.writerow([recipe, count])

    csv_scores = os.path.join(out_dir, base_name + "_scores.csv")
    with open(csv_scores, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["RunIndex", "Score"])
        for index, score in enumerate(scores):
            writer.writerow([index, score])

    return {
        "summary_txt": txt_path,
        "ingredients_csv": csv_ing,
        "tastes_csv": csv_taste,
        "recipes_csv": csv_recipes,
        "scores_csv": csv_scores,
    }


def format_summary_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Deck Simulator")
    default_config = SimulationConfig()
    parser.add_argument(
        "--nbplay",
        type=int,
        default=1,
        help="Number of full game plays to simulate (default=1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of Monte Carlo runs executed per play (default=200)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Default",
        help="Theme / Market name from themes.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports",
        help="Output folder for report files (default=reports)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility (default=None=randomized)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=default_config.rounds,
        help=f"Number of rounds per run (default={default_config.rounds})",
    )
    parser.add_argument(
        "--cooks-per-round",
        type=int,
        default=default_config.cooks,
        help=f"Number of cooking turns per round (default={default_config.cooks})",
    )
    parser.add_argument(
        "--active-chefs",
        type=int,
        default=default_config.active_chefs,
        help=f"Number of active chefs per run (default={default_config.active_chefs})",
    )
    parser.add_argument(
        "--hand-size",
        type=int,
        default=default_config.hand_size,
        help=f"Number of ingredients in hand (default={default_config.hand_size})",
    )
    parser.add_argument(
        "--pick-size",
        type=int,
        default=default_config.pick_size,
        help=f"Number of ingredients picked each cook (default={default_config.pick_size})",
    )
    args = parser.parse_args()

    data = GameData.from_json()

    if args.theme not in data.themes:
        available = ", ".join(sorted(data.themes.keys()))
        print(f"Theme '{args.theme}' not found. Available: {available}")
        raise SystemExit(1)

    if args.nbplay <= 0:
        raise SystemExit("--nbplay must be a positive integer")
    if args.runs <= 0:
        raise SystemExit("--runs must be a positive integer")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be a positive integer")
    if args.cooks_per_round <= 0:
        raise SystemExit("--cooks-per-round must be a positive integer")
    if args.active_chefs < 0:
        raise SystemExit("--active-chefs cannot be negative")
    if args.hand_size <= 0:
        raise SystemExit("--hand-size must be a positive integer")
    if args.pick_size <= 0:
        raise SystemExit("--pick-size must be a positive integer")
    if args.pick_size > args.hand_size:
        raise SystemExit("--pick-size cannot exceed --hand-size")

    seed_used, _ = resolve_seed(args.seed)
    print(f"Using RNG seed: {seed_used}")

    sim_config = SimulationConfig(
        rounds=args.rounds,
        cooks=args.cooks_per_round,
        active_chefs=args.active_chefs,
        hand_size=args.hand_size,
        pick_size=args.pick_size,
    )

    plays = args.nbplay
    runs_per_play = args.runs
    total_simulated_runs = plays * runs_per_play
    turns_per_run = sim_config.rounds * sim_config.cooks
    total_rounds = total_simulated_runs * sim_config.rounds
    total_cooks = total_simulated_runs * turns_per_run

    summary, ingredient_totals, taste_totals, recipe_totals, scores = simulate_many(
        data,
        n=total_simulated_runs,
        theme_name=args.theme,
        seed=seed_used,
        config=sim_config,
    )

    summary = dict(summary)
    summary["nbplay"] = plays
    summary["runs_per_play"] = runs_per_play
    summary.setdefault("rounds", sim_config.rounds)
    summary.setdefault("cooks_per_round", sim_config.cooks)
    summary.setdefault("active_chefs", sim_config.active_chefs)
    summary.setdefault("hand_size", sim_config.hand_size)
    summary.setdefault("pick_size", sim_config.pick_size)
    summary["turns_per_run"] = turns_per_run
    summary["total_rounds"] = total_rounds
    summary["total_cooks"] = total_cooks
    summary["rules_version"] = RULES_VERSION

    print(f"=== SUMMARY (Rules v{RULES_VERSION}) ===")
    for key, value in iter_summary_items(summary):
        print(f"{key}: {format_summary_value(value)}")

    paths = write_report_files(args.out, summary, ingredient_totals, taste_totals, recipe_totals, scores)
    print("\nFiles written:")
    for label, path in paths.items():
        print(f" - {label}: {path}")
