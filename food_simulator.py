from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from datetime import datetime
from typing import Dict, List

from food_api import (
    RULES_VERSION,
    GameData,
    ensure_dir,
    format_report_header,
    simulate_many,
)


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
    runs = summary.get("runs")
    base_name = f"report_{theme}_runs{runs}_seed{seed}_{timestamp}"

    txt_path = os.path.join(out_dir, base_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(format_report_header())
        handle.write("\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Deck Simulator")
    parser.add_argument(
        "--runs", type=int, default=200, help="Number of Monte Carlo runs (default=200)"
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Mediterranean",
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
    args = parser.parse_args()

    data = GameData.from_json()

    if args.theme not in data.themes:
        available = ", ".join(sorted(data.themes.keys()))
        print(f"Theme '{args.theme}' not found. Available: {available}")
        raise SystemExit(1)

    seed_used = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    summary, ingredient_totals, taste_totals, recipe_totals, scores = simulate_many(
        data,
        n=args.runs,
        theme_name=args.theme,
        seed=seed_used,
    )

    summary = dict(summary)
    summary["rules_version"] = RULES_VERSION

    print(f"=== SUMMARY (Rules v{RULES_VERSION}) ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    paths = write_report_files(args.out, summary, ingredient_totals, taste_totals, recipe_totals, scores)
    print("\nFiles written:")
    for label, path in paths.items():
        print(f" - {label}: {path}")
