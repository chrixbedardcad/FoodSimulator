from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is optional
    import statistics as stats

    HAVE_NUMPY = False

RULES_VERSION = "1.0.0"

DEFAULT_INGREDIENTS_JSON = "ingredients.json"
DEFAULT_TASTE_JSON = "taste_matrix.json"
DEFAULT_RECIPES_JSON = "recipes.json"
DEFAULT_CHEFS_JSON = "chefs.json"
DEFAULT_THEMES_JSON = "themes.json"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class Ingredient:
    name: str
    taste: str
    chips: int


@dataclass(frozen=True)
class Recipe:
    name: str
    trio: Tuple[str, str, str]


@dataclass
class Chef:
    name: str
    recipe_names: List[str]
    perks: MutableMapping[str, object]


@dataclass
class GameData:
    ingredients: Dict[str, Ingredient]
    recipes: List[Recipe]
    chefs: List[Chef]
    themes: Dict[str, List[Tuple[str, int]]]
    taste_matrix: MutableMapping[str, MutableMapping[str, int]]
    recipe_by_name: Dict[str, Recipe] = field(init=False)
    recipe_trio_lookup: Dict[Tuple[str, str, str], str] = field(init=False)

    def __post_init__(self) -> None:
        self.recipe_by_name = {recipe.name: recipe for recipe in self.recipes}
        self.recipe_trio_lookup = {
            tuple(sorted(recipe.trio)): recipe.name for recipe in self.recipes
        }

    @classmethod
    def from_json(
        cls,
        ingredients_path: str = DEFAULT_INGREDIENTS_JSON,
        recipes_path: str = DEFAULT_RECIPES_JSON,
        chefs_path: str = DEFAULT_CHEFS_JSON,
        taste_path: str = DEFAULT_TASTE_JSON,
        themes_path: str = DEFAULT_THEMES_JSON,
    ) -> "GameData":
        return cls(
            ingredients=_load_ingredients(ingredients_path),
            recipes=_load_recipes(recipes_path),
            chefs=_load_chefs(chefs_path),
            themes=_load_themes(themes_path),
            taste_matrix=_load_taste_matrix(taste_path),
        )

    # --- Helpers that operate on the loaded data ---
    def chef_key_ingredients(self, chef: Chef) -> set[str]:
        keys: set[str] = set()
        for recipe_name in chef.recipe_names:
            recipe = self.recipe_by_name.get(recipe_name)
            if recipe:
                keys.update(recipe.trio)
        return keys

    def trio_score(self, ingredients: Sequence[Ingredient]) -> Tuple[int, int, int, int]:
        a, b, c = ingredients
        chips = a.chips + b.chips + c.chips
        taste_sum = (
            self.taste_matrix[a.taste][b.taste]
            + self.taste_matrix[a.taste][c.taste]
            + self.taste_matrix[b.taste][c.taste]
        )
        multiplier = max(1, taste_sum)
        return chips * multiplier, chips, taste_sum, multiplier

    def which_recipe(self, ingredients: Sequence[Ingredient]) -> Optional[str]:
        key = tuple(sorted(ingredient.name for ingredient in ingredients))
        return self.recipe_trio_lookup.get(key)


def _load_ingredients(path: str) -> Dict[str, Ingredient]:
    raw = load_json(path)
    return {
        entry["name"]: Ingredient(entry["name"], entry["taste"], int(entry["chips"]))
        for entry in raw
    }


def _load_recipes(path: str) -> List[Recipe]:
    raw = load_json(path)
    return [Recipe(entry["name"], tuple(entry["trio"])) for entry in raw]


def _load_chefs(path: str) -> List[Chef]:
    raw = load_json(path)
    return [
        Chef(
            entry["name"],
            list(entry.get("recipe_names", [])),
            dict(entry.get("perks", {})),
        )
        for entry in raw
    ]


def _load_taste_matrix(path: str):
    raw = load_json(path)
    return raw["matrix"]


def _load_themes(path: str):
    raw = load_json(path)
    fixed: Dict[str, List[Tuple[str, int]]] = {}
    for theme_name, items in raw.items():
        fixed[theme_name] = [
            (item["ingredient"], int(item["copies"])) for item in items
        ]
    return fixed


DEFAULT_HAND_SIZE = 5
TRIO_SIZE = 3


def build_market_deck(
    data: GameData,
    theme_name: str,
    chef: Chef,
    deck_size: int = 100,
    bias: float = 2.7,
    rng: Optional[random.Random] = None,
) -> List[Ingredient]:
    rng = rng or random
    theme_pool = data.themes[theme_name]
    keyset = data.chef_key_ingredients(chef)
    weighted: List[Ingredient] = []
    for ingredient_name, copies in theme_pool:
        ingredient = data.ingredients.get(ingredient_name)
        if not ingredient:
            continue
        weight = bias if ingredient_name in keyset else 1.0
        total = max(0, int(round(copies * weight)))
        weighted.extend([ingredient] * total)
    rng.shuffle(weighted)
    if len(weighted) <= deck_size:
        return list(weighted)
    return list(weighted[:deck_size])


def _refill_hand(
    hand: List[Ingredient],
    deck: List[Ingredient],
    hand_size: int = DEFAULT_HAND_SIZE,
) -> Tuple[List[Ingredient], List[Ingredient]]:
    while len(hand) < hand_size and deck:
        hand.append(deck.pop())
    return hand, deck


def _select_trio_from_hand(
    hand: List[Ingredient],
    key_ingredients: Iterable[str],
    guarantee_prob: float,
    rng: random.Random,
) -> List[Ingredient]:
    key_set = set(key_ingredients)
    if len(hand) < TRIO_SIZE:
        return []

    trio: List[Ingredient] = []
    if rng.random() < guarantee_prob:
        key_cards = [card for card in hand if card.name in key_set]
        if key_cards:
            pick = rng.choice(key_cards)
            trio.append(pick)
            hand.remove(pick)

    while len(trio) < TRIO_SIZE and hand:
        pick = rng.choice(hand)
        trio.append(pick)
        hand.remove(pick)
    return trio


def draw_cook(
    data: GameData,
    hand: List[Ingredient],
    deck: List[Ingredient],
    chef: Chef,
    guarantee_prob: float = 0.6,
    hand_size: int = DEFAULT_HAND_SIZE,
    key_ingredients: Optional[Iterable[str]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Ingredient], List[Ingredient], List[Ingredient]]:
    rng = rng or random
    hand, deck = _refill_hand(hand, deck, hand_size)
    if len(hand) < TRIO_SIZE:
        return [], hand, deck

    keyset = set(key_ingredients) if key_ingredients is not None else data.chef_key_ingredients(chef)
    trio = _select_trio_from_hand(hand, keyset, guarantee_prob, rng)
    hand, deck = _refill_hand(hand, deck, hand_size)
    return trio, hand, deck


@dataclass
class LearningState:
    recipe_name: Optional[str] = None
    hits: int = 0


def update_learning(state: LearningState, chef: Chef, recipe_name: Optional[str]) -> LearningState:
    if recipe_name and recipe_name in chef.recipe_names:
        if state.recipe_name == recipe_name:
            state.hits += 1
        elif state.recipe_name is None:
            state.recipe_name, state.hits = recipe_name, 1
        elif state.hits < 2:
            state.recipe_name, state.hits = recipe_name, 1
    return state


@dataclass
class SimulationConfig:
    deck_size: int = 100
    cooks: int = 6
    rounds: int = 3
    bias: float = 2.7
    guarantee_prob: float = 0.6
    reshuffle_every: int = 8
    hand_size: int = DEFAULT_HAND_SIZE


def simulate_run(
    data: GameData,
    theme_name: str = "Mediterranean",
    start_chef: Optional[Chef] = None,
    config: Optional[SimulationConfig] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, object]:
    rng = rng or random
    cfg = config or SimulationConfig()
    chef = start_chef or rng.choice(data.chefs)
    learning = LearningState()
    mastered: set[str] = set()
    total_score = 0

    chefkey_per_draw: List[int] = []
    taste_counts: Counter[str] = Counter()
    ingredient_use: Counter[str] = Counter()
    recipe_counts: Counter[str] = Counter()

    deck = build_market_deck(data, theme_name, chef, cfg.deck_size, cfg.bias, rng)
    hand: List[Ingredient] = []
    hand, deck = _refill_hand(hand, deck, cfg.hand_size)
    current_keys = data.chef_key_ingredients(chef)
    draws = 0

    for _ in range(cfg.rounds):
        for _ in range(cfg.cooks):
            if len(hand) < TRIO_SIZE:
                if len(deck) < TRIO_SIZE:
                    deck = build_market_deck(data, theme_name, chef, cfg.deck_size, cfg.bias, rng)
                    draws = 0
                hand, deck = _refill_hand(hand, deck, cfg.hand_size)
                if len(hand) < TRIO_SIZE:
                    break

            if len(deck) < TRIO_SIZE or draws >= cfg.reshuffle_every:
                deck = build_market_deck(data, theme_name, chef, cfg.deck_size, cfg.bias, rng)
                draws = 0
                hand, deck = _refill_hand(hand, deck, cfg.hand_size)

            trio, hand, deck = draw_cook(
                data,
                hand,
                deck,
                chef,
                cfg.guarantee_prob,
                cfg.hand_size,
                current_keys,
                rng,
            )
            if len(trio) < TRIO_SIZE:
                break

            chefkey_per_draw.append(sum(1 for ingredient in trio if ingredient.name in current_keys))
            for ingredient in trio:
                taste_counts[ingredient.taste] += 1
                ingredient_use[ingredient.name] += 1

            score, _, _, _ = data.trio_score(trio)
            total_score += score

            recipe_name = data.which_recipe(trio)
            if recipe_name:
                recipe_counts[recipe_name] += 1
            learning = update_learning(learning, chef, recipe_name)

            if learning.recipe_name and learning.hits >= 2:
                mastered.add(learning.recipe_name)
                learning = LearningState()

            draws += 1

        if rng.random() < 0.5:
            chef = rng.choice(data.chefs)
            current_keys = data.chef_key_ingredients(chef)
            deck = build_market_deck(data, theme_name, chef, cfg.deck_size, cfg.bias, rng)
            hand = []
            hand, deck = _refill_hand(hand, deck, cfg.hand_size)
            draws = 0

    return {
        "score": total_score,
        "mastered": mastered,
        "ingredient_use": ingredient_use,
        "taste_counts": taste_counts,
        "chefkey_per_draw": chefkey_per_draw,
        "recipe_counts": recipe_counts,
    }


def summarize_scores(scores: Sequence[float]) -> Tuple[float, float, float, float, float]:
    if HAVE_NUMPY:
        arr = list(scores)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
    else:
        mean_val = stats.mean(scores)
        std_val = stats.pstdev(scores)
        ordered = sorted(scores)

        def percentile(pct: float) -> float:
            k = int(round((pct / 100.0) * (len(ordered) - 1)))
            k = max(0, min(k, len(ordered) - 1))
            return float(ordered[k])

        p50, p90, p99 = percentile(50), percentile(90), percentile(99)
    return mean_val, std_val, p50, p90, p99


def simulate_many(
    data: GameData,
    n: int = 200,
    theme_name: str = "Mediterranean",
    seed: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
) -> Tuple[Dict[str, object], Counter[str], Counter[str], Counter[str], List[float]]:
    rng = random.Random(seed) if seed is not None else random.Random()
    cfg = config or SimulationConfig()

    scores: List[float] = []
    mastered_any = 0
    ingredient_totals: Counter[str] = Counter()
    taste_totals: Counter[str] = Counter()
    chefkey_all: List[int] = []
    recipe_totals: Counter[str] = Counter()

    for _ in range(n):
        result = simulate_run(data, theme_name=theme_name, config=cfg, rng=rng)
        score = float(result["score"])
        scores.append(score)
        mastered = result["mastered"]
        if mastered:
            mastered_any += 1
        ingredient_totals.update(result["ingredient_use"])  # type: ignore[arg-type]
        taste_totals.update(result["taste_counts"])  # type: ignore[arg-type]
        chefkey_all.extend(result["chefkey_per_draw"])  # type: ignore[arg-type]
        recipe_totals.update(result["recipe_counts"])  # type: ignore[arg-type]

    mean_val, std_val, p50, p90, p99 = summarize_scores(scores)
    total_ing = sum(ingredient_totals.values())
    hhi = sum((count / total_ing) ** 2 for count in ingredient_totals.values()) if total_ing else 0.0
    avg_chef_keys = (sum(chefkey_all) / len(chefkey_all)) if chefkey_all else 0.0

    summary = {
        "runs": n,
        "theme": theme_name,
        "seed": seed,
        "average_score": round(mean_val, 2),
        "std_score": round(std_val, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "p99": round(p99, 2),
        "mastery_rate_pct": round(100.0 * mastered_any / n, 1) if n else 0.0,
        "avg_chef_key_per_draw": round(avg_chef_keys, 2),
        "ingredient_hhi": round(hhi, 4),
    }
    return summary, ingredient_totals, taste_totals, recipe_totals, scores


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def format_report_header() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"=== Food Simulator Report (Rules v{RULES_VERSION}) ===\nGenerated: {timestamp}\n"

