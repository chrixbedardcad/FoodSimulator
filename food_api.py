from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is optional
    import statistics as stats

    HAVE_NUMPY = False

RULES_VERSION = "1.0.1"

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
    base_multiplier: float = 1.0
    delta_multiplier: float = 0.0


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
    recipe_multipliers: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.recipe_by_name = {recipe.name: recipe for recipe in self.recipes}
        self.recipe_trio_lookup = {
            tuple(sorted(recipe.trio)): recipe.name for recipe in self.recipes
        }
        self.recipe_multipliers = self._build_recipe_multipliers()

    def _build_recipe_multipliers(self) -> Dict[str, float]:
        return {recipe.name: float(recipe.base_multiplier) for recipe in self.recipes}

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

    def chefs_key_ingredients(self, chefs: Iterable[Chef]) -> set[str]:
        combined: set[str] = set()
        for chef in chefs:
            combined.update(self.chef_key_ingredients(chef))
        return combined

    def trio_score(self, ingredients: Sequence[Ingredient]) -> Tuple[int, int, int, int]:
        if not ingredients:
            return 0, 0, 0, 1

        chips = sum(ingredient.chips for ingredient in ingredients)
        # Taste synergy no longer affects scoring but is retained for reference.
        taste_sum = 0
        multiplier = 1
        return chips, chips, taste_sum, multiplier

    def which_recipe(self, ingredients: Sequence[Ingredient]) -> Optional[str]:
        key = tuple(sorted(ingredient.name for ingredient in ingredients))
        return self.recipe_trio_lookup.get(key)

    def recipe_multiplier(
        self,
        recipe_name: Optional[str],
        *,
        chefs: Optional[Sequence[Chef]] = None,
        times_cooked: int = 0,
    ) -> float:
        if not recipe_name:
            return 1.0
        recipe = self.recipe_by_name.get(recipe_name)
        if not recipe:
            return 1.0

        cooked = max(int(times_cooked), 0)
        base_multiplier = recipe.base_multiplier + (recipe.delta_multiplier * cooked)
        if base_multiplier < 0:
            base_multiplier = 0.0

        chef_multiplier = 1.0
        if chefs:
            for chef in chefs:
                perks = chef.perks.get("recipe_multipliers")
                if not isinstance(perks, Mapping):
                    continue
                value = perks.get(recipe_name)
                if value is None:
                    continue
                try:
                    chef_multiplier *= float(value)
                except (TypeError, ValueError):
                    continue

        total = base_multiplier * chef_multiplier
        return total if total > 0 else 0.0


def _load_ingredients(path: str) -> Dict[str, Ingredient]:
    raw = load_json(path)
    return {
        entry["name"]: Ingredient(entry["name"], entry["taste"], int(entry["chips"]))
        for entry in raw
    }


def _load_recipes(path: str) -> List[Recipe]:
    raw = load_json(path)
    recipes: List[Recipe] = []
    for entry in raw:
        base = entry.get("base_multiplier", 1.0)
        delta = entry.get("delta_multiplier", 0.0)
        try:
            base_val = float(base)
        except (TypeError, ValueError):
            base_val = 1.0
        try:
            delta_val = float(delta)
        except (TypeError, ValueError):
            delta_val = 0.0
        recipes.append(
            Recipe(
                entry["name"],
                tuple(entry["trio"]),
                base_multiplier=base_val,
                delta_multiplier=delta_val,
            )
        )
    return recipes


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
DEFAULT_PICK_SIZE = 3
TRIO_SIZE = 3


def build_market_deck(
    data: GameData,
    theme_name: str,
    chefs: Sequence[Chef],
    deck_size: int = 100,
    bias: float = 2.7,
    rng: Optional[random.Random] = None,
) -> List[Ingredient]:
    rng = rng or random
    theme_pool = data.themes[theme_name]
    keyset = data.chefs_key_ingredients(chefs)
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
    pick_size: int,
) -> List[Ingredient]:
    key_set = set(key_ingredients)
    if pick_size <= 0:
        return []
    if len(hand) < pick_size:
        return []

    trio: List[Ingredient] = []
    if rng.random() < guarantee_prob:
        key_cards = [card for card in hand if card.name in key_set]
        if key_cards:
            pick = rng.choice(key_cards)
            trio.append(pick)
            hand.remove(pick)

    while len(trio) < pick_size and hand:
        pick = rng.choice(hand)
        trio.append(pick)
        hand.remove(pick)
    return trio


def draw_cook(
    data: GameData,
    hand: List[Ingredient],
    deck: List[Ingredient],
    chefs: Sequence[Chef],
    guarantee_prob: float = 0.6,
    hand_size: int = DEFAULT_HAND_SIZE,
    pick_size: int = DEFAULT_PICK_SIZE,
    key_ingredients: Optional[Iterable[str]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Ingredient], List[Ingredient], List[Ingredient]]:
    rng = rng or random
    hand, deck = _refill_hand(hand, deck, hand_size)
    if len(hand) < pick_size or pick_size <= 0:
        return [], hand, deck

    keyset = (
        set(key_ingredients)
        if key_ingredients is not None
        else data.chefs_key_ingredients(chefs)
    )
    trio = _select_trio_from_hand(hand, keyset, guarantee_prob, rng, pick_size)
    hand, deck = _refill_hand(hand, deck, hand_size)
    return trio, hand, deck


@dataclass
class LearningState:
    hits: Counter[str] = field(default_factory=Counter)

def update_learning(
    state: LearningState, chefs: Sequence[Chef], recipe_name: Optional[str]
) -> LearningState:
    if recipe_name:
        state.hits[recipe_name] += 1
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
    active_chefs: int = 3
    pick_size: int = DEFAULT_PICK_SIZE


def simulate_run(
    data: GameData,
    theme_name: str = "Mediterranean",
    start_chefs: Optional[Sequence[Chef]] = None,
    config: Optional[SimulationConfig] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, object]:
    rng = rng or random
    cfg = config or SimulationConfig()
    if cfg.hand_size <= 0:
        raise ValueError("SimulationConfig.hand_size must be a positive integer")
    if cfg.pick_size <= 0:
        raise ValueError("SimulationConfig.pick_size must be a positive integer")
    if cfg.pick_size > cfg.hand_size:
        raise ValueError("SimulationConfig.pick_size cannot exceed hand_size")
    if cfg.active_chefs < 0:
        raise ValueError("SimulationConfig.active_chefs cannot be negative")

    if start_chefs is not None:
        active_chefs = list(start_chefs)
    else:
        available_chefs = list(data.chefs)
        if not available_chefs:
            raise ValueError("No chefs available to start the simulation")
        rng.shuffle(available_chefs)
        desired_chefs = min(cfg.active_chefs, len(available_chefs))
        active_chefs = available_chefs[:desired_chefs]
    learning = LearningState()
    mastered: set[str] = set()
    total_score = 0
    recipe_multiplier_total = 0.0
    recipe_multiplier_events = 0

    chefkey_per_draw: List[int] = []
    taste_counts: Counter[str] = Counter()
    ingredient_use: Counter[str] = Counter()
    recipe_counts: Counter[str] = Counter()
    cookbook: Dict[str, Tuple[str, ...]] = {}

    deck = build_market_deck(data, theme_name, active_chefs, cfg.deck_size, cfg.bias, rng)
    hand: List[Ingredient] = []
    hand, deck = _refill_hand(hand, deck, cfg.hand_size)
    current_keys = data.chefs_key_ingredients(active_chefs)
    draws = 0
    round_scores: List[int] = []
    cumulative_scores: List[int] = []
    cumulative_total = 0

    pick_size = cfg.pick_size

    for _ in range(cfg.rounds):
        round_total = 0
        for _ in range(cfg.cooks):
            if len(hand) < pick_size:
                if len(deck) < pick_size:
                    deck = build_market_deck(
                        data, theme_name, active_chefs, cfg.deck_size, cfg.bias, rng
                    )
                    draws = 0
                hand, deck = _refill_hand(hand, deck, cfg.hand_size)
                if len(hand) < pick_size:
                    break

            if len(deck) < pick_size or draws >= cfg.reshuffle_every:
                deck = build_market_deck(
                    data, theme_name, active_chefs, cfg.deck_size, cfg.bias, rng
                )
                draws = 0
                hand, deck = _refill_hand(hand, deck, cfg.hand_size)

            trio, hand, deck = draw_cook(
                data,
                hand,
                deck,
                active_chefs,
                cfg.guarantee_prob,
                cfg.hand_size,
                pick_size,
                current_keys,
                rng,
            )
            if len(trio) < pick_size:
                break

            chefkey_per_draw.append(sum(1 for ingredient in trio if ingredient.name in current_keys))
            for ingredient in trio:
                taste_counts[ingredient.taste] += 1
                ingredient_use[ingredient.name] += 1

            recipe_name = data.which_recipe(trio)
            times_cooked_before = recipe_counts.get(recipe_name, 0) if recipe_name else 0
            if recipe_name:
                recipe_counts[recipe_name] = times_cooked_before + 1
                cookbook.setdefault(
                    recipe_name,
                    tuple(sorted(ingredient.name for ingredient in trio)),
                )
            learning = update_learning(learning, active_chefs, recipe_name)

            score, _, _, _ = data.trio_score(trio)
            recipe_multiplier = data.recipe_multiplier(
                recipe_name,
                chefs=active_chefs,
                times_cooked=times_cooked_before,
            )
            final_score = int(round(score * recipe_multiplier))
            total_score += final_score
            round_total += final_score

            if recipe_name:
                recipe_multiplier_total += recipe_multiplier
                recipe_multiplier_events += 1

            for mastered_recipe, hits in list(learning.hits.items()):
                if hits >= 2:
                    mastered.add(mastered_recipe)
                    del learning.hits[mastered_recipe]

            draws += 1

        round_scores.append(round_total)
        cumulative_total += round_total
        cumulative_scores.append(cumulative_total)

        deck = build_market_deck(data, theme_name, active_chefs, cfg.deck_size, cfg.bias, rng)
        hand = []
        hand, deck = _refill_hand(hand, deck, cfg.hand_size)
        current_keys = data.chefs_key_ingredients(active_chefs)
        draws = 0

    return {
        "score": total_score,
        "mastered": mastered,
        "ingredient_use": ingredient_use,
        "taste_counts": taste_counts,
        "chefkey_per_draw": chefkey_per_draw,
        "recipe_counts": recipe_counts,
        "recipe_multiplier_total": recipe_multiplier_total,
        "recipe_multiplier_events": recipe_multiplier_events,
        "round_scores": round_scores,
        "cumulative_scores": cumulative_scores,
        "cookbook": cookbook,
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
    recipe_multiplier_total = 0.0
    recipe_multiplier_events = 0
    round_score_totals: List[float] = [0.0] * cfg.rounds
    cumulative_score_totals: List[float] = [0.0] * cfg.rounds

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
        recipe_multiplier_total += float(result.get("recipe_multiplier_total", 0.0))
        recipe_multiplier_events += int(result.get("recipe_multiplier_events", 0))
        round_scores = result.get("round_scores")
        if isinstance(round_scores, (list, tuple)):
            for idx, value in enumerate(round_scores):
                if idx < len(round_score_totals):
                    round_score_totals[idx] += float(value)
        cumulative_scores = result.get("cumulative_scores")
        if isinstance(cumulative_scores, (list, tuple)):
            for idx, value in enumerate(cumulative_scores):
                if idx < len(cumulative_score_totals):
                    cumulative_score_totals[idx] += float(value)

    mean_val, std_val, p50, p90, p99 = summarize_scores(scores)
    total_ing = sum(ingredient_totals.values())
    hhi = sum((count / total_ing) ** 2 for count in ingredient_totals.values()) if total_ing else 0.0
    avg_chef_keys = (sum(chefkey_all) / len(chefkey_all)) if chefkey_all else 0.0
    avg_recipe_multiplier = (
        recipe_multiplier_total / recipe_multiplier_events
        if recipe_multiplier_events
        else 0.0
    )
    average_round_scores = (
        [round(total / n, 2) for total in round_score_totals] if n else []
    )
    average_cumulative_scores = (
        [round(total / n, 2) for total in cumulative_score_totals] if n else []
    )
    average_points_per_round = round(mean_val / cfg.rounds, 2) if cfg.rounds else 0.0

    summary = {
        "runs": n,
        "theme": theme_name,
        "seed": seed,
        "rounds": cfg.rounds,
        "cooks_per_round": cfg.cooks,
        "active_chefs": cfg.active_chefs,
        "hand_size": cfg.hand_size,
        "pick_size": cfg.pick_size,
        "average_score": round(mean_val, 2),
        "std_score": round(std_val, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "p99": round(p99, 2),
        "mastery_rate_pct": round(100.0 * mastered_any / n, 1) if n else 0.0,
        "avg_chef_key_per_draw": round(avg_chef_keys, 2),
        "ingredient_hhi": round(hhi, 4),
        "avg_recipe_multiplier": round(avg_recipe_multiplier, 2),
        "average_round_scores": average_round_scores,
        "average_cumulative_scores": average_cumulative_scores,
        "average_points_per_round": average_points_per_round,
    }
    return summary, ingredient_totals, taste_totals, recipe_totals, scores


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def format_report_header() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"=== Food Simulator Report (Rules v{RULES_VERSION}) ===\nGenerated: {timestamp}\n"

