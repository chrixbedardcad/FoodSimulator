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
DEFAULT_DISH_MATRIX_JSON = "dish_matrix.json"
DEFAULT_RECIPES_JSON = "recipes.json"
DEFAULT_CHEFS_JSON = "chefs.json"
DEFAULT_THEMES_JSON = "themes.json"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class Ingredient:
    name: str
    ingredient_id: str
    taste: str
    Value: int
    family: str


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


@dataclass(frozen=True)
class DishMatrixEntry:
    id: int
    name: str
    min_ingredients: int
    max_ingredients: int
    family_pattern: str
    flavor_pattern: str
    multiplier: float
    tier: str
    chance: float
    description: str

    def matches(
        self, count: int, family_pattern: str, flavor_pattern: str
    ) -> bool:
        return (
            self.min_ingredients <= count <= self.max_ingredients
            and self.family_pattern == family_pattern
            and self.flavor_pattern == flavor_pattern
        )


FAMILY_LABELS = {
    "all_same": "Harmony",
    "all_different": "Rich",
    "balanced": "Balanced",
}

FLAVOR_LABELS = {
    "all_same": "Terrible",
    "all_different": "Tasteful",
    "mixed": "Neutral",
    "single_taste_varied_family": "Unified",
}


def describe_family_pattern(pattern: str) -> str:
    return FAMILY_LABELS.get(pattern, pattern.replace("_", " ").title())


def describe_flavor_pattern(pattern: str) -> str:
    return FLAVOR_LABELS.get(pattern, pattern.replace("_", " ").title())


@dataclass(frozen=True)
class DishOutcome:
    base_value: int
    dish_value: float
    dish_multiplier: float
    family_pattern: str
    flavor_pattern: str
    family_label: str
    flavor_label: str
    entry: Optional[DishMatrixEntry]
    alerts: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> Optional[str]:
        return self.entry.name if self.entry else None

    @property
    def tier(self) -> Optional[str]:
        return self.entry.tier if self.entry else None

    def is_terrible(self) -> bool:
        return self.entry is None and self.flavor_pattern == "all_same"


@dataclass
class GameData:
    ingredients: Dict[str, Ingredient]
    recipes: List[Recipe]
    chefs: List[Chef]
    themes: Dict[str, List[Tuple[str, int]]]
    taste_matrix: MutableMapping[str, MutableMapping[str, int]]
    dish_matrix: List[DishMatrixEntry]
    rules: Mapping[str, object] = field(default_factory=dict)
    recipe_by_name: Dict[str, Recipe] = field(init=False)
    recipe_trio_lookup: Dict[Tuple[str, str, str], str] = field(init=False)
    recipe_multipliers: Dict[str, float] = field(init=False)
    ingredient_recipes: Dict[str, List[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.recipe_by_name = {recipe.name: recipe for recipe in self.recipes}
        self.recipe_trio_lookup = {
            tuple(sorted(recipe.trio)): recipe.name for recipe in self.recipes
        }
        self.recipe_multipliers = self._build_recipe_multipliers()
        self.ingredient_recipes = self._build_ingredient_recipes()

    def _build_recipe_multipliers(self) -> Dict[str, float]:
        return {recipe.name: float(recipe.base_multiplier) for recipe in self.recipes}

    def _build_ingredient_recipes(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for recipe in self.recipes:
            for ingredient in recipe.trio:
                mapping.setdefault(ingredient, []).append(recipe.name)
        for names in mapping.values():
            names.sort()
        return mapping

    @classmethod
    def from_json(
        cls,
        ingredients_path: str = DEFAULT_INGREDIENTS_JSON,
        recipes_path: str = DEFAULT_RECIPES_JSON,
        chefs_path: str = DEFAULT_CHEFS_JSON,
        taste_path: str = DEFAULT_TASTE_JSON,
        themes_path: str = DEFAULT_THEMES_JSON,
        dish_matrix_path: str = DEFAULT_DISH_MATRIX_JSON,
    ) -> "GameData":
        dish_matrix_entries, rules = _load_dish_matrix(dish_matrix_path)
        return cls(
            ingredients=_load_ingredients(ingredients_path),
            recipes=_load_recipes(recipes_path),
            chefs=_load_chefs(chefs_path),
            themes=_load_themes(themes_path),
            taste_matrix=_load_taste_matrix(taste_path),
            dish_matrix=dish_matrix_entries,
            rules=rules,
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
        outcome = self.evaluate_dish(ingredients)
        return (
            int(round(outcome.dish_value)),
            outcome.base_value,
            0,
            int(round(outcome.dish_multiplier * 100))
            if outcome.dish_multiplier
            else 0,
        )

    def which_recipe(self, ingredients: Sequence[Ingredient]) -> Optional[str]:
        key = tuple(sorted(ingredient.name for ingredient in ingredients))
        return self.recipe_trio_lookup.get(key)

    def recipes_using_ingredient(self, ingredient_name: str) -> Sequence[str]:
        return self.ingredient_recipes.get(ingredient_name, [])

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
        return float(total)

    def evaluate_dish(self, ingredients: Sequence[Ingredient]) -> DishOutcome:
        if not ingredients:
            return DishOutcome(
                base_value=0,
                dish_value=0.0,
                dish_multiplier=0.0,
                family_pattern="",
                flavor_pattern="",
                family_label="",
                flavor_label="",
                entry=None,
            )

        base_value = sum(ingredient.Value for ingredient in ingredients)
        families = [ingredient.family for ingredient in ingredients]
        tastes = [ingredient.taste for ingredient in ingredients]

        unique_families = set(families)
        if len(unique_families) == 1:
            family_pattern = "all_same"
        elif len(unique_families) == len(families):
            family_pattern = "all_different"
        else:
            family_pattern = "balanced"

        unique_tastes = set(tastes)
        if len(unique_tastes) == 1:
            if len(unique_families) == len(families) and len(unique_families) > 1:
                flavor_pattern = "single_taste_varied_family"
            else:
                flavor_pattern = "all_same"
        elif len(unique_tastes) == len(tastes):
            flavor_pattern = "all_different"
        else:
            flavor_pattern = "mixed"

        entry: Optional[DishMatrixEntry] = None
        alerts: List[str] = []
        count = len(ingredients)
        multiplier = 1.0
        has_duplicates = False
        effective_penalty_multiplier = 1.0

        entry = self._match_dish_matrix(count, family_pattern, flavor_pattern)
        if entry:
            multiplier = float(entry.multiplier)
        elif flavor_pattern == "all_same":
            multiplier = 0.0

        dish_value = float(base_value) * multiplier

        penalty_rules: Optional[Mapping[str, object]] = None
        if isinstance(self.rules, Mapping):
            maybe = self.rules.get("duplicate_ingredient_penalty")
            if isinstance(maybe, Mapping):
                penalty_rules = maybe

        if penalty_rules and penalty_rules.get("enabled", True):
            applies_to = str(penalty_rules.get("applies_to", "")).lower()
            if applies_to in ("", "same_ingredient", "same_ingredient_exact"):
                def _ingredient_identifier(item: Ingredient) -> str:
                    for attr in ("ingredient_id", "id"):
                        value = getattr(item, attr, None)
                        if value is not None:
                            return str(value)
                    return item.name

                identifiers = [_ingredient_identifier(ingredient) for ingredient in ingredients]
                id_to_names: Dict[str, List[str]] = {}
                for ingredient, identifier in zip(ingredients, identifiers):
                    id_to_names.setdefault(identifier, []).append(ingredient.name)
                counts = Counter(identifiers)
                duplicate_ids = [key for key, value in counts.items() if value > 1]

                if duplicate_ids:
                    has_duplicates = True
                    duplicate_messages: List[str] = []
                    per_copy_raw = penalty_rules.get("per_copy_multipliers", {})
                    per_copy_multipliers: Dict[str, float] = {}
                    if isinstance(per_copy_raw, Mapping):
                        for key, value in per_copy_raw.items():
                            try:
                                per_copy_multipliers[str(key)] = float(value)
                            except (TypeError, ValueError):
                                continue

                    default_after_defined = penalty_rules.get("default_after_defined", 1.0)
                    try:
                        fallback = float(default_after_defined)
                    except (TypeError, ValueError):
                        fallback = 1.0

                    max_copies_raw = penalty_rules.get("max_copies_scored", 0)
                    try:
                        max_copies = int(max_copies_raw)
                    except (TypeError, ValueError):
                        max_copies = 0
                    if max_copies <= 0:
                        max_copies = None

                    application = str(penalty_rules.get("application", "global")).lower()

                    def _lookup_multiplier(copy_count: int) -> float:
                        effective = copy_count
                        if max_copies is not None:
                            effective = min(copy_count, max_copies)
                        value = per_copy_multipliers.get(str(effective))
                        if value is None:
                            return fallback
                        return value

                    over_taste_ids = [
                        identifier
                        for identifier in duplicate_ids
                        if counts[identifier] > 2
                    ]
                    if over_taste_ids:
                        labelled = []
                        for identifier in over_taste_ids:
                            names = id_to_names.get(identifier) or [identifier]
                            name = names[0]
                            labelled.append(f"{name} (x{counts[identifier]})")
                        joined = ", ".join(sorted(labelled))
                        alerts.append(
                            f"Recipe is over taste of the duplicated ingredient: {joined}."
                        )

                    for identifier in duplicate_ids:
                        count = counts[identifier]
                        factor = _lookup_multiplier(count)
                        change = int(round((1 - factor) * 100))
                        names = id_to_names.get(identifier) or [identifier]
                        primary_name = names[0]
                        if change > 0:
                            duplicate_messages.append(
                                f"Recipe has too much {primary_name} (x{count}), penalty -{change}% to scoring."
                            )
                        elif change < 0:
                            duplicate_messages.append(
                                f"Recipe has extra {primary_name} (x{count}), bonus +{abs(change)}% to scoring."
                            )
                        else:
                            duplicate_messages.append(
                                f"Recipe has repeated {primary_name} (x{count}) with no scoring change."
                            )

                    if application == "per_card":
                        seen = Counter()
                        penalized_total = 0.0
                        for ingredient in ingredients:
                            identifier = _ingredient_identifier(ingredient)
                            seen[identifier] += 1
                            copy_index = seen[identifier]
                            if copy_index == 1:
                                factor = 1.0
                            else:
                                factor = _lookup_multiplier(copy_index)
                            penalized_total += float(ingredient.Value) * factor
                        dish_value = float(penalized_total) * multiplier
                        if base_value:
                            effective_penalty_multiplier = (
                                float(penalized_total) / float(base_value)
                            )
                    else:
                        penalty_multiplier = 1.0
                        for identifier in duplicate_ids:
                            count = counts[identifier]
                            factor = _lookup_multiplier(count)
                            penalty_multiplier *= factor
                        dish_value *= penalty_multiplier
                        effective_penalty_multiplier = penalty_multiplier

                    if duplicate_messages:
                        alerts.extend(duplicate_messages)

        if (
            multiplier == 0.0
            and has_duplicates
            and effective_penalty_multiplier < 1.0
        ):
            dish_value = float(base_value) * effective_penalty_multiplier

        if base_value:
            final_multiplier = float(dish_value) / float(base_value)
        else:
            final_multiplier = 0.0

        return DishOutcome(
            base_value=base_value,
            dish_value=dish_value,
            dish_multiplier=final_multiplier,
            family_pattern=family_pattern,
            flavor_pattern=flavor_pattern,
            family_label=describe_family_pattern(family_pattern),
            flavor_label=describe_flavor_pattern(flavor_pattern),
            entry=entry,
            alerts=tuple(alerts),
        )

    def _match_dish_matrix(
        self, count: int, family_pattern: str, flavor_pattern: str
    ) -> Optional[DishMatrixEntry]:
        for entry in self.dish_matrix:
            if entry.matches(count, family_pattern, flavor_pattern):
                return entry

        if flavor_pattern == "all_different":
            for entry in self.dish_matrix:
                if (
                    entry.min_ingredients <= count <= entry.max_ingredients
                    and entry.family_pattern == family_pattern
                    and entry.flavor_pattern == "mixed"
                ):
                    return entry

        return None


def _load_ingredients(path: str) -> Dict[str, Ingredient]:
    raw = load_json(path)
    ingredients: Dict[str, Ingredient] = {}
    for entry in raw:
        name = entry["name"]
        identifier = entry.get("ingredient_id") or f"ing.{name.lower()}"
        ingredient = Ingredient(
            name,
            str(identifier),
            entry["taste"],
            int(entry["Value"]),
            entry.get("family", "Unknown"),
        )
        ingredients[name] = ingredient
    return ingredients


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


def _load_dish_matrix(path: str) -> Tuple[List[DishMatrixEntry], Mapping[str, object]]:
    raw = load_json(path)
    entries: List[DishMatrixEntry] = []
    for entry in raw.get("dish_matrix", []):
        entries.append(
            DishMatrixEntry(
                id=int(entry["id"]),
                name=str(entry["name"]),
                min_ingredients=int(entry["min_ingredients"]),
                max_ingredients=int(entry["max_ingredients"]),
                family_pattern=str(entry["family_pattern"]),
                flavor_pattern=str(entry["flavor_pattern"]),
                multiplier=float(entry["multiplier"]),
                tier=str(entry["tier"]),
                chance=float(entry["chance"]),
                description=str(entry["description"]),
            )
        )
    rules_raw = raw.get("rules", {})
    if isinstance(rules_raw, Mapping):
        rules: Mapping[str, object] = dict(rules_raw)
    else:
        rules = {}
    return entries, rules


def _load_themes(path: str):
    raw = load_json(path)
    fixed: Dict[str, List[Tuple[str, int]]] = {}
    for theme_name, items in raw.items():
        fixed[theme_name] = [
            (item["ingredient"], int(item["copies"])) for item in items
        ]
    return fixed


DEFAULT_HAND_SIZE = 8
DEFAULT_PICK_SIZE = 5
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

            dish = data.evaluate_dish(trio)
            recipe_multiplier = data.recipe_multiplier(
                recipe_name,
                chefs=active_chefs,
                times_cooked=times_cooked_before,
            )
            final_score = int(round(dish.dish_value * recipe_multiplier))
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

