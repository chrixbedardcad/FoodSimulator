"""Basket challenge generation utilities for the Food Simulator prototype."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from food_api import GameData

DIFFICULTY_ORDER = ("easy", "medium", "hard")


@dataclass(frozen=True)
class BasketChallenge:
    """Serializable description of a generated basket challenge."""

    id: str
    difficulty: str
    basket_name: str
    added_ing_ids: Tuple[str, ...]
    target_score: int
    plays_budget: int
    reward: Mapping[str, str]


class TargetScoreCalculator:
    """Estimate target scores for a basket challenge via Monte Carlo sampling."""

    def __init__(self, data: GameData, cfg: Mapping[str, object]):
        self.data = data
        self.cfg = cfg

    def estimate_target(
        self,
        pantry_ids: Sequence[str],
        added_ids: Sequence[str],
        difficulty: str,
        chefs_active_ids: Sequence[str],
        seasoning_owned_ids: Sequence[str],
        rng: Optional[random.Random] = None,
    ) -> Tuple[int, int]:
        rng = rng or random.Random()
        samples = int(self.cfg.get("samples", 220))
        top_p = float(self.cfg.get("top_percentile", 0.80))
        plays_map = self.cfg.get(
            "plays_by_difficulty", {"easy": 8, "medium": 10, "hard": 12}
        )
        mult_map = self.cfg.get(
            "mult_by_difficulty", {"easy": 0.85, "medium": 1.0, "hard": 1.2}
        )
        plays_budget = int(plays_map.get(difficulty, 10))
        diff_mult = float(mult_map.get(difficulty, 1.0))

        all_ids = list(pantry_ids) + list(added_ids)
        all_ing = [
            self.data.ingredient_for_id(identifier)
            for identifier in all_ids
            if self.data.ingredient_for_id(identifier)
        ]
        if len(all_ing) < 3:
            return (int(self.cfg.get("min_target", 60)), plays_budget)

        lengths = self.data.feasible_dish_sizes()
        if not lengths:
            lengths = list(range(3, min(len(all_ing), 6) + 1))

        scored: List[float] = []
        for _ in range(samples):
            k = rng.choice(lengths)
            if k > len(all_ing):
                k = len(all_ing)
            pick = rng.sample(all_ing, k)
            outcome = self.data.evaluate_dish(pick)
            scored.append(outcome.dish_value)

        scored.sort()
        cut = max(1, int(len(scored) * top_p))
        top_slice = scored[-cut:]
        avg_top = sum(top_slice) / len(top_slice)

        owned = len(seasoning_owned_ids or ())
        cushion = 0.16 if owned >= 4 else (0.08 if owned >= 2 else 0.0)

        target = int(round(avg_top * plays_budget * diff_mult * (1.0 - cushion)))
        target = max(target, int(self.cfg.get("min_target", 60)))
        return (target, plays_budget)


class BasketChallengeFactory:
    """Generate challenge offerings for the basket loop."""

    def __init__(self, data: GameData, target_cfg: Mapping[str, object]):
        self.data = data
        self.target_cfg = target_cfg
        self.tcalc = TargetScoreCalculator(data, target_cfg)

    def three_offers(
        self, run_state: Mapping[str, object], rng: Optional[random.Random] = None
    ) -> Tuple[BasketChallenge, BasketChallenge, BasketChallenge]:
        rng = rng or random.Random()
        names = list(self.data.baskets.keys())
        if len(names) >= 3:
            rng.shuffle(names)
            picks = names[:3]
        else:
            picks = (names * 3)[:3]

        challenges: List[BasketChallenge] = []
        for basket_name, difficulty in zip(picks, DIFFICULTY_ORDER):
            added_ids: List[str] = []
            for ing_name, copies in self.data.baskets.get(basket_name, []):
                ingredient = self.data.ingredients.get(ing_name)
                if not ingredient:
                    continue
                added_ids.extend([ingredient.ingredient_id] * max(0, copies))

            target_score, plays_budget = self.tcalc.estimate_target(
                pantry_ids=run_state.get("pantry", []),
                added_ids=added_ids,
                difficulty=difficulty,
                chefs_active_ids=run_state.get("chefs_active", []),
                seasoning_owned_ids=run_state.get("seasoning_owned", []),
                rng=rng,
            )

            reward = self._pick_reward(difficulty, rng)
            cid = f"{basket_name}_{difficulty}_{abs(hash((basket_name, difficulty, target_score))) % 10000:04d}"
            challenges.append(
                BasketChallenge(
                    id=cid,
                    difficulty=difficulty,
                    basket_name=basket_name,
                    added_ing_ids=tuple(added_ids),
                    target_score=target_score,
                    plays_budget=plays_budget,
                    reward=reward,
                )
            )

        return tuple(challenges)  # type: ignore[return-value]

    def _pick_reward(self, difficulty: str, rng: random.Random) -> Mapping[str, str]:
        if difficulty == "easy":
            pool = (("seasoning", 0.6), ("chef", 0.4))
            rarity_bias = {"common": 1.0, "uncommon": 0.7, "rare": 0.3, "epic": 0.1}
        elif difficulty == "medium":
            pool = (("seasoning", 0.5), ("chef", 0.5))
            rarity_bias = {
                "common": 0.4,
                "uncommon": 0.9,
                "rare": 0.5,
                "epic": 0.2,
            }
        else:
            pool = (("seasoning", 0.4), ("chef", 0.6))
            rarity_bias = {
                "uncommon": 0.8,
                "rare": 0.7,
                "epic": 0.35,
                "legendary": 0.15,
            }

        reward_type = self._weighted_pick(pool, rng)
        rarity = self._rarity_by_bias(rarity_bias, rng)
        return {"type": reward_type, "id": "", "rarity": rarity}

    def _rarity_by_bias(self, bias: Mapping[str, float], rng: random.Random) -> str:
        keys = list(bias.keys())
        weights = [bias[key] for key in keys]
        total = sum(weights) or 1.0
        roll = rng.random() * total
        tally = 0.0
        for key, weight in zip(keys, weights):
            tally += weight
            if roll <= tally:
                return key
        return keys[-1]

    def _weighted_pick(
        self, entries: Sequence[Tuple[str, float]], rng: random.Random
    ) -> str:
        total = sum(weight for _, weight in entries) or 1.0
        roll = rng.random() * total
        tally = 0.0
        for value, weight in entries:
            tally += weight
            if roll <= tally:
                return value
        return entries[-1][0]


__all__ = [
    "BasketChallenge",
    "BasketChallengeFactory",
    "TargetScoreCalculator",
    "DIFFICULTY_ORDER",
]
