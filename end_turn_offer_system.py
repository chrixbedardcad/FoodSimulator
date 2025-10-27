"""End-of-turn offer generation for Food Simulator."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

OptionType = str

RARITY_WEIGHTS: Mapping[str, float] = {
    "common": 1.0,
    "uncommon": 0.6,
    "rare": 0.25,
    "epic": 0.15,
    "legendary": 0.1,
}

RARITY_BONUS: Mapping[str, float] = {
    "common": 0.0,
    "uncommon": 0.25,
    "rare": 0.5,
    "epic": 0.75,
    "legendary": 1.0,
}


@dataclass
class Offer:
    """Container describing an offer presented to the player."""

    type: OptionType
    id: str
    display_name: str
    summary: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the offer."""

        return asdict(self)


class EndTurnOfferSystem:
    """Semi-random, weighted end-of-turn offer generator."""

    def __init__(self, config: Mapping[str, Any], catalogs: Mapping[str, Sequence[Mapping[str, Any]]]):
        self.config = config
        self.catalogs = catalogs
        self._chef_by_id = {entry["id"]: entry for entry in catalogs.get("chefs", [])}
        self._seasoning_by_id = {entry["id"]: entry for entry in catalogs.get("seasonings", [])}
        self._ingredient_by_id = {entry["id"]: entry for entry in catalogs.get("ingredients", [])}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_offers(self, state: Mapping[str, Any]) -> List[Offer]:
        """Generate the weighted end-of-turn offers for the provided state."""

        rng = self._make_rng(state)
        weights = self.compute_option_weights(state)
        offers: List[Offer] = []
        available_types = list(weights.keys())
        pool = dict(weights)
        offer_count = int(self.config["offer_rules"].get("offers_per_turn", 3))
        allow_duplicates = bool(self.config["offer_rules"].get("allow_duplicates", False))

        for _ in range(offer_count):
            if not available_types:
                break
            choice_type = self._weighted_choice(available_types, pool, rng)
            offer = self._build_specific_offer(choice_type, state, rng)
            offers.append(offer)
            if not allow_duplicates:
                pool.pop(choice_type, None)
                available_types = [key for key in available_types if key != choice_type]
                pool = self._normalize(pool)
            else:
                pool = self._normalize(pool)

        if not offers:
            # Fallback to add_ingredient if everything else is exhausted
            fallback_offer = self._build_specific_offer("add_ingredient", state, rng)
            offers.append(fallback_offer)

        return offers

    def apply_choice(self, state: MutableMapping[str, Any], offer: Offer) -> MutableMapping[str, Any]:
        """Apply the selected offer to the run state."""

        offer_type = offer.type
        payload = offer.payload

        if offer_type == "new_chef":
            chef_id = payload["chef_id"]
            max_chefs = int(self.config["caps"].get("max_active_chefs", 99))
            active_chefs: List[str] = list(state.get("chefs_active", []))
            if chef_id in active_chefs:
                # Already active; nothing to do
                pass
            elif len(active_chefs) >= max_chefs:
                swap_out_id = payload.get("swap_out_id")
                if not swap_out_id:
                    raise ValueError(
                        "swap_out_id must be supplied when applying a chef offer at capacity"
                    )
                if swap_out_id not in active_chefs:
                    raise ValueError("swap_out_id must reference an active chef")
                active_chefs.remove(swap_out_id)
                active_chefs.append(chef_id)
                state["chefs_active"] = active_chefs
            else:
                active_chefs.append(chef_id)
                state["chefs_active"] = active_chefs
        elif offer_type == "new_seasoning":
            seasoning_id = payload["seasoning_id"]
            owned = list(state.get("seasoning_owned", []))
            if seasoning_id not in owned:
                owned.append(seasoning_id)
                state["seasoning_owned"] = owned
        elif offer_type == "add_ingredient":
            ingredient_id = payload["ingredient_id"]
            basket = dict(state.get("basket", {}))
            ingredient_ids = list(basket.get("ingredient_ids", []))
            max_size = int(basket.get("max_size", len(ingredient_ids) + 1))
            if len(ingredient_ids) >= max_size:
                raise ValueError("basket is at max capacity; cannot add ingredient")
            ingredient_ids.append(ingredient_id)
            basket["ingredient_ids"] = ingredient_ids
            state["basket"] = basket
        elif offer_type == "remove_ingredient":
            ingredient_id = payload.get("ingredient_id")
            if ingredient_id is None:
                raise ValueError("remove_ingredient payload must include ingredient_id")
            basket = dict(state.get("basket", {}))
            ingredient_ids = list(basket.get("ingredient_ids", []))
            if ingredient_id not in ingredient_ids:
                raise ValueError("ingredient to remove is not present in basket")
            ingredient_ids.remove(ingredient_id)
            basket["ingredient_ids"] = ingredient_ids
            state["basket"] = basket
        else:
            raise ValueError(f"Unknown offer type: {offer_type}")

        state["last_picked_type"] = offer_type
        return state

    # ------------------------------------------------------------------
    # Weight calculations
    # ------------------------------------------------------------------
    def compute_option_weights(self, state: Mapping[str, Any]) -> Dict[OptionType, float]:
        """Compute normalized weights for each option type given the run state."""

        weights = dict(self._weights_for_phase(int(state.get("turn_index", 1))))
        basket = state.get("basket", {})
        basket_size = len(basket.get("ingredient_ids", []))
        max_basket_size = int(basket.get("max_size", basket_size + 1))

        # Availability adjustments
        if "new_chef" in weights and not self._available_chefs(state):
            weights["new_chef"] = 0.0
        if "new_seasoning" in weights and not self._available_seasonings(state):
            weights["new_seasoning"] = 0.0
        if "add_ingredient" in weights and basket_size >= max_basket_size:
            weights["add_ingredient"] = 0.0
        if "remove_ingredient" in weights and basket_size <= 1:
            weights["remove_ingredient"] = 0.0

        # Cooldown logic
        last_picked_type = state.get("last_picked_type")
        min_gap = int(self.config.get("cooldowns", {}).get("same_option_min_gap_turns", 0))
        if last_picked_type and min_gap >= 1 and last_picked_type in weights:
            weights[last_picked_type] = 0.0

        # Pity rules
        weights = self._apply_pity(weights, state)
        weights = self._normalize(weights)

        if not weights:
            # Fallback to add ingredient if nothing else is available
            return {"add_ingredient": 1.0}
        return weights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_rng(self, state: Mapping[str, Any]) -> random.Random:
        rng_config = self.config.get("rng", {})
        strategy = rng_config.get("seed_strategy", "system_time")
        if strategy == "run_id_plus_turn":
            run_id = str(state.get("run_id", ""))
            turn_index = str(state.get("turn_index", 0))
            seed_input = f"{run_id}:{turn_index}".encode("utf-8")
            digest = hashlib.sha256(seed_input).hexdigest()
            seed = int(digest, 16) % (2**32)
            return random.Random(seed)
        if strategy == "fixed_for_debug":
            seed = int(rng_config.get("fixed_seed", 0))
            return random.Random(seed)
        return random.Random()

    def _weights_for_phase(self, turn_index: int) -> Dict[OptionType, float]:
        for entry in self.config.get("phase_weights", []):
            lower, upper = entry.get("turns", [1, 1])
            if lower <= turn_index <= upper:
                return dict(entry.get("weights", {}))
        # Default to first entry if no range matches
        phase_weights = self.config.get("phase_weights", [])
        if phase_weights:
            return dict(phase_weights[0].get("weights", {}))
        return {
            "new_chef": 0.25,
            "new_seasoning": 0.25,
            "add_ingredient": 0.25,
            "remove_ingredient": 0.25,
        }

    def _apply_pity(self, weights: Dict[OptionType, float], state: Mapping[str, Any]) -> Dict[OptionType, float]:
        pity = self.config.get("pity_rules", {})
        chefs_active = state.get("chefs_active", [])
        seasonings_owned = state.get("seasoning_owned", [])
        basket = state.get("basket", {})
        basket_size = len(basket.get("ingredient_ids", []))

        adjusted = dict(weights)

        # Boost new chef if none active
        no_chef_cfg = pity.get("boost_if_no_chef")
        if no_chef_cfg and not chefs_active and "new_chef" in adjusted:
            adjusted["new_chef"] = self._clamped_boost(
                adjusted["new_chef"],
                no_chef_cfg.get("add", 0.0),
                no_chef_cfg.get("cap"),
            )

        # Boost seasoning if low count
        low_season_cfg = pity.get("boost_if_low_seasoning")
        if low_season_cfg and len(seasonings_owned) <= low_season_cfg.get("threshold", 0):
            adjusted["new_seasoning"] = self._clamped_boost(
                adjusted.get("new_seasoning", 0.0),
                low_season_cfg.get("add", 0.0),
                low_season_cfg.get("cap"),
            )

        oversized_cfg = pity.get("boost_if_oversized_deck")
        if oversized_cfg and basket_size >= oversized_cfg.get("size_threshold", 0):
            adjusted["remove_ingredient"] = self._clamped_boost(
                adjusted.get("remove_ingredient", 0.0),
                oversized_cfg.get("remove_add", 0.0),
                oversized_cfg.get("cap"),
            )

        thin_cfg = pity.get("boost_if_thin_deck")
        if thin_cfg and basket_size <= thin_cfg.get("size_threshold", 0):
            adjusted["add_ingredient"] = self._clamped_boost(
                adjusted.get("add_ingredient", 0.0),
                thin_cfg.get("add_add", 0.0),
                thin_cfg.get("cap"),
            )

        return adjusted

    def _clamped_boost(self, base: float, add: float, cap: Optional[float]) -> float:
        boosted = base + add
        if cap is None:
            return boosted
        if base > cap:
            return base
        return min(boosted, cap)

    def _normalize(self, weights: Mapping[OptionType, float]) -> Dict[OptionType, float]:
        positive = {k: v for k, v in weights.items() if v > 0}
        total = sum(positive.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in positive.items()}

    def _weighted_choice(
        self,
        choices: Sequence[OptionType],
        weights: Mapping[OptionType, float],
        rng: random.Random,
    ) -> OptionType:
        total = sum(weights.get(choice, 0.0) for choice in choices)
        if total <= 0:
            return choices[0]
        threshold = rng.random() * total
        cumulative = 0.0
        for choice in choices:
            cumulative += weights.get(choice, 0.0)
            if threshold <= cumulative:
                return choice
        return choices[-1]

    # ------------------------------------------------------------------
    # Offer construction
    # ------------------------------------------------------------------
    def _build_specific_offer(self, offer_type: OptionType, state: Mapping[str, Any], rng: random.Random) -> Offer:
        if offer_type == "new_chef":
            return self._build_new_chef_offer(state, rng)
        if offer_type == "new_seasoning":
            return self._build_new_seasoning_offer(state, rng)
        if offer_type == "add_ingredient":
            return self._build_add_ingredient_offer(state, rng)
        if offer_type == "remove_ingredient":
            return self._build_remove_ingredient_offer(state, rng)
        raise ValueError(f"Unsupported offer type: {offer_type}")

    def _build_new_chef_offer(self, state: Mapping[str, Any], rng: random.Random) -> Offer:
        available = self._available_chefs(state)
        if not available:
            # Provide a no-op fallback offer
            return Offer(
                type="new_chef",
                id="none",
                display_name="No Chef Available",
                summary="All chefs recruited.",
                payload={"chef_id": ""},
            )
        pick = self._pick_by_rarity(available, rng)
        max_chefs = int(self.config.get("caps", {}).get("max_active_chefs", 99))
        requires_swap = len(state.get("chefs_active", [])) >= max_chefs
        payload = {"chef_id": pick["id"], "requires_swap": requires_swap}
        summary = self._format_chef_summary(pick, requires_swap)
        return Offer(
            type="new_chef",
            id=pick["id"],
            display_name=pick.get("name", self._title_from_id(pick["id"])),
            summary=summary,
            payload=payload,
        )

    def _build_new_seasoning_offer(self, state: Mapping[str, Any], rng: random.Random) -> Offer:
        available = self._available_seasonings(state)
        if not available:
            return Offer(
                type="new_seasoning",
                id="none",
                display_name="No Seasoning Available",
                summary="All seasonings collected.",
                payload={"seasoning_id": ""},
            )
        pick = self._pick_by_rarity(available, rng)
        summary = self._format_seasoning_summary(pick)
        return Offer(
            type="new_seasoning",
            id=pick["id"],
            display_name=pick.get("name", self._title_from_id(pick["id"])),
            summary=summary,
            payload={"seasoning_id": pick["id"]},
        )

    def _build_add_ingredient_offer(self, state: Mapping[str, Any], rng: random.Random) -> Offer:
        basket_ids = list(state.get("basket", {}).get("ingredient_ids", []))
        candidates = list(self.catalogs.get("ingredients", []))
        if not candidates:
            return Offer(
                type="add_ingredient",
                id="none",
                display_name="No Ingredient Available",
                summary="Ingredient catalog exhausted.",
                payload={"ingredient_id": ""},
            )

        scored = [(self._ingredient_synergy(entry, state), entry) for entry in candidates]
        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        top_score, top_entry = scored[0]
        summary = self._format_add_ingredient_summary(top_entry, top_score)
        payload = {
            "ingredient_id": top_entry["id"],
            "synergy_score": top_score,
        }
        return Offer(
            type="add_ingredient",
            id=top_entry["id"],
            display_name=top_entry.get("name", self._title_from_id(top_entry["id"])),
            summary=summary,
            payload=payload,
        )

    def _build_remove_ingredient_offer(self, state: Mapping[str, Any], rng: random.Random) -> Offer:
        basket_ids = list(state.get("basket", {}).get("ingredient_ids", []))
        if not basket_ids:
            return Offer(
                type="remove_ingredient",
                id="none",
                display_name="No Ingredient To Remove",
                summary="Basket already empty.",
                payload={"ingredient_id": ""},
            )
        scored: List[tuple[float, str]] = []
        for ingredient_id in basket_ids:
            ingredient = self._ingredient_by_id.get(ingredient_id)
            if not ingredient:
                continue
            score = self._ingredient_synergy(ingredient, state)
            scored.append((score, ingredient_id))
        if not scored:
            ingredient_id = basket_ids[0]
            return Offer(
                type="remove_ingredient",
                id=ingredient_id,
                display_name=self._title_from_id(ingredient_id),
                summary="Remove an unknown ingredient.",
                payload={"ingredient_id": ingredient_id},
            )
        scored.sort(key=lambda item: (item[0], item[1]))
        candidate_ids = [item[1] for item in scored[:5]]
        weakest_score, weakest_id = scored[0]
        summary = self._format_remove_summary(candidate_ids, weakest_score)
        payload = {
            "ingredient_id": weakest_id,
            "candidate_ids": candidate_ids,
            "suggested_score": weakest_score,
        }
        ingredient_entry = self._ingredient_by_id.get(weakest_id, {"id": weakest_id})
        return Offer(
            type="remove_ingredient",
            id=weakest_id,
            display_name=ingredient_entry.get("name", self._title_from_id(weakest_id)),
            summary=summary,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Availability helpers
    # ------------------------------------------------------------------
    def _available_chefs(self, state: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        owned = set(state.get("chefs_active", []))
        return [entry for entry in self.catalogs.get("chefs", []) if entry["id"] not in owned]

    def _available_seasonings(self, state: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        owned = set(state.get("seasoning_owned", []))
        return [entry for entry in self.catalogs.get("seasonings", []) if entry["id"] not in owned]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _format_chef_summary(self, entry: Mapping[str, Any], requires_swap: bool) -> str:
        perk_summaries = []
        for perk in entry.get("perks", []):
            perk_type = perk.get("type")
            if perk_type == "taste_mult":
                perk_summaries.append(
                    f"{perk.get('taste', 'taste')} taste x{perk.get('mult', 1.0):.2f}"
                )
            elif perk_type == "family_mult":
                perk_summaries.append(
                    f"{perk.get('family', 'family').title()} family x{perk.get('mult', 1.0):.2f}"
                )
            elif perk_type == "taste_add":
                perk_summaries.append(
                    f"Adds {perk.get('value', 0)} {perk.get('taste', 'taste')} taste"
                )
        summary = ", ".join(perk_summaries) if perk_summaries else "Unique chef perks"
        if requires_swap:
            summary += " (requires swap)"
        return summary

    def _format_seasoning_summary(self, entry: Mapping[str, Any]) -> str:
        parts = []
        for effect in entry.get("effects", []):
            effect_type = effect.get("type")
            if effect_type == "taste_add":
                parts.append(
                    f"Adds {effect.get('value', 0)} {effect.get('taste', 'taste')}"
                )
            elif effect_type == "synergy_window":
                parts.append(f"Synergy tolerance {effect.get('tolerance')}")
            else:
                parts.append(effect_type.replace("_", " ").title())
        return ", ".join(parts) if parts else "Seasoning bonus"

    def _format_add_ingredient_summary(self, entry: Mapping[str, Any], score: float) -> str:
        tastes = ", ".join(entry.get("tastes", []))
        families = ", ".join(entry.get("families", []))
        return f"Adds {tastes} tastes; synergy score {score:.2f}; families: {families}"

    def _format_remove_summary(self, candidate_ids: Sequence[str], score: float) -> str:
        names = [
            self._ingredient_by_id.get(identifier, {}).get("name", self._title_from_id(identifier))
            for identifier in candidate_ids
        ]
        joined = ", ".join(names)
        return f"Consider removing: {joined} (lowest score {score:.2f})"

    def _title_from_id(self, identifier: str) -> str:
        return identifier.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _pick_by_rarity(self, entries: Sequence[Mapping[str, Any]], rng: random.Random) -> Mapping[str, Any]:
        weights = []
        for entry in entries:
            rarity = entry.get("rarity", "common").lower()
            weight = RARITY_WEIGHTS.get(rarity, 1.0)
            weights.append(weight)
        total = sum(weights)
        if total <= 0:
            return entries[0]
        threshold = rng.random() * total
        cumulative = 0.0
        for entry, weight in zip(entries, weights):
            cumulative += weight
            if threshold <= cumulative:
                return entry
        return entries[-1]

    def _ingredient_synergy(self, ingredient: Mapping[str, Any], state: Mapping[str, Any]) -> float:
        tastes = set(ingredient.get("tastes", []))
        families = set(ingredient.get("families", []))
        basket_ids = list(state.get("basket", {}).get("ingredient_ids", []))
        basket_entries = [self._ingredient_by_id.get(identifier) for identifier in basket_ids]

        taste_overlap = 0
        family_overlap = 0
        for other in basket_entries:
            if not other:
                continue
            taste_overlap += len(tastes.intersection(other.get("tastes", [])))
            family_overlap += len(families.intersection(other.get("families", [])))

        chef_bonus = 0.0
        for chef_id in state.get("chefs_active", []):
            chef = self._chef_by_id.get(chef_id)
            if not chef:
                continue
            for perk in chef.get("perks", []):
                perk_type = perk.get("type")
                if perk_type == "family_mult" and perk.get("family") in families:
                    chef_bonus += 1.5
                elif perk_type == "taste_mult" and perk.get("taste") in tastes:
                    chef_bonus += 1.0

        seasoning_bonus = 0.0
        for seasoning_id in state.get("seasoning_owned", []):
            seasoning = self._seasoning_by_id.get(seasoning_id)
            if not seasoning:
                continue
            for effect in seasoning.get("effects", []):
                if effect.get("type") == "taste_add" and effect.get("taste") in tastes:
                    seasoning_bonus += 0.5

        rarity = ingredient.get("rarity", "common").lower()
        rarity_bonus = RARITY_BONUS.get(rarity, 0.0)

        score = (
            taste_overlap * 1.0
            + family_overlap * 1.5
            + chef_bonus
            + seasoning_bonus
            + rarity_bonus
        )
        return score


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_catalogs(
    chef_path: str,
    seasoning_path: str,
    ingredient_path: str,
) -> Dict[str, Sequence[Mapping[str, Any]]]:
    with open(chef_path, "r", encoding="utf-8") as handle:
        chefs = json.load(handle)
    with open(seasoning_path, "r", encoding="utf-8") as handle:
        seasonings = json.load(handle)
    with open(ingredient_path, "r", encoding="utf-8") as handle:
        ingredients = json.load(handle)
    return {
        "chefs": chefs,
        "seasonings": seasonings,
        "ingredients": ingredients,
    }


__all__ = [
    "EndTurnOfferSystem",
    "Offer",
    "load_catalogs",
    "load_config",
]
