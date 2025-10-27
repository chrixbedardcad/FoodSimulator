import copy
from pathlib import Path

import pytest

from end_turn_offer_system import EndTurnOfferSystem, load_catalogs, load_config


@pytest.fixture(scope="module")
def offer_system():
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "end_turn_config.json")
    catalogs = load_catalogs(
        root / "chefs_catalog.json",
        root / "seasonings_catalog.json",
        root / "ingredients_catalog.json",
    )
    return EndTurnOfferSystem(config, catalogs)


def base_state(**overrides):
    state = {
        "run_id": "test-run",
        "turn_index": 1,
        "basket": {
            "ingredient_ids": ["tomato", "bacon", "lemon"],
            "max_size": 20,
        },
        "seasoning_owned": [],
        "chefs_active": [],
        "last_offered_types": [],
        "last_picked_type": None,
    }
    state.update(overrides)
    return state


def test_new_chef_weight_is_boosted(offer_system):
    state = base_state()
    weights = offer_system.compute_option_weights(state)
    expected_raw = {
        "new_chef": 0.30 + 0.15,
        "new_seasoning": 0.20 + 0.10,
        "add_ingredient": 0.40 + 0.10,
        "remove_ingredient": 0.10,
    }
    expected_total = sum(expected_raw.values())
    expected_weight = expected_raw["new_chef"] / expected_total
    assert weights["new_chef"] == pytest.approx(expected_weight, rel=1e-5)


def test_cooldown_removes_recent_pick(offer_system):
    state = base_state(last_picked_type="remove_ingredient")
    weights = offer_system.compute_option_weights(state)
    assert "remove_ingredient" not in weights


def test_generate_unique_offers(offer_system):
    state = base_state()
    offers = offer_system.generate_offers(state)
    types = {offer.type for offer in offers}
    assert len(types) == len(offers)
    assert len(offers) == offer_system.config["offer_rules"]["offers_per_turn"]


def test_generate_is_deterministic(offer_system):
    state = base_state(turn_index=5, run_id="deterministic")
    offers_first = [offer.as_dict() for offer in offer_system.generate_offers(state)]
    offers_second = [offer.as_dict() for offer in offer_system.generate_offers(state)]
    assert offers_first == offers_second


def test_apply_choice_adds_ingredient(offer_system):
    state = base_state()
    offers = offer_system.generate_offers(state)
    add_offer = next(offer for offer in offers if offer.type == "add_ingredient")
    mutable_state = copy.deepcopy(state)
    offer_system.apply_choice(mutable_state, add_offer)
    assert add_offer.payload["ingredient_id"] in mutable_state["basket"]["ingredient_ids"]
    assert mutable_state["last_picked_type"] == "add_ingredient"
