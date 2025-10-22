"""Utilities for creating reproducible RNG seeds shared by the simulator and game."""
from __future__ import annotations

import random
from typing import Optional, Tuple

MAX_SEED_VALUE = 2**32 - 1


def resolve_seed(seed: Optional[int]) -> Tuple[int, random.Random]:
    """Return a normalized seed and a Random instance seeded with it.

    When ``seed`` is ``None`` a fresh seed is generated using ``SystemRandom`` so it is
    unpredictable yet recorded for later reuse.  The returned integer can be fed back to
    either the simulator (``food_simulator.py``) or the interactive game
    (``food_game.py``) to recreate a session.
    """

    if seed is None:
        seed = random.SystemRandom().randint(0, MAX_SEED_VALUE)
    else:
        seed = int(seed)
    return seed, random.Random(seed)

