"""Ingredient family icon mapping for UI layers."""

from __future__ import annotations

FAMILY_ICON_MAP = {
    "Protein": "🥩",
    "Vegetable": "🥦",
    "Grain": "🍚",
    "Dairy": "🧀",
    "Fruit": "🍎",
}


def get_family_icon(family: str) -> str:
    """Return the emoji associated with an ingredient family name.

    The lookup is case-sensitive. When the family is not recognised the
    empty string is returned so callers can easily omit the icon.
    """

    return FAMILY_ICON_MAP.get(family, "")
