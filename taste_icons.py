"""Taste icon mapping for UI layers."""

from __future__ import annotations

TASTE_ICON_MAP = {
    "Sweet": "🍬",
    "Salty": "🧂",
    "Sour": "🍋",
    "Umami": "🍄",
    "Bitter": "☕",
}


def get_taste_icon(taste: str) -> str:
    """Return the emoji associated with a taste name.

    The lookup is case-sensitive. When the taste is not recognised the
    empty string is returned so callers can easily omit the icon.
    """

    return TASTE_ICON_MAP.get(taste, "")
