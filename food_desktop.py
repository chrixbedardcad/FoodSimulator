"""Desktop GUI harness for Food Deck Simulator.

This module provides a Tkinter-based application that reuses the shared
`food_api` rules so designers can explore the card game with a richer visual
interface.  The gameplay loop mirrors the CLI version in ``food_game.py``: pick
chefs, choose a market basket, draw ingredient hands, and cook trios to chase high
scores.  No networking is involved—everything runs in-process on top of the
existing data files.
"""
from __future__ import annotations

import math
import random
import re
import sys
import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

from tkinter import ttk
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ModuleNotFoundError:  # pragma: no cover - optional dependency for CLI debug mode
    Image = ImageDraw = ImageFont = ImageTk = None  # type: ignore[assignment]

from basket_challenges import BasketChallenge, BasketChallengeFactory
from food_api import (
    DEFAULT_HAND_SIZE,
    DEFAULT_PICK_SIZE,
    DEFAULT_CHEFS_JSON,
    DEFAULT_INGREDIENTS_JSON,
    DEFAULT_RECIPES_JSON,
    DEFAULT_SEASONINGS_JSON,
    DEFAULT_TASTE_JSON,
    DEFAULT_BASKETS_JSON,
    DEFAULT_DISH_MATRIX_JSON,
    Chef,
    DishMatrixEntry,
    DishOutcome,
    GameData,
    Ingredient,
    Seasoning,
    SimulationConfig,
    describe_family_pattern,
    describe_flavor_pattern,
    quantize_multiplier,
    build_market_deck,
    distribute_unique_draws,
)
from rotting_round import IngredientCard, rot_circles
from seed_utils import resolve_seed


def _format_percent(value: int) -> str:
    if value < 0:
        return f"{value}%"
    return f"+{value}%"
ASSET_DIR = Path(__file__).resolve().parent
ICON_ASSET_DIR = ASSET_DIR / "icons"
INGREDIENT_ASSET_DIR = ASSET_DIR / "Ingredients"
RECIPE_ASSET_DIR = ASSET_DIR / "recipes"
BASKET_ART_DIR = ASSET_DIR / "baskets"


TASTE_ICON_FILES: Mapping[str, str] = {
    "Sweet": "Sweet.png",
    "Salty": "Salty.png",
    "Sour": "Sour.png",
    "Umami": "Umami.png",
    "Bitter": "Bitter.png",
}


FAMILY_ICON_FILES: Mapping[str, str] = {
    "Protein": "Protein.png",
    "Vegetable": "Vegetable.png",
    "Grain": "Grain.png",
    "Dairy": "Dairy.png",
    "Fruit": "Fruit.png",
}


ICON_TARGET_PX = 64
INGREDIENT_DIALOG_ICON_PX = 72
DIALOG_ICON_TARGET_PX = 40
RESOURCE_BUTTON_ICON_PX = 72
COOKBOOK_ICON_TARGET_PX = 24
RUINED_SEASONING_MESSAGES = [
    "You seasoned with your heart… and your heart was too salty.",
    "The dish called— it wants a lifeguard. It’s drowning in mustard.",
    "Bold choice. Jury’s still chewing.",
    "Somewhere, a nonna just shook her head.",
    "We discovered a new taste today: Regret.",
]
INGREDIENT_IMAGE_TARGET_PX = 160
RECIPE_IMAGE_TARGET_PX = 240

TARGET_SCORE_CONFIG = {
    "samples": 220,
    "top_percentile": 0.80,
    "plays_by_difficulty": {"easy": 8, "medium": 10, "hard": 12},
    "mult_by_difficulty": {"easy": 0.85, "medium": 1.00, "hard": 1.20},
    "perk_cushion": 0.08,
    "min_target": 60,
}


def format_challenge_reward_text(reward: Mapping[str, str]) -> str:
    """Return a readable description of a basket reward."""

    reward_type = reward.get("type", "reward").replace("_", " ").strip().lower()
    rarity = reward.get("rarity", "").strip().lower()

    descriptor_parts = [part for part in (rarity, reward_type) if part]
    descriptor = " ".join(descriptor_parts) or "reward"
    article = "an" if descriptor[:1] in "aeiou" else "a"
    return f"{article} {descriptor}"


def challenge_ingredient_counts(
    data: GameData, challenge: BasketChallenge
) -> Tuple[int, int]:
    """Return (unique ingredient count, total card count) for a basket challenge."""

    entries = data.baskets.get(challenge.basket_name, [])
    unique_count = len(entries)
    total_count = sum(max(0, copies) for _name, copies in entries)

    if total_count <= 0:
        total_count = len(challenge.added_ing_ids)
    if unique_count <= 0 and challenge.added_ing_ids:
        unique_count = len(set(challenge.added_ing_ids))

    return unique_count, total_count


def format_challenge_ingredient_text(unique_count: int, total_count: int) -> str:
    """Return readable text describing ingredient counts for a challenge."""

    if total_count > 0:
        if unique_count == total_count or unique_count <= 0:
            return f"{total_count} ingredients"
        return f"{unique_count} ingredients ({total_count} cards)"
    return "No listed ingredients"


if Image is not None:
    try:
        RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:  # Pillow<9.1 fallback
        RESAMPLE_LANCZOS = Image.LANCZOS
else:  # pragma: no cover - CLI debug path when Pillow missing
    RESAMPLE_LANCZOS = None


_icon_cache: Dict[str, tk.PhotoImage] = {}
_ingredient_image_cache: Dict[str, tk.PhotoImage] = {}
_seasoning_icon_cache: Dict[str, tk.PhotoImage] = {}
_chef_icon_cache: Dict[str, tk.PhotoImage] = {}
_button_icon_cache: Dict[str, tk.PhotoImage] = {}
_recipe_image_cache: Dict[str, Optional[tk.PhotoImage]] = {}
_cookbook_indicator_cache: Dict[str, Optional[tk.PhotoImage]] = {}
_challenge_tile_image_cache: Dict[Tuple[str, int], Optional[tk.PhotoImage]] = {}


RecipeIconState = Literal["available", "blocked"]


def _recipe_asset_directories() -> List[Path]:
    """Return available recipe artwork directories in preferred order."""

    return [RECIPE_ASSET_DIR]


def _load_icon(
    category: str, name: str, *, target_px: Optional[int] = None
) -> Optional[tk.PhotoImage]:
    if not name:
        return None

    mapping = TASTE_ICON_FILES if category == "taste" else FAMILY_ICON_FILES
    filename = mapping.get(name)
    if not filename:
        return None

    icon_path = ICON_ASSET_DIR / filename
    if not icon_path.exists():
        return None

    target = target_px if target_px is not None else ICON_TARGET_PX
    cache_key = f"{category}:{name}:{target}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    with Image.open(icon_path) as source_image:
        working = source_image.convert("RGBA")
        max_side = max(working.size)
        if max_side > target:
            scale = target / max_side
            new_size = (
                max(1, int(round(working.width * scale))),
                max(1, int(round(working.height * scale))),
            )
            working = working.resize(new_size, RESAMPLE_LANCZOS)
        else:
            working = working.copy()

    image = ImageTk.PhotoImage(working)
    _icon_cache[cache_key] = image
    return image


def _candidate_image_basenames(values: Iterable[Optional[str]]) -> List[str]:
    seen: List[str] = []
    for value in values:
        if not value:
            continue
        variants = {
            value,
            value.lower(),
            value.replace(" ", "_"),
            value.replace(" ", "_").lower(),
            re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_"),
        }
        for candidate in variants:
            if candidate and candidate not in seen:
                seen.append(candidate)
    return seen


def _find_ingredient_image_path(ingredient: Ingredient) -> Optional[Path]:
    if not INGREDIENT_ASSET_DIR.exists():
        return None

    candidates = _candidate_image_basenames(
        (
            getattr(ingredient, "display_name", None),
            ingredient.name,
            getattr(ingredient, "ingredient_id", None),
            (getattr(ingredient, "ingredient_id", None) or "").split(".")[-1],
        )
    )

    extensions = (".png", ".jpg", ".jpeg", ".gif")
    for base in candidates:
        for ext in extensions:
            path = INGREDIENT_ASSET_DIR / f"{base}{ext}"
            if path.exists():
                return path
    return None


def _load_ingredient_image(
    ingredient: Ingredient, *, target_px: int = INGREDIENT_IMAGE_TARGET_PX
) -> tk.PhotoImage:
    cache_key = f"ingredient:{getattr(ingredient, 'ingredient_id', ingredient.name)}:{target_px}"
    cached = _ingredient_image_cache.get(cache_key)
    if cached is not None:
        return cached

    image_path = _find_ingredient_image_path(ingredient)
    if image_path and image_path.exists():
        with Image.open(image_path) as source_image:
            working = source_image.convert("RGBA")
            working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
            working = working.copy()
    else:
        working = Image.new("RGBA", (target_px, target_px), (255, 255, 255, 255))

    image = ImageTk.PhotoImage(working)
    _ingredient_image_cache[cache_key] = image
    return image


def _load_rotten_image(target_px: int = INGREDIENT_IMAGE_TARGET_PX) -> tk.PhotoImage:
    cache_key = f"rotten:{target_px}"
    cached = _ingredient_image_cache.get(cache_key)
    if cached is not None:
        return cached

    rotten_path = INGREDIENT_ASSET_DIR / "rotten.png"
    if rotten_path.exists():
        with Image.open(rotten_path) as source_image:
            working = source_image.convert("RGBA")
            working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
            working = working.copy()
    else:
        working = Image.new("RGBA", (target_px, target_px), (96, 64, 64, 255))

    image = ImageTk.PhotoImage(working)
    _ingredient_image_cache[cache_key] = image
    return image


def _find_recipe_image_path(
    recipe_name: str, display_name: Optional[str] = None
) -> Optional[Path]:
    candidates = _candidate_image_basenames((recipe_name, display_name))
    extensions = (".png", ".jpg", ".jpeg", ".gif")
    directories: List[Tuple[Path, Dict[str, Path]]] = []
    for directory in _recipe_asset_directories():
        if not directory.exists():
            continue
        try:
            stem_map = {
                path.stem.lower(): path
                for path in directory.iterdir()
                if path.is_file()
            }
        except OSError:
            continue
        directories.append((directory, stem_map))

    for base in candidates:
        base_lower = base.lower()
        for directory, stem_map in directories:
            for ext in extensions:
                path = directory / f"{base}{ext}"
                if path.exists():
                    return path
            match = stem_map.get(base_lower)
            if match:
                return match
    return None


def _load_recipe_image(
    recipe_name: str,
    display_name: Optional[str] = None,
    *,
    target_px: int = RECIPE_IMAGE_TARGET_PX,
) -> Optional[tk.PhotoImage]:
    if not recipe_name:
        return None

    cache_key = f"recipe:{recipe_name}:{target_px}"
    if cache_key in _recipe_image_cache:
        return _recipe_image_cache[cache_key]

    image_path = _find_recipe_image_path(recipe_name, display_name)
    if not image_path or not image_path.exists():
        fallback = RECIPE_ASSET_DIR / "emptydish.png"
        image_path = fallback if fallback.exists() else None

    if image_path and image_path.exists():
        try:
            with Image.open(image_path) as source_image:
                working = source_image.convert("RGBA")
                working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
                working = working.copy()
        except OSError:
            _recipe_image_cache[cache_key] = None
            return None
        image = ImageTk.PhotoImage(working)
        _recipe_image_cache[cache_key] = image
        return image

    _recipe_image_cache[cache_key] = None
    return None


def _load_seasoning_icon(
    seasoning: Optional[Seasoning], *, target_px: int = 80
) -> tk.PhotoImage:
    if seasoning is None:
        cache_key = f"seasoning:__blank__:{target_px}"
        cached = _seasoning_icon_cache.get(cache_key)
        if cached is not None:
            return cached
        working = Image.new("RGBA", (target_px, target_px), (255, 255, 255, 255))
        image = ImageTk.PhotoImage(working)
        _seasoning_icon_cache[cache_key] = image
        return image

    name = getattr(seasoning, "seasoning_id", None) or seasoning.name
    cache_key = f"seasoning:{name}:{target_px}"
    cached = _seasoning_icon_cache.get(cache_key)
    if cached is not None:
        return cached

    image_path = _find_ingredient_image_path(seasoning)  # type: ignore[arg-type]
    if image_path and image_path.exists():
        with Image.open(image_path) as source_image:
            working = source_image.convert("RGBA")
            working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
            working = working.copy()
    else:
        working = Image.new("RGBA", (target_px, target_px), (255, 255, 255, 255))

    image = ImageTk.PhotoImage(working)
    _seasoning_icon_cache[cache_key] = image
    return image


def _load_chef_icon(*, target_px: int = 80) -> tk.PhotoImage:
    cache_key = f"chef:generic:{target_px}"
    cached = _chef_icon_cache.get(cache_key)
    if cached is not None:
        return cached

    image = _load_button_image("chefs.png", target_px=target_px)
    if image is None:
        working = Image.new("RGBA", (target_px, target_px), (255, 255, 255, 255))
        image = ImageTk.PhotoImage(working)
    _chef_icon_cache[cache_key] = image
    return image


def _load_button_image(filename: str, *, target_px: int = 88) -> Optional[tk.PhotoImage]:
    cache_key = f"button_image:{filename}:{target_px}"
    cached = _button_icon_cache.get(cache_key)
    if cached is not None:
        return cached

    icon_path = ICON_ASSET_DIR / filename
    if not icon_path.exists():
        return None

    with Image.open(icon_path) as source_image:
        working = source_image.convert("RGBA")
        working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
        working = working.copy()

    image = ImageTk.PhotoImage(working)
    _button_icon_cache[cache_key] = image
    return image


def _basket_icon_path(basket_name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", basket_name.lower()).strip("_")
    if not slug:
        slug = "basic"
    filename = f"basket_{slug}.png"
    image_path = BASKET_ART_DIR / filename
    if not image_path.exists():
        image_path = BASKET_ART_DIR / "basket_basic.png"
    return image_path


def _load_challenge_tile_image(
    basket_name: str, *, target_px: int = 160
) -> Optional[tk.PhotoImage]:
    slug = re.sub(r"[^a-z0-9]+", "_", basket_name.lower()).strip("_") or "basic"
    cache_key = (slug, target_px)
    cached = _challenge_tile_image_cache.get(cache_key)
    if cached is not None:
        return cached

    if Image is None:
        _challenge_tile_image_cache[cache_key] = None
        return None

    image_path = _basket_icon_path(basket_name)
    if not image_path.exists():
        _challenge_tile_image_cache[cache_key] = None
        return None

    with Image.open(image_path) as source_image:
        working = source_image.convert("RGBA")
        working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
        working = working.copy()

    image = ImageTk.PhotoImage(working)
    _challenge_tile_image_cache[cache_key] = image
    return image


def _generate_button_icon(
    key: str, text: str, *, size: int = 88, bg: str = "#f0f0f0", fg: str = "#1c1c1c"
) -> tk.PhotoImage:
    cache_key = f"button:{key}:{size}:{bg}:{fg}:{text}"
    cached = _button_icon_cache.get(cache_key)
    if cached is not None:
        return cached

    base = Image.new("RGBA", (32, 32), bg)
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (base.width - text_width) / 2
    text_y = (base.height - text_height) / 2
    draw.rectangle([(0, 0), (base.width - 1, base.height - 1)], outline=fg, width=1)
    draw.text((text_x, text_y), text, font=font, fill=fg)
    working = base.resize((size, size), Image.NEAREST)
    image = ImageTk.PhotoImage(working)
    _button_icon_cache[cache_key] = image
    return image


def _load_cookbook_indicator(
    state: RecipeIconState, *, target_px: int = COOKBOOK_ICON_TARGET_PX
) -> Optional[tk.PhotoImage]:
    cache_key = f"cookbook_indicator:{state}:{target_px}"
    if cache_key in _cookbook_indicator_cache:
        return _cookbook_indicator_cache[cache_key]

    icon_path = ICON_ASSET_DIR / "cookbook.png"
    if not icon_path.exists():
        _cookbook_indicator_cache[cache_key] = None
        return None

    with Image.open(icon_path) as source_image:
        working = source_image.convert("RGBA")
        max_side = max(working.size)
        if max_side > target_px:
            scale = target_px / max_side
            new_size = (
                max(1, int(round(working.width * scale))),
                max(1, int(round(working.height * scale))),
            )
            working = working.resize(new_size, RESAMPLE_LANCZOS)
        else:
            working = working.copy()

        if state == "blocked":
            working = working.convert("LA").convert("RGBA")

    image = ImageTk.PhotoImage(working)
    _cookbook_indicator_cache[cache_key] = image
    return image


FAMILY_EXAMPLE_ORDER = ["Protein", "Vegetable", "Grain", "Dairy", "Fruit"]
TASTE_EXAMPLE_ORDER = ["Sweet", "Salty", "Sour", "Umami", "Bitter"]


def _cycle_example(values: Sequence[str], length: int) -> List[str]:
    if not values:
        return [""] * length
    return [values[index % len(values)] for index in range(length)]


def _family_example(pattern: str, length: int) -> List[str]:
    if length <= 0:
        return []
    if pattern == "all_same":
        return [FAMILY_EXAMPLE_ORDER[0]] * length
    if pattern == "all_different":
        return _cycle_example(FAMILY_EXAMPLE_ORDER, length)
    if pattern == "mixed":
        base = [FAMILY_EXAMPLE_ORDER[0], FAMILY_EXAMPLE_ORDER[0]]
        remainder = max(0, length - len(base))
        return base + _cycle_example(FAMILY_EXAMPLE_ORDER[1:], remainder)
    return _cycle_example(FAMILY_EXAMPLE_ORDER, length)


def _flavor_example(pattern: str, length: int) -> List[str]:
    if length <= 0:
        return []
    if pattern == "all_same":
        return [TASTE_EXAMPLE_ORDER[0]] * length
    if pattern == "all_different":
        return _cycle_example(TASTE_EXAMPLE_ORDER, length)
    if pattern == "mixed":
        base = [TASTE_EXAMPLE_ORDER[0], TASTE_EXAMPLE_ORDER[0]]
        remainder = max(0, length - len(base))
        return base + _cycle_example(TASTE_EXAMPLE_ORDER[1:], remainder)
    return _cycle_example(TASTE_EXAMPLE_ORDER, length)


def _pattern_explanation(dimension: str, pattern: str) -> str:
    if dimension == "family":
        mapping = {
            "all_same": "All ingredients come from the same family.",
            "all_different": "Each ingredient uses a different family.",
            "mixed": "A dominant family supported by contrasting partners.",
        }
    else:
        mapping = {
            "all_same": "Every ingredient shares the same taste.",
            "all_different": "Each ingredient highlights a unique taste.",
            "mixed": "Repeating tastes blended with contrasting accents.",
        }
    return mapping.get(pattern, pattern.replace("_", " ").title())

DEFAULT_CONFIG = SimulationConfig()
DEFAULT_DECK_SIZE = DEFAULT_CONFIG.deck_size
DEFAULT_BIAS = DEFAULT_CONFIG.bias
DEFAULT_MAX_CHEFS = DEFAULT_CONFIG.active_chefs


def _load_game_data() -> GameData:
    """Load the shared JSON data using paths relative to this file."""

    return GameData.from_json(
        ingredients_path=str(ASSET_DIR / DEFAULT_INGREDIENTS_JSON),
        recipes_path=str(ASSET_DIR / DEFAULT_RECIPES_JSON),
        chefs_path=str(ASSET_DIR / DEFAULT_CHEFS_JSON),
        seasonings_path=str(ASSET_DIR / DEFAULT_SEASONINGS_JSON),
        taste_path=str(ASSET_DIR / DEFAULT_TASTE_JSON),
        baskets_path=str(ASSET_DIR / DEFAULT_BASKETS_JSON),
        dish_matrix_path=str(ASSET_DIR / DEFAULT_DISH_MATRIX_JSON),
    )


DATA = _load_game_data()


def format_multiplier(multiplier: float) -> str:
    quantized = quantize_multiplier(multiplier)
    if quantized.is_integer():
        return f"x{int(quantized)}"
    return f"x{quantized:.1f}"


class DishMatrixDialog(tk.Toplevel):
    def __init__(
        self,
        master: tk.Widget,
        entries: Sequence[DishMatrixEntry],
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master)
        self.title("Dish Matrix Reference")
        self.geometry("600x560")
        self.minsize(500, 440)
        self.entries: List[DishMatrixEntry] = list(entries)
        self._on_close = on_close
        self._icon_refs: List[tk.PhotoImage] = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        intro = ttk.Label(
            container,
            text=(
                "Match the icon patterns to trigger dish multipliers. "
                "Icons show example families and tastes—any ingredients that "
                "fit the pattern will work."
            ),
            style="Info.TLabel",
            wraplength=520,
            justify="left",
        )
        intro.grid(row=0, column=0, columnspan=2, sticky="w")

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview
        )
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.entries_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.entries_frame, anchor="nw")
        self.entries_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Escape>", lambda _e: self._handle_close())

        self.set_entries(self.entries)

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        self.destroy()

    def set_entries(self, entries: Sequence[DishMatrixEntry]) -> None:
        for child in self.entries_frame.winfo_children():
            child.destroy()
        self.entries = list(sorted(entries, key=lambda item: item.chance, reverse=True))
        self._icon_refs.clear()

        if not self.entries:
            ttk.Label(
                self.entries_frame,
                text="No dish matrix entries available.",
                style="Info.TLabel",
                wraplength=500,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        row = 0
        for index, entry in enumerate(self.entries):
            self._build_entry(entry, row)
            row += 1
            if index < len(self.entries) - 1:
                separator = ttk.Separator(self.entries_frame, orient="horizontal")
                separator.grid(row=row, column=0, sticky="ew", pady=(6, 10))
                row += 1

    def _build_entry(self, entry: DishMatrixEntry, row: int) -> None:
        frame = ttk.Frame(self.entries_frame, padding=(0, 2))
        frame.grid(row=row, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text=entry.name, style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text=format_multiplier(entry.multiplier),
            style="Summary.TLabel",
            anchor="center",
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))

        count_text = (
            f"Requires {entry.min_ingredients}-{entry.max_ingredients} ingredients"
            if entry.min_ingredients != entry.max_ingredients
            else f"Requires {entry.min_ingredients} ingredients"
        )
        ttk.Label(
            frame,
            text=(
                f"{entry.tier} tier • {entry.chance:.2%} chance • {count_text}"
            ),
            style="Info.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Label(
            frame,
            text=entry.description,
            style="Info.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        self._build_pattern_row(
            frame,
            "Families",
            entry.family_pattern,
            _family_example(entry.family_pattern, entry.min_ingredients),
            "family",
            row=3,
        )
        self._build_pattern_row(
            frame,
            "Tastes",
            entry.flavor_pattern,
            _flavor_example(entry.flavor_pattern, entry.min_ingredients),
            "taste",
            row=4,
        )

    def _build_pattern_row(
        self,
        parent: ttk.Frame,
        title: str,
        pattern: str,
        examples: Sequence[str],
        category: str,
        row: int,
    ) -> None:
        wrapper = ttk.Frame(parent)
        wrapper.grid(row=row, column=0, sticky="w", pady=(6, 0))
        dimension = "family" if category == "family" else "taste"
        explanation = _pattern_explanation(dimension, pattern)
        ttk.Label(
            wrapper,
            text=f"{title}: {explanation}",
            style="Info.TLabel",
            wraplength=380,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        icons_frame = ttk.Frame(wrapper)
        icons_frame.grid(row=0, column=1, sticky="w", padx=(8, 0))
        if not examples:
            ttk.Label(icons_frame, text="—", style="Info.TLabel").pack(side="left")
            return

        for name in examples:
            icon = _load_icon(category, name, target_px=DIALOG_ICON_TARGET_PX)
            if icon:
                label = ttk.Label(icons_frame, image=icon)
                label.image = icon
                label.pack(side="left", padx=1)
                self._icon_refs.append(icon)
            else:
                fallback = ttk.Label(
                    icons_frame,
                    text=name[:3],
                    style="Info.TLabel",
                    width=4,
                    anchor="center",
                )
                fallback.pack(side="left", padx=1)

@dataclass(frozen=True)
class SeasoningUsage:
    seasoning: Seasoning
    count: int


@dataclass(frozen=True)
class SeasoningCalculation:
    base_score: int
    total_boost_pct: float
    total_penalty: float
    seasoned_score: int
    ruined: bool
    usage: Tuple[SeasoningUsage, ...] = ()


@dataclass
class RoundStats:
    dishes_cooked: int = 0
    rotten_ingredients: int = 0
    recipes_completed: int = 0
    points_earned: int = 0
    _unique_recipes: set[str] = field(default_factory=set, init=False, repr=False)
    _ingredient_usage: Counter[str] = field(
        default_factory=Counter, init=False, repr=False
    )
    _ingredient_labels: Dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _recipe_usage: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _recipe_labels: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _dish_usage: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _dish_details: Dict[str, Dict[str, str]] = field(
        default_factory=dict, init=False, repr=False
    )

    def record_turn(
        self,
        *,
        score: int,
        rotten_count: int,
        recipe_name: Optional[str],
        ingredients: Sequence[Ingredient],
        dish_name: Optional[str],
        dish_tier: Optional[str],
        recipe_display_name: Optional[str] = None,
        family_label: str = "",
        flavor_label: str = "",
    ) -> None:
        self.dishes_cooked += 1
        self.points_earned += score
        self.rotten_ingredients += rotten_count
        if recipe_name:
            self.recipes_completed += 1
            self._unique_recipes.add(recipe_name)
            self._recipe_usage[recipe_name] += 1
            if recipe_display_name:
                self._recipe_labels.setdefault(recipe_name, recipe_display_name)

        for ingredient in ingredients:
            identifier = getattr(ingredient, "ingredient_id", "") or ingredient.name
            display = getattr(ingredient, "display_name", "") or ingredient.name
            self._ingredient_usage[identifier] += 1
            self._ingredient_labels.setdefault(identifier, display)

        if dish_name:
            label = dish_name
            if dish_tier:
                label = f"{dish_name} ({dish_tier})"
        else:
            label = "Freestyle Dish"
        self._dish_usage[label] += 1
        self._dish_details.setdefault(
            label,
            {
                "family_label": family_label,
                "flavor_label": flavor_label,
                "tier": dish_tier or "",
                "name": dish_name or label,
            },
        )

    def summary_payload(self) -> dict[str, object]:
        payload: Dict[str, object] = {
            "dishes_cooked": self.dishes_cooked,
            "rotten_ingredients": self.rotten_ingredients,
            "recipes_completed": self.recipes_completed,
            "unique_recipes": len(self._unique_recipes),
            "round_points": self.points_earned,
        }

        payload["ingredient_usage"] = [
            {
                "id": identifier,
                "name": self._ingredient_labels.get(identifier, identifier),
                "count": count,
            }
            for identifier, count in self._ingredient_usage.most_common()
        ]
        payload["recipe_usage"] = [
            {
                "name": name,
                "display": self._recipe_labels.get(name, name),
                "count": count,
            }
            for name, count in self._recipe_usage.most_common()
        ]
        payload["dish_usage"] = [
            {
                "label": label,
                "count": count,
                "family_label": self._dish_details.get(label, {}).get(
                    "family_label", ""
                ),
                "flavor_label": self._dish_details.get(label, {}).get(
                    "flavor_label", ""
                ),
                "tier": self._dish_details.get(label, {}).get("tier", ""),
            }
            for label, count in self._dish_usage.most_common()
        ]
        return payload


@dataclass
class TurnOutcome:
    selected: Sequence[Ingredient]
    Value: int
    dish_value: float
    dish_multiplier: float
    dish_name: Optional[str]
    dish_tier: Optional[str]
    family_label: str
    flavor_label: str
    family_pattern: str
    flavor_pattern: str
    recipe_name: Optional[str]
    recipe_display_name: Optional[str]
    recipe_multiplier: float
    final_score: int
    times_cooked_total: int
    base_score: int
    chef_hits: int
    round_index: int
    turn_number: int
    deck_refreshed: bool
    discovered_recipe: bool
    personal_discovery: bool
    alerts: Tuple[str, ...] = ()
    seasoning_boost_pct: float = 0.0
    seasoning_penalty: float = 0.0
    seasoned_score: int = 0
    ruined: bool = False
    applied_seasonings: Tuple[Tuple[str, int], ...] = ()


@dataclass
class CookbookEntry:
    """Track a discovered recipe and how often it has been cooked."""

    ingredients: Tuple[str, ...]
    display_name: str = ""
    count: int = 0
    multiplier: float = 1.0
    personal_discovery: bool = False

    def clone(self) -> "CookbookEntry":
        return CookbookEntry(
            self.ingredients,
            self.display_name,
            self.count,
            self.multiplier,
            self.personal_discovery,
        )


class DishMatrixTile(ttk.Frame):
    def __init__(
        self, master: tk.Widget, entries: Sequence[DishMatrixEntry]
    ) -> None:
        super().__init__(master, style="Tile.TFrame", padding=(14, 12))
        self.entries = list(sorted(entries, key=lambda item: item.chance, reverse=True))
        self.dialog: Optional[DishMatrixDialog] = None

        self.columnconfigure(0, weight=1)

        ttk.Label(
            self,
            text="🍽️ Dish Matrix",
            style="TileHeader.TLabel",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        self.subtitle_var = tk.StringVar(
            value="Open the reference to see icon-based dish multipliers."
        )
        ttk.Label(
            self,
            textvariable=self.subtitle_var,
            style="TileSub.TLabel",
            wraplength=260,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))

        ttk.Label(
            self,
            text="Use the resource bar to open the pop-up guide and match families with tastes for bonuses.",
            style="TileInfo.TLabel",
            wraplength=260,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

    def open_dialog(self) -> None:
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.lift()
            self.dialog.focus_force()
            return

        def _on_close() -> None:
            self.dialog = None

        self.dialog = DishMatrixDialog(self, self.entries, on_close=_on_close)
        toplevel = self.winfo_toplevel()
        if toplevel:
            self.dialog.transient(toplevel)
        self.dialog.focus_force()

    def set_entries(self, entries: Sequence[DishMatrixEntry]) -> None:
        self.entries = list(sorted(entries, key=lambda item: item.chance, reverse=True))
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.set_entries(self.entries)
        else:
            self.dialog = None


class InvalidDishSelection(Exception):
    """Raised when a chosen ingredient combination fails to produce any dish."""

    def __init__(
        self,
        message: str,
        ingredient_names: Sequence[str],
        *,
        primary_ingredient: Optional[Ingredient] = None,
        ingredients: Optional[Sequence[Ingredient]] = None,
    ) -> None:
        super().__init__(message)
        self.ingredient_names = tuple(ingredient_names)
        self.primary_ingredient = primary_ingredient
        self.ingredients = tuple(ingredients) if ingredients is not None else ()


class GameSession:
    """Manage deck, hand, and scoring for a single run."""

    def __init__(
        self,
        data: GameData,
        basket_name: str,
        chefs: Sequence[Chef],
        hand_size: int,
        pick_size: int,
        rounds: Optional[int] = None,
        deck_size: int = DEFAULT_DECK_SIZE,
        bias: float = DEFAULT_BIAS,
        max_chefs: int = DEFAULT_MAX_CHEFS,
        challenge: Optional[BasketChallenge] = None,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        pantry_card_ids: Optional[Sequence[str]] = None,
        starting_cookbook: Optional[Mapping[str, "CookbookEntry"]] = None,
        starting_seasonings: Optional[Sequence[Seasoning]] = None,
    ) -> None:
        if rounds is not None and rounds <= 0:
            raise ValueError("rounds must be positive when specified")
        if hand_size <= 0:
            raise ValueError("hand_size must be positive")
        if pick_size <= 0:
            raise ValueError("pick_size must be positive")
        if pick_size > hand_size:
            raise ValueError("pick_size cannot exceed hand_size")
        if max_chefs <= 0:
            raise ValueError("max_chefs must be positive")
        if len(chefs) > max_chefs:
            raise ValueError("Initial chef roster exceeds the configured maximum")

        self.data = data
        self.basket_name = basket_name
        self.chefs = list(chefs)
        self.rounds = rounds if rounds is not None else 0
        self.hand_size = hand_size
        self.pick_size = pick_size
        self.deck_size = deck_size
        self.bias = bias
        self.max_chefs = max_chefs
        self.challenge: Optional[BasketChallenge] = challenge
        self.challenge_target: Optional[int] = (
            int(challenge.target_score) if challenge else None
        )
        self.challenge_reward: Optional[Mapping[str, str]] = (
            dict(challenge.reward) if challenge else None
        )
        self.challenge_reward_claimed = False
        if rng is not None:
            self.rng = rng
            self.seed = seed
        else:
            resolved_seed, resolved_rng = resolve_seed(seed)
            self.rng = resolved_rng
            self.seed = resolved_seed

        self.turn_number = 0
        self.round_index = 0
        self.total_score = 0
        self.finished = False
        self.pending_new_chef_offer = False
        self._post_run_reward_pending = False
        self._round_score = 0
        self._round_stats = RoundStats()
        self._awaiting_basket_reset = False
        self._basket_bonus_choices: List[Ingredient] = []
        self._basket_clear_summary: Optional[Dict[str, object]] = None
        self._permanent_bonus_ingredients: List[Ingredient] = []
        self._starting_pantry_cards: Optional[List[Ingredient]] = None
        if pantry_card_ids:
            self._starting_pantry_cards = self.data.ingredients_for_ids(pantry_card_ids)

        self.hand: List[IngredientCard] = []
        self.deck: List[IngredientCard] = []
        self.seasonings: List[Seasoning] = []
        self._seasoning_charges: Dict[str, Optional[int]] = {}
        self._events: List[str] = []
        self._carryover_cards: List[IngredientCard] = []
        self._cleanup_rotten_cards: List[IngredientCard] = []
        self._cleanup_acknowledged = True

        if self.challenge:
            reward_text = format_challenge_reward_text(self.challenge.reward)
            unique_count, total_count = challenge_ingredient_counts(
                self.data, self.challenge
            )
            ingredient_text = format_challenge_ingredient_text(
                unique_count, total_count
            )
            if ingredient_text != "No listed ingredients":
                ingredient_sentence = f"This basket adds {ingredient_text}."
            else:
                ingredient_sentence = "This basket has no listed ingredients."

            self._push_event(
                "Basket challenge accepted: "
                f"Reach {self.challenge.target_score} points to earn {reward_text}. "
                f"{ingredient_sentence}"
            )

        self._cookbook_ingredients: set[str] = set()
        self._ingredient_recipe_map = {
            name: tuple(recipes)
            for name, recipes in self.data.ingredient_recipes.items()
        }

        if starting_seasonings:
            for seasoning in starting_seasonings:
                if not isinstance(seasoning, Seasoning):
                    continue
                if any(existing.seasoning_id == seasoning.seasoning_id for existing in self.seasonings):
                    continue
                self.seasonings.append(seasoning)
                self._seasoning_charges[seasoning.seasoning_id] = seasoning.charges

        self._refresh_chef_data()
        self.cookbook: Dict[str, CookbookEntry] = {}
        if starting_cookbook:
            for recipe_name, entry in starting_cookbook.items():
                if not isinstance(entry, CookbookEntry):
                    continue
                clone = entry.clone()
                self.cookbook[recipe_name] = clone
                self._cookbook_ingredients.update(clone.ingredients)
                chef_has_recipe = any(
                    recipe_name in chef.recipe_names for chef in self.chefs
                )
                display_times = clone.count
                if clone.personal_discovery and not chef_has_recipe:
                    display_times = max(display_times - 1, 0)
                clone.multiplier = self.data.recipe_multiplier(
                    recipe_name,
                    chefs=self.chefs,
                    times_cooked=display_times,
                )

        self._current_deck_total = 0
        self._start_next_round(initial=True)

    def _refresh_chef_data(self) -> None:
        self._chef_key_map = {
            chef.name: self.data.chef_key_ingredients(chef) for chef in self.chefs
        }
        self._chef_key_set = self.data.chefs_key_ingredients(self.chefs)

    # ----------------- Event helpers -----------------
    def _push_event(self, message: str) -> None:
        self._events.append(message)

    def consume_events(self) -> List[str]:
        events = list(self._events)
        self._events.clear()
        return events

    def _card_display_name(self, card: IngredientCard) -> str:
        display_name = getattr(card.ingredient, "display_name", None) or ""
        return display_name or card.ingredient.name

    def _card_rot_status(self, card: IngredientCard) -> str:
        rot_limit = getattr(card.ingredient, "rotten_turns", 0) or 0
        total = max(rot_limit, 0)

        if card.is_rotten:
            if total > 0:
                return f"rots {total}/{total} (rotten)"
            return "rotten (ruins dishes)"
        if total <= 0:
            return "rots immediately"
        if total >= 12 or rot_limit >= 900:
            return "shelf life stable"

        progress = min(max(card.turns_in_hand, 0), total)
        return f"rots {progress}/{total}"

    def _announce_card_added(
        self, card: IngredientCard, *, message_prefix: Optional[str] = None
    ) -> None:
        name = self._card_display_name(card)
        status = self._card_rot_status(card)
        if message_prefix:
            message = f"{message_prefix} - {status}."
        else:
            message = f"{name} joins your hand - {status}."
        self._push_event(message)

    def _log_hand_snapshot(
        self, title: str, *, new_cards: Optional[Sequence[IngredientCard]] = None
    ) -> None:
        if not self.hand:
            return
        new_ids: set[int] = set()
        if new_cards:
            new_ids = {id(card) for card in new_cards}
        descriptions = []
        for card in self.hand:
            name = self._card_display_name(card)
            if id(card) in new_ids:
                name = f"(new) {name}"
            descriptions.append(f"{name} - {self._card_rot_status(card)}")
        description_text = "; ".join(descriptions)
        self._push_event(f"{title}: {description_text}")

    # ----------------- Round & hand management -----------------
    def _start_next_round(self, initial: bool = False) -> None:
        if initial:
            self.round_index = 1
        else:
            self.round_index += 1

        self._round_score = 0
        self._round_stats = RoundStats()
        self._awaiting_basket_reset = False
        self._basket_bonus_choices = []
        self._basket_clear_summary = None
        self.finished = False
        self._post_run_reward_pending = False
        carryover_cards: List[IngredientCard] = []
        if not initial:
            if self.needs_cleanup_confirmation():
                raise RuntimeError(
                    "Cannot start the next round before cleaning rotten ingredients."
                )
            if self._carryover_cards:
                carryover_cards = list(self._carryover_cards)
            else:
                carryover_cards = self._collect_pantry_carryover()
            self._carryover_cards.clear()
        else:
            self._carryover_cards.clear()
            self._cleanup_rotten_cards.clear()
            self._cleanup_acknowledged = True
        self.deck = self._build_market_deck(
            carryover_cards=carryover_cards,
            use_starting_pantry=initial,
        )
        self._current_deck_total = len(self.deck)
        self.hand.clear()
        self.pending_new_chef_offer = False
        self._push_event(
            f"Round {self.round_index} begins. Pantry shuffled for the team."
        )
        self._refill_hand(log_new_cards=not initial)
        if initial:
            self._log_hand_snapshot("Opening hand")

    def _collect_pantry_carryover(self) -> List[IngredientCard]:
        carryover: List[IngredientCard] = []
        if self.hand:
            carryover.extend(self.hand)
            self.hand.clear()
        if self.deck:
            carryover.extend(self.deck)
            self.deck.clear()
        return carryover

    def _rebalance_deck(self, cards: Sequence[IngredientCard]) -> List[IngredientCard]:
        if not cards:
            return []

        return distribute_unique_draws(
            list(cards),
            key_func=lambda card: card.ingredient.name,
            rng=self.rng,
        )

    def _build_market_deck(
        self,
        *,
        carryover_cards: Optional[Sequence[IngredientCard]] = None,
        use_starting_pantry: Optional[bool] = None,
    ) -> List[IngredientCard]:
        cards: List[IngredientCard] = []
        if carryover_cards:
            cards.extend(carryover_cards)

        if use_starting_pantry is None:
            use_saved_pantry = self._starting_pantry_cards is not None
        else:
            use_saved_pantry = (
                self._starting_pantry_cards is not None and use_starting_pantry
            )
        if use_saved_pantry:
            base_deck = list(self._starting_pantry_cards or [])
        else:
            base_deck = build_market_deck(
                self.data,
                self.basket_name,
                self.chefs,
                deck_size=self.deck_size,
                bias=self.bias,
                rng=self.rng,
            )

        cards.extend(IngredientCard(ingredient=item) for item in base_deck)
        if self._permanent_bonus_ingredients:
            cards.extend(
                IngredientCard(ingredient=ingredient)
                for ingredient in self._permanent_bonus_ingredients
            )
        return self._rebalance_deck(cards)

    def _choose_bonus_ingredients(self, count: int = 3) -> List[Ingredient]:
        pool: List[Ingredient] = []
        seen: set[str] = set()
        for ingredient in self.data.ingredients.values():
            if ingredient.name in seen:
                continue
            pool.append(ingredient)
            seen.add(ingredient.name)
        if len(pool) <= count:
            return list(pool)
        return self.rng.sample(pool, count)

    def _refill_hand(self, *, log_new_cards: bool = True) -> bool:
        needed = self.hand_size - len(self.hand)
        deck_refreshed = False
        new_cards: List[IngredientCard] = []
        skipped_duplicates: List[IngredientCard] = []
        seen_names = {card.ingredient.name for card in self.hand}
        while needed > 0 and not self.finished:
            if not self.deck:
                break
            drawn = self.deck.pop()
            ingredient_name = drawn.ingredient.name
            if ingredient_name in seen_names:
                skipped_duplicates.append(drawn)
                continue
            self.hand.append(drawn)
            seen_names.add(ingredient_name)
            if log_new_cards:
                new_cards.append(drawn)
            needed -= 1
        if skipped_duplicates:
            # Reinsert skipped cards at the bottom of the pantry so they remain
            # available for future turns once the hand composition changes.
            self.deck[0:0] = list(reversed(skipped_duplicates))
        if len(self.hand) == 0 and not self.deck and not self._awaiting_basket_reset:
            self._enter_round_summary()
        if log_new_cards and new_cards:
            self._log_hand_snapshot("Update Hand", new_cards=new_cards)
        return deck_refreshed

    def _enter_round_summary(self, *, reason: str = "pantry_empty") -> None:
        if self._awaiting_basket_reset and reason != "target_reached":
            return
        if self.finished and reason != "target_reached":
            return

        summary = {
            "round_index": self.round_index,
            "total_rounds": self.rounds or self.round_index,
            "round_score": self._round_score,
            "total_score": self.total_score,
            "round_end_reason": reason,
        }
        summary.update(self._round_stats.summary_payload())
        self._basket_clear_summary = summary
        if reason == "target_reached":
            self.finished = True
            self._awaiting_basket_reset = False
            self._basket_bonus_choices = []
            self.pending_new_chef_offer = False
            self._post_run_reward_pending = True
            summary["run_finished"] = True
            summary["bonus_choices_available"] = False
            self._push_event("Round complete! Basket target score reached.")
            return

        collected_cards = self._collect_pantry_carryover()
        if collected_cards:
            rotten_cards = [card for card in collected_cards if card.is_rotten]
            self._carryover_cards = [card for card in collected_cards if not card.is_rotten]
        else:
            rotten_cards = []
            self._carryover_cards = []
        self._cleanup_rotten_cards = rotten_cards
        self._cleanup_acknowledged = not bool(rotten_cards)

        self._awaiting_basket_reset = True
        self._basket_bonus_choices = self._choose_bonus_ingredients()
        self.pending_new_chef_offer = False
        summary["run_finished"] = False
        summary["bonus_choices_available"] = bool(self._basket_bonus_choices)
        if self._basket_bonus_choices:
            self._push_event(
                "Round complete! Choose an ingredient to prepare for the next round."
            )
        else:
            self._push_event(
                "Round complete! Draws will resume automatically when you continue."
            )

    def _rebuild_deck_for_new_chef(self) -> None:
        if self.finished:
            return
        self.deck = self._build_market_deck()
        self._current_deck_total = len(self.deck)
        self.hand.clear()
        self._push_event("Pantry refreshed to reflect your expanded chef lineup.")
        self._refill_hand()

    def _advance_decay(
        self, cards: Iterable[IngredientCard], *, record_events: bool = True
    ) -> List[IngredientCard]:
        newly_rotten: List[IngredientCard] = []
        for card in cards:
            if card.is_rotten:
                continue
            limit = max(card.ingredient.rotten_turns, 0)
            card.turns_in_hand += 1
            if limit and card.turns_in_hand >= limit:
                card.turns_in_hand = limit
                if not card.is_rotten:
                    card.is_rotten = True
                    newly_rotten.append(card)
                    if record_events:
                        name = getattr(card.ingredient, "display_name", None) or card.ingredient.name
                        self._push_event(
                            f"{name} has gone rotten and will ruin any dish it joins."
                        )
        return newly_rotten

    def _apply_end_turn_decay(self) -> None:
        if not self.hand:
            return

        self._advance_decay(self.hand, record_events=True)

    def _handle_invalid_selection(
        self,
        indices: Sequence[int],
        selected_cards: Sequence[IngredientCard],
        *,
        reason: Optional[str] = None,
        primary_ingredient: Optional[Ingredient] = None,
    ) -> InvalidDishSelection:
        display_names = [
            getattr(card.ingredient, "display_name", None) or card.ingredient.name
            for card in selected_cards
        ]
        combo_text = " + ".join(display_names) if display_names else "The selected cards"

        if primary_ingredient is None and selected_cards:
            primary_ingredient = selected_cards[0].ingredient

        # Remove the cards from the hand before shuffling them back into the deck.
        for offset, index in enumerate(indices):
            self.hand.pop(index - offset)

        newly_rotten = self._advance_decay(self.hand, record_events=False)

        self.deck.extend(selected_cards)
        if self.deck:
            self.deck = self._rebalance_deck(self.deck)

        self._push_event(
            f"{combo_text} didn't form a dish and returned to the pantry to be reshuffled."
        )

        if newly_rotten:
            rotten_names = ", ".join(
                getattr(card.ingredient, "display_name", None) or card.ingredient.name
                for card in newly_rotten
            )
            plural_word = "have" if len(newly_rotten) > 1 else "has"
            ruin_pronoun = "they" if len(newly_rotten) > 1 else "it"
            self._push_event(
                f"{rotten_names} {plural_word} now gone rotten and will ruin any dish {ruin_pronoun} joins."
            )

        self._refill_hand()

        plural = len(display_names) != 1
        verb = "do" if plural else "does"
        pronoun = "They" if plural else "It"
        return_verb = "return" if plural else "returns"
        message = (
            f"{combo_text} {verb} not form a dish. {pronoun} {return_verb} to the pantry to be "
            "reshuffled and will only rot while in your hand."
        )
        if reason == "all_same_ingredient" and primary_ingredient is not None:
            ingredient_name = (
                getattr(primary_ingredient, "display_name", None)
                or primary_ingredient.name
            )
            message += (
                f" All selected cards were {ingredient_name}, so there was nothing new to cook."
            )
        if newly_rotten:
            rotten_pronoun = "They have" if len(newly_rotten) > 1 else "It has"
            ruin_pronoun = "they" if len(newly_rotten) > 1 else "it"
            message += (
                f" {rotten_pronoun} now gone rotten and will ruin any dish {ruin_pronoun} joins."
            )

        return InvalidDishSelection(
            message,
            display_names,
            primary_ingredient=primary_ingredient,
            ingredients=[card.ingredient for card in selected_cards],
        )

    def _times_cooked(self, recipe_name: Optional[str]) -> int:
        if not recipe_name:
            return 0
        entry = self.cookbook.get(recipe_name)
        return entry.count if entry else 0

    # ----------------- Public API -----------------
    def get_hand(self) -> Sequence[IngredientCard]:
        return list(self.hand)

    def get_remaining_deck(self) -> Sequence[Ingredient]:
        return [card.ingredient for card in self.deck]

    def get_pantry_card_ids(self) -> List[str]:
        ids: List[str] = []

        def append_id(ingredient: Ingredient) -> None:
            ingredient_id = getattr(ingredient, "ingredient_id", "")
            if ingredient_id:
                ids.append(ingredient_id)
                return
            lookup = self.data.ingredients.get(getattr(ingredient, "name", ""))
            if lookup:
                fallback_id = getattr(lookup, "ingredient_id", "")
                if fallback_id:
                    ids.append(fallback_id)

        for card in self.get_hand():
            append_id(card.ingredient)
        for ingredient in self.get_remaining_deck():
            append_id(ingredient)

        return ids

    def get_basket_counts(self) -> Tuple[int, int]:
        return len(self.deck), self._current_deck_total

    def awaiting_new_round(self) -> bool:
        return self._awaiting_basket_reset and not self.finished

    def get_basket_bonus_choices(self) -> Sequence[Ingredient]:
        return list(self._basket_bonus_choices)

    def peek_basket_clear_summary(self) -> Optional[Dict[str, object]]:
        if self._basket_clear_summary is None:
            return None
        return dict(self._basket_clear_summary)

    def pending_cleanup_ingredients(self) -> Sequence[Ingredient]:
        return [card.ingredient for card in self._cleanup_rotten_cards]

    def needs_cleanup_confirmation(self) -> bool:
        return bool(self._cleanup_rotten_cards) and not self._cleanup_acknowledged

    def acknowledge_cleanup(self) -> Sequence[Ingredient]:
        if not self._awaiting_basket_reset:
            raise RuntimeError("No pantry cleanup is pending.")
        if not self._cleanup_rotten_cards:
            self._cleanup_acknowledged = True
            self._push_event("Pantry cleanup confirmed: no rotten ingredients to remove.")
            return ()

        removed_cards = tuple(card.ingredient for card in self._cleanup_rotten_cards)
        names = ", ".join(
            getattr(ingredient, "display_name", None) or ingredient.name
            for ingredient in removed_cards
        )
        plural = "ingredients" if len(removed_cards) != 1 else "ingredient"
        self._push_event(
            f"Removed rotten {plural} from the pantry: {names}."
        )
        self._cleanup_rotten_cards.clear()
        self._cleanup_acknowledged = True
        return removed_cards

    def begin_next_round_from_empty_basket(self, ingredient: Ingredient) -> None:
        if not self._awaiting_basket_reset:
            raise RuntimeError("No pantry refill is pending.")
        if not self._cleanup_acknowledged:
            raise RuntimeError(
                "Cannot start the next round until the pantry cleanup is confirmed."
            )

        self._awaiting_basket_reset = False
        self._basket_bonus_choices = []
        self._basket_clear_summary = None

        self._permanent_bonus_ingredients.append(ingredient)

        self._start_next_round()

        bonus_card: Optional[IngredientCard] = None
        match_id = getattr(ingredient, "ingredient_id", None)
        for index, card in enumerate(self.deck):
            candidate_id = getattr(card.ingredient, "ingredient_id", None)
            if match_id and candidate_id == match_id:
                bonus_card = self.deck.pop(index)
                break
            if card.ingredient.name == ingredient.name:
                bonus_card = self.deck.pop(index)
                break
        if bonus_card is None:
            bonus_card = IngredientCard(ingredient=ingredient)
        bonus_card.freshen()
        inserted_into_empty_hand = False
        if self.hand:
            replacement = self.hand[0]
            self.hand[0] = bonus_card
            replacement.freshen()
            self.deck.append(replacement)
            if self.deck:
                self.deck = self._rebalance_deck(self.deck)
        else:
            self.hand.append(bonus_card)
            inserted_into_empty_hand = True
        self._current_deck_total = len(self.deck)

        message_prefix = (
            f"{self._card_display_name(bonus_card)} joins your opening hand to celebrate the new round"
        )
        self._announce_card_added(bonus_card, message_prefix=message_prefix)
        if inserted_into_empty_hand:
            self._refill_hand()
            self._current_deck_total = len(self.deck)

    def begin_next_round_after_reward(self) -> None:
        if not self._awaiting_basket_reset:
            raise RuntimeError("No new round is pending.")
        if not self._cleanup_acknowledged:
            raise RuntimeError(
                "Cannot start the next round until the pantry cleanup is confirmed."
            )

        self._awaiting_basket_reset = False
        self._basket_bonus_choices = []
        self._basket_clear_summary = None

        self._start_next_round()

        self._refill_hand()
        self._current_deck_total = len(self.deck)

    def get_total_score(self) -> int:
        return self.total_score

    def get_cookbook(self) -> Dict[str, CookbookEntry]:
        return {name: entry.clone() for name, entry in self.cookbook.items()}

    def available_chefs(self) -> List[Chef]:
        if len(self.chefs) >= self.max_chefs:
            return []
        active_names = {chef.name for chef in self.chefs}
        return [chef for chef in self.data.chefs if chef.name not in active_names]

    def available_seasonings(self) -> List[Seasoning]:
        owned = {seasoning.name for seasoning in self.seasonings}
        return [
            seasoning
            for seasoning in self.data.seasonings
            if seasoning.name not in owned
        ]

    def can_recruit_chef(self) -> bool:
        return (
            not self.finished
            and self.pending_new_chef_offer
            and (bool(self.available_chefs()) or bool(self.available_seasonings()))
        )

    def add_chef(self, chef: Chef) -> None:
        if any(existing.name == chef.name for existing in self.chefs):
            raise ValueError(f"{chef.name} is already on your team.")
        if self.finished and not self._post_run_reward_pending:
            raise RuntimeError("Cannot add chefs after the run has finished.")
        if len(self.chefs) >= self.max_chefs:
            raise ValueError("Your chef roster is already at the maximum size.")
        self.chefs.append(chef)
        self._refresh_chef_data()
        self.pending_new_chef_offer = False
        self._push_event(
            f"{chef.name} joins the team! New key ingredients unlocked. "
            f"Roster {len(self.chefs)}/{self.max_chefs}."
        )
        if not self._awaiting_basket_reset:
            self._rebuild_deck_for_new_chef()

    def add_seasoning(self, seasoning: Seasoning) -> None:
        if any(existing.name == seasoning.name for existing in self.seasonings):
            raise ValueError(f"{seasoning.display_name or seasoning.name} is already in your pantry.")
        if self.finished and not self._post_run_reward_pending:
            raise RuntimeError("Cannot add seasonings after the run has finished.")
        self.seasonings.append(seasoning)
        self._seasoning_charges[seasoning.seasoning_id] = seasoning.charges
        self.pending_new_chef_offer = False
        display_name = seasoning.display_name or seasoning.name
        perk_text = seasoning.perk.strip()
        if perk_text:
            self._push_event(
                f"{display_name} added to your seasoning rack — {perk_text}"
            )
        else:
            self._push_event(f"{display_name} added to your seasoning rack.")

    def skip_chef_recruitment(self) -> None:
        if self.pending_new_chef_offer:
            self.pending_new_chef_offer = False
            self._push_event(
                "You continue without recruiting an additional chef or seasoning this round."
            )

    def random_available_chef(self) -> Optional[Chef]:
        available = self.available_chefs()
        if not available:
            return None
        return self.rng.choice(available)

    def random_available_seasoning(self) -> Optional[Seasoning]:
        available = self.available_seasonings()
        if not available:
            return None
        return self.rng.choice(available)

    def get_seasonings(self) -> Sequence[Seasoning]:
        return list(self.seasonings)

    def get_seasoning_charges(self, seasoning_id: str) -> Optional[int]:
        if seasoning_id in self._seasoning_charges:
            return self._seasoning_charges[seasoning_id]
        base = self.data.seasoning_by_id.get(seasoning_id)
        if base is None:
            return None
        self._seasoning_charges[seasoning_id] = base.charges
        return base.charges

    def get_active_seasonings(self) -> List[Tuple[Seasoning, Optional[int]]]:
        active: List[Tuple[Seasoning, Optional[int]]] = []
        for seasoning in self.seasonings:
            remaining = self.get_seasoning_charges(seasoning.seasoning_id)
            if remaining == 0:
                continue
            active.append((seasoning, remaining))
        return active

    def preview_recipe_multiplier(self, recipe_name: Optional[str]) -> float:
        return self.data.recipe_multiplier(
            recipe_name,
            chefs=self.chefs,
            times_cooked=self._times_cooked(recipe_name),
        )

    def _recipe_availability_state(
        self, ingredient: Ingredient
    ) -> Optional[RecipeIconState]:
        recipe_names = self._ingredient_recipe_map.get(ingredient.name, ())
        if not recipe_names:
            return None

        available_counts: Counter[str] = Counter(
            card.ingredient.name for card in self.hand
        )
        available_counts.update(card.ingredient.name for card in self.deck)

        for recipe_name in recipe_names:
            recipe = self.data.recipe_by_name.get(recipe_name)
            if not recipe:
                continue
            required = Counter(recipe.trio)
            if all(available_counts.get(name, 0) >= count for name, count in required.items()):
                return "available"

        return "blocked"

    def get_selection_markers(
        self, ingredient: Ingredient
    ) -> Tuple[List[str], Optional[RecipeIconState]]:
        markers = [
            chef.name
            for chef in self.chefs
            if ingredient.name in self._chef_key_map.get(chef.name, set())
        ]
        cookbook_state = self._recipe_availability_state(ingredient)
        return markers, cookbook_state

    def get_recipe_hints(self, ingredient: Ingredient) -> List[str]:
        hints: List[str] = []
        for recipe_name in self._ingredient_recipe_map.get(ingredient.name, ()): 
            display = self.data.recipe_display_name(recipe_name)
            hints.append(display or recipe_name)
        return sorted(hints, key=lambda value: value.lower())

    def _normalize_seasoning_usage(
        self,
        applied_seasonings: Union[Mapping[str, int], Sequence[Tuple[str, int]]],
    ) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        items: Iterable[Tuple[str, int]]
        if isinstance(applied_seasonings, Mapping):
            items = applied_seasonings.items()
        else:
            items = applied_seasonings
        for key, raw_count in items:
            if raw_count is None:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            identifier = str(key)
            usage[identifier] = usage.get(identifier, 0) + count
        return usage

    def calculate_seasoning_adjustments(
        self,
        ingredients: Sequence[Ingredient],
        base_score: float,
        applied_seasonings: Union[Mapping[str, int], Sequence[Tuple[str, int]]],
    ) -> SeasoningCalculation:
        usage_map = self._normalize_seasoning_usage(applied_seasonings)
        if not usage_map:
            rounded_base = int(round(base_score))
            return SeasoningCalculation(
                base_score=rounded_base,
                total_boost_pct=0.0,
                total_penalty=0.0,
                seasoned_score=max(0, rounded_base),
                ruined=rounded_base <= 0,
                usage=(),
            )

        owned_ids = {seasoning.seasoning_id for seasoning in self.seasonings}
        usages: List[SeasoningUsage] = []
        total_boost_pct = 0.0
        total_penalty = 0.0
        rounded_base = int(round(base_score))

        dish_tastes = Counter(ingredient.taste for ingredient in ingredients)
        dish_families = {ingredient.family for ingredient in ingredients}

        for seasoning_id, count in usage_map.items():
            if seasoning_id not in owned_ids:
                raise ValueError(f"Seasoning {seasoning_id} is not in the player's pantry.")
            seasoning = self.data.seasoning_by_id.get(seasoning_id)
            if not seasoning:
                raise ValueError(f"Unknown seasoning identifier: {seasoning_id}")
            stack_limit = max(1, seasoning.stack_limit)
            if count > stack_limit:
                display_name = seasoning.display_name or seasoning.name
                raise ValueError(
                    f"Cannot apply {display_name} more than {stack_limit} time(s) to one dish."
                )
            remaining = self.get_seasoning_charges(seasoning_id)
            if remaining is not None and count > remaining:
                display_name = seasoning.display_name or seasoning.name
                raise ValueError(
                    f"{display_name} only has {remaining} charge(s) remaining."
                )

            per_pct = 0.0
            for taste, boost in seasoning.boosts.items():
                if dish_tastes.get(taste, 0) > 0:
                    per_pct += float(boost)
            total_boost_pct += per_pct * count

            if seasoning.conflicts and seasoning.conflict_penalty > 0:
                matches = sum(1 for family in seasoning.conflicts if family in dish_families)
                if matches:
                    total_penalty += float(seasoning.conflict_penalty) * matches * count

            usages.append(SeasoningUsage(seasoning=seasoning, count=count))

        adjusted_value = math.floor(rounded_base * (1 + total_boost_pct) - total_penalty)
        if adjusted_value < 0:
            adjusted_value = 0
        ruined = adjusted_value <= 0

        return SeasoningCalculation(
            base_score=rounded_base,
            total_boost_pct=total_boost_pct,
            total_penalty=total_penalty,
            seasoned_score=adjusted_value,
            ruined=ruined,
            usage=tuple(usages),
        )

    def return_indices(self, indices: Sequence[int]) -> Tuple[List[Ingredient], bool]:
        if self.finished:
            raise RuntimeError("The session has already finished.")
        if self._awaiting_basket_reset:
            raise RuntimeError("Cannot return cards while waiting to start the next round.")
        if not indices:
            raise ValueError("You must select at least one card to return.")
        if len(indices) > self.pick_size:
            raise ValueError(
                f"You may return up to {self.pick_size} cards at a time."
            )

        unique = sorted(set(indices))
        if len(unique) != len(indices):
            raise ValueError("Selections contain duplicates.")

        if any(index < 0 or index >= len(self.hand) for index in unique):
            raise IndexError("Selection index out of range for the current hand.")

        removed_cards = [self.hand[index] for index in unique]
        if any(card.is_rotten for card in removed_cards):
            raise ValueError("Rotten ingredients cannot be returned to the pantry.")

        returned_cards: List[IngredientCard] = []
        for offset, index in enumerate(unique):
            returned_cards.append(self.hand.pop(index - offset))

        newly_rotten = self._advance_decay(self.hand, record_events=False)

        self.deck.extend(returned_cards)
        if self.deck:
            self.deck = self._rebalance_deck(self.deck)

        deck_refreshed = self._refill_hand()

        if returned_cards:
            if len(returned_cards) == 1:
                name = returned_cards[0].ingredient.name
                self._push_event(
                    f"Returned {name} to the pantry and drew a replacement."
                )
            else:
                names = ", ".join(card.ingredient.name for card in returned_cards)
                self._push_event(
                    f"Returned {names} to the pantry and drew replacements."
                )

        if newly_rotten:
            rotten_names = ", ".join(
                getattr(card.ingredient, "display_name", None) or card.ingredient.name
                for card in newly_rotten
            )
            plural_word = "have" if len(newly_rotten) > 1 else "has"
            ruin_pronoun = "they" if len(newly_rotten) > 1 else "it"
            self._push_event(
                f"{rotten_names} {plural_word} now gone rotten and will ruin any dish {ruin_pronoun} joins."
            )

        removed = [card.ingredient for card in returned_cards]
        return removed, deck_refreshed

    def discard_indices(self, indices: Sequence[int]) -> Tuple[List[Ingredient], bool]:
        """Backward-compatible alias for returning cards to the pantry."""

        return self.return_indices(indices)

    def play_turn(
        self,
        indices: Sequence[int],
        applied_seasonings: Optional[
            Union[Mapping[str, int], Sequence[Tuple[str, int]]]
        ] = None,
    ) -> TurnOutcome:
        if self.finished:
            raise RuntimeError("The session has already finished.")
        if self._awaiting_basket_reset:
            raise RuntimeError("Cannot cook until the next round begins.")
        if not indices:
            raise ValueError("You must select at least one card to cook.")
        if len(indices) > self.pick_size:
            raise ValueError(
                f"You may select up to {self.pick_size} cards for a single cook."
            )

        unique = sorted(set(indices))
        if len(unique) != len(indices):
            raise ValueError("Selections contain duplicates.")

        if any(index < 0 or index >= len(self.hand) for index in unique):
            raise IndexError("Selection index out of range for the current hand.")

        selected_cards: List[IngredientCard] = []
        for index in unique:
            card = self.hand[index]
            if card is None:
                raise ValueError("Selected slot is empty.")
            selected_cards.append(card)

        rotten_cards = [card for card in selected_cards if card.is_rotten]
        rotten_count = len(rotten_cards)

        selected = [card.ingredient for card in selected_cards]

        dish = self.data.evaluate_dish(selected)
        recipe_name = self.data.which_recipe(selected)

        fallback_combo = recipe_name is None and dish.entry is None

        dish_multiplier = dish.dish_multiplier
        dish_value_for_scoring = dish.dish_value
        if fallback_combo:
            dish_multiplier = 1.0
            dish_value_for_scoring = float(dish.base_value)

        alerts = list(dish.alerts)
        if alerts:
            for alert in alerts:
                self._push_event(alert)
        Value = dish.base_value
        ingredient_value_total = sum(card.ingredient.Value for card in selected_cards)
        original_recipe_name = recipe_name
        rotted = rotten_count > 0
        active_recipe_name = None if rotted else recipe_name
        recipe_display_name = (
            self.data.recipe_display_name(active_recipe_name)
            if active_recipe_name
            else None
        )
        times_cooked_before = self._times_cooked(active_recipe_name)
        recipe_multiplier = self.data.recipe_multiplier(
            active_recipe_name,
            chefs=self.chefs,
            times_cooked=times_cooked_before,
        )
        applied = applied_seasonings or {}
        seasoning_calc = self.calculate_seasoning_adjustments(
            selected, dish_value_for_scoring, applied
        )
        if seasoning_calc.ruined and seasoning_calc.usage:
            ruined_message = self.rng.choice(RUINED_SEASONING_MESSAGES)
            alerts.append(ruined_message)
            self._push_event(ruined_message)
        base_value = seasoning_calc.seasoned_score
        final_score = int(round(base_value * recipe_multiplier))
        if rotted:
            penalty_value = int(ingredient_value_total * rotten_count)
            base_value = -penalty_value
            final_score = -penalty_value
            recipe_multiplier = 1.0
            active_recipe_name = None
            recipe_display_name = None
            seasoning_calc = replace(
                seasoning_calc,
                base_score=base_value,
                seasoned_score=base_value,
                total_boost_pct=0.0,
                total_penalty=0.0,
                ruined=True,
            )
            penalty_message = (
                "Rotten ingredients spoiled the dish! Penalty "
                f"{final_score} points ({ingredient_value_total} total value x{rotten_count})."
            )
            alerts.append(penalty_message)
            self._push_event(penalty_message)
            if original_recipe_name:
                ruined_display = (
                    self.data.recipe_display_name(original_recipe_name)
                    or original_recipe_name
                )
                alerts.append(
                    f"{ruined_display} was ruined by spoiled ingredients."
                )
        chef_hits = sum(1 for ing in selected if ing.name in self._chef_key_set)

        discovered = False
        personal_discovery = False
        times_cooked_total = 0
        if active_recipe_name:
            recipe = self.data.recipe_by_name.get(active_recipe_name)
            if recipe:
                combo: Tuple[str, ...] = tuple(recipe.trio)
            else:
                combo = tuple(sorted(ingredient.name for ingredient in selected))
            entry = self.cookbook.get(active_recipe_name)
            chef_has_recipe = any(
                active_recipe_name in chef.recipe_names for chef in self.chefs
            )
            if not entry:
                display_name = (
                    recipe_display_name
                    or active_recipe_name
                    or ", ".join(combo)
                )
                entry = CookbookEntry(combo, display_name)
                entry.personal_discovery = not chef_has_recipe
                self.cookbook[active_recipe_name] = entry
                discovered = True
                personal_discovery = entry.personal_discovery
                self._cookbook_ingredients.update(combo)
            else:
                personal_discovery = False
            entry.count += 1
            times_cooked_total = entry.count
            display_times = entry.count
            if entry.personal_discovery and not chef_has_recipe:
                display_times = max(entry.count - 1, 0)
            entry.multiplier = self.data.recipe_multiplier(
                active_recipe_name,
                chefs=self.chefs,
                times_cooked=display_times,
            )
            if discovered:
                event_name = recipe_display_name or active_recipe_name
                if personal_discovery:
                    self._push_event(
                        f"You personally discovered {event_name}! Added to your cookbook."
                    )
                else:
                    self._push_event(
                        f"{event_name} added to your cookbook thanks to your chef team."
                    )

        current_round = self.round_index
        current_turn = self.turn_number + 1

        self.total_score += final_score
        self._round_score += final_score
        self.turn_number += 1
        self._round_stats.record_turn(
            score=final_score,
            rotten_count=rotten_count,
            recipe_name=active_recipe_name,
            ingredients=selected,
            dish_name=dish.name,
            dish_tier=dish.tier,
            recipe_display_name=recipe_display_name,
            family_label=dish.family_label,
            flavor_label=dish.flavor_label,
        )

        for offset, index in enumerate(unique):
            self.hand.pop(index - offset)

        if seasoning_calc.usage:
            boost_pct = seasoning_calc.total_boost_pct
            penalty_value = seasoning_calc.total_penalty
            boost_text = _format_percent(int(round(boost_pct * 100)))
            penalty_text = f"-{int(round(penalty_value))}" if penalty_value else "0"
            self._push_event(
                f"Seasonings applied: {boost_text} to base score, penalty {penalty_text}."
            )
            for usage in seasoning_calc.usage:
                charges = self.get_seasoning_charges(usage.seasoning.seasoning_id)
                if charges is not None:
                    remaining = max(charges - usage.count, 0)
                    self._seasoning_charges[usage.seasoning.seasoning_id] = remaining
                    display_name = usage.seasoning.display_name or usage.seasoning.name
                    if remaining > 0:
                        plural = "uses" if remaining != 1 else "use"
                        self._push_event(
                            f"{display_name} has {remaining} {plural} remaining this run."
                        )
                    else:
                        self._push_event(
                            f"{display_name} is tapped out for the rest of the run."
                        )

        deck_refreshed = False

        if (
            not self.finished
            and self.challenge_target is not None
            and self.total_score >= self.challenge_target
        ):
            self._enter_round_summary(reason="target_reached")

        if not self.finished:
            self._apply_end_turn_decay()
        if not self.finished:
            deck_refreshed = self._refill_hand() or deck_refreshed

        return TurnOutcome(
            selected=selected,
            Value=Value,
            dish_value=float(base_value),
            dish_multiplier=dish_multiplier,
            dish_name=dish.name,
            dish_tier=dish.tier,
            family_label=dish.family_label,
            flavor_label=dish.flavor_label,
            family_pattern=dish.family_pattern,
            flavor_pattern=dish.flavor_pattern,
            recipe_name=active_recipe_name,
            recipe_display_name=recipe_display_name,
            recipe_multiplier=recipe_multiplier,
            final_score=final_score,
            times_cooked_total=times_cooked_total,
            base_score=seasoning_calc.base_score,
            seasoning_boost_pct=seasoning_calc.total_boost_pct,
            seasoning_penalty=seasoning_calc.total_penalty,
            seasoned_score=base_value,
            ruined=seasoning_calc.ruined,
            applied_seasonings=tuple(
                (usage.seasoning.seasoning_id, usage.count)
                for usage in seasoning_calc.usage
            ),
            chef_hits=chef_hits,
            round_index=current_round,
            turn_number=current_turn,
            deck_refreshed=deck_refreshed,
            discovered_recipe=discovered,
            personal_discovery=personal_discovery,
            alerts=tuple(alerts),
        )

    def is_finished(self) -> bool:
        return self.finished


class CookbookTile(ttk.Frame):
    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master, style="Tile.TFrame", padding=(14, 12))
        self.entries: Dict[str, CookbookEntry] = {}
        self._entry_widgets: list[tk.Widget] = []
        self._empty_label: Optional[ttk.Label] = None

        self.columnconfigure(0, weight=1)

        self.header_var = tk.StringVar(value="📖 Cookbook")
        self.header_label = ttk.Label(
            self,
            textvariable=self.header_var,
            style="TileHeader.TLabel",
            anchor="w",
            justify="left",
        )
        self.header_label.grid(row=0, column=0, sticky="ew")

        self.subtitle_var = tk.StringVar(value="No recipes unlocked yet.")
        self.subtitle_label = ttk.Label(
            self,
            textvariable=self.subtitle_var,
            style="TileSub.TLabel",
            wraplength=260,
            justify="left",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.body_frame = ttk.Frame(self, style="TileBody.TFrame")
        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.entries_frame = ttk.Frame(self.body_frame, style="TileBody.TFrame")
        self.entries_frame.grid(row=0, column=0, sticky="ew")
        self.entries_frame.columnconfigure(0, weight=1)

    def set_entries(self, entries: Mapping[str, CookbookEntry]) -> None:
        self.entries = {name: entry.clone() for name, entry in entries.items()}
        self._refresh_entries()
        self._update_subtitle()

    def clear(self) -> None:
        self.entries.clear()
        self._refresh_entries()
        self._update_subtitle()

    def _refresh_entries(self) -> None:
        for widget in self._entry_widgets:
            widget.destroy()
        self._entry_widgets.clear()
        if self._empty_label:
            self._empty_label.destroy()
            self._empty_label = None

        if not self.entries:
            self._empty_label = ttk.Label(
                self.entries_frame,
                text="No recipes unlocked yet.",
                style="TileInfo.TLabel",
                wraplength=260,
                justify="left",
            )
            self._empty_label.grid(row=0, column=0, sticky="w")
            return

        sorted_entries = sorted(
            self.entries.values(), key=lambda entry: entry.display_name.lower()
        )
        for row, entry in enumerate(sorted_entries):
            times = "time" if entry.count == 1 else "times"
            multiplier_text = format_multiplier(entry.multiplier)
            source_note = " (Personal discovery)" if entry.personal_discovery else ""
            header = ttk.Label(
                self.entries_frame,
                text=(
                    f"{entry.display_name} — multiplier {multiplier_text}; "
                    f"cooked {entry.count} {times}{source_note}"
                ),
                style="TileInfo.TLabel",
                wraplength=260,
                justify="left",
            )
            header.grid(row=row * 2, column=0, sticky="w")
            self._entry_widgets.append(header)

            ingredients = ttk.Label(
                self.entries_frame,
                text="Ingredients: " + ", ".join(entry.ingredients),
                style="TileInfo.TLabel",
                wraplength=260,
                justify="left",
            )
            ingredients.grid(row=row * 2 + 1, column=0, sticky="w", pady=(0, 8))
            self._entry_widgets.append(ingredients)

    def _update_subtitle(self) -> None:
        if not self.entries:
            self.subtitle_var.set("No recipes unlocked yet.")
            return
        count = len(self.entries)
        plural = "recipe" if count == 1 else "recipes"
        self.subtitle_var.set(f"{count} {plural} discovered.")


class ChefTeamTile(ttk.Frame):
    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master, style="Tile.TFrame", padding=(14, 12))
        self.max_slots: int = DEFAULT_MAX_CHEFS
        self._slot_widgets: list[tk.Widget] = []

        self.columnconfigure(0, weight=1)

        self.header_var = tk.StringVar(value="👩‍🍳 Chef Team (0/0)")
        self.header_label = ttk.Label(
            self,
            textvariable=self.header_var,
            style="TileSub.TLabel",
            anchor="w",
            justify="left",
        )
        self.header_label.configure(font=("Helvetica", 12, "bold"))
        self.header_label.grid(row=0, column=0, sticky="ew")

        self.subtitle_var = tk.StringVar(value="No chefs recruited yet.")
        self.subtitle_label = ttk.Label(
            self,
            textvariable=self.subtitle_var,
            style="TileSub.TLabel",
            wraplength=260,
            justify="left",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.slots_frame = ttk.Frame(self, style="TileBody.TFrame")
        self.slots_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.slots_frame.columnconfigure(0, weight=1)

    def _clear_slots(self) -> None:
        for widget in self._slot_widgets:
            widget.destroy()
        self._slot_widgets.clear()

    def set_team(self, chefs: Sequence[Chef], max_slots: int) -> None:
        self.max_slots = max_slots
        count = len(chefs)
        self.header_var.set(f"👩‍🍳 Chef Team ({count}/{max_slots})")
        if count == 0:
            self.subtitle_var.set("No chefs recruited yet.")
        elif count >= max_slots:
            self.subtitle_var.set("Chef roster is full.")
        else:
            remaining = max_slots - count
            plural = "slot" if remaining == 1 else "slots"
            self.subtitle_var.set(f"{remaining} {plural} available for recruitment.")

        self._clear_slots()
        if max_slots <= 0:
            return

        for index in range(max_slots):
            if index < count:
                chef = chefs[index]
                recipes = ", ".join(chef.recipe_names) if chef.recipe_names else "No signature recipes"
                text = f"{chef.name}\nRecipes: {recipes}"
            else:
                text = "Open slot — recruit a chef to fill this space."
            label = ttk.Label(
                self.slots_frame,
                text=text,
                style="TileInfo.TLabel",
                wraplength=260,
                justify="left",
            )
            label.grid(row=index, column=0, sticky="w", pady=(0, 6))
            self._slot_widgets.append(label)

    def clear(self) -> None:
        self.set_team([], self.max_slots or DEFAULT_MAX_CHEFS)


class SeasoningTile(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        on_select: Optional[Callable[[Optional[Seasoning]], None]] = None,
    ) -> None:
        super().__init__(master, style="Tile.TFrame", padding=(14, 12))
        self._entry_widgets: list[tk.Widget] = []
        self._entry_frames: list[ttk.Frame] = []
        self._entry_labels: list[ttk.Label] = []
        self._icon_refs: list[tk.PhotoImage] = []
        self._seasonings: list[Seasoning] = []
        self._selected_name: Optional[str] = None
        self._on_select: Optional[Callable[[Optional[Seasoning]], None]] = on_select

        self.columnconfigure(0, weight=1)

        self.header_var = tk.StringVar(value="🧂 Seasonings (0)")
        self.header_label = ttk.Label(
            self,
            textvariable=self.header_var,
            style="TileHeader.TLabel",
            anchor="w",
            justify="left",
        )
        self.header_label.grid(row=0, column=0, sticky="ew")

        self.subtitle_var = tk.StringVar(
            value="Collect wildcards to cover missing flavors."
        )
        self.subtitle_label = ttk.Label(
            self,
            textvariable=self.subtitle_var,
            style="TileSub.TLabel",
            wraplength=260,
            justify="left",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.entries_frame = ttk.Frame(self, style="TileBody.TFrame")
        self.entries_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.entries_frame.columnconfigure(0, weight=1)

    def set_on_select(
        self, callback: Optional[Callable[[Optional[Seasoning]], None]]
    ) -> None:
        self._on_select = callback

    def _clear_entries(self) -> None:
        for widget in self._entry_widgets:
            widget.destroy()
        self._entry_widgets.clear()
        self._entry_frames.clear()
        self._entry_labels.clear()
        self._icon_refs.clear()

    def set_seasonings(
        self, seasonings: Sequence[Seasoning], remaining: int
    ) -> None:
        previous = self._selected_name
        self._seasonings = list(seasonings)
        count = len(self._seasonings)
        self.header_var.set(f"🧂 Seasonings ({count})")
        if count == 0:
            if remaining > 0:
                self.subtitle_var.set("No seasonings collected yet. Finish a round to claim one.")
            else:
                self.subtitle_var.set("All available seasonings have been collected.")
        else:
            if remaining > 0:
                self.subtitle_var.set(
                    f"{remaining} seasoning{'s' if remaining != 1 else ''} still waiting to be found."
                )
            else:
                self.subtitle_var.set("Every seasoning wildcard is in your pantry.")

        self._clear_entries()
        if not self._seasonings:
            self._selected_name = None
            empty_label = ttk.Label(
                self.entries_frame,
                text="Seasonings appear here once you claim them.",
                style="TileInfo.TLabel",
                wraplength=260,
                justify="left",
            )
            empty_label.grid(row=0, column=0, sticky="w")
            self._entry_widgets.append(empty_label)
            self._notify_selection()
            return

        for index, seasoning in enumerate(self._seasonings):
            self._build_entry(index, seasoning)

        if previous and any(seasoning.name == previous for seasoning in self._seasonings):
            self._set_selected_name(previous, notify=False)
        else:
            first = self._seasonings[0]
            self._set_selected_name(first.name, notify=False)

        self._notify_selection()

    def _build_entry(self, index: int, seasoning: Seasoning) -> None:
        frame = ttk.Frame(
            self.entries_frame,
            style="SeasoningEntry.TFrame",
            padding=(8, 6),
        )
        frame.grid(row=index, column=0, sticky="ew", pady=(0, 6))
        frame.columnconfigure(1, weight=1)

        icon = _load_seasoning_icon(seasoning, target_px=64)
        icon_label = ttk.Label(frame, image=icon, style="SeasoningEntry.TLabel")
        icon_label.grid(row=0, column=0, sticky="w")
        self._icon_refs.append(icon)

        display_name = seasoning.display_name or seasoning.name
        perk = seasoning.perk.strip()
        text_lines = [display_name]
        if perk:
            text_lines.append(perk)
        text = "\n".join(text_lines)

        label = ttk.Label(
            frame,
            text=text,
            style="SeasoningEntry.TLabel",
            wraplength=200,
            justify="left",
        )
        label.grid(row=0, column=1, sticky="w", padx=(10, 0))

        for widget in (frame, icon_label, label):
            widget.bind(
                "<Button-1>",
                lambda _e, idx=index: self._handle_select(idx),
            )

        self._entry_widgets.extend([frame, icon_label, label])
        self._entry_frames.append(frame)
        self._entry_labels.append(label)

    def _handle_select(self, index: int) -> None:
        if index < 0 or index >= len(self._seasonings):
            return
        self._set_selected_name(self._seasonings[index].name)

    def _set_selected_name(self, name: Optional[str], *, notify: bool = True) -> None:
        self._selected_name = name
        self._apply_selection_styles()
        if notify:
            self._notify_selection()

    def _apply_selection_styles(self) -> None:
        for frame, label, seasoning in zip(
            self._entry_frames, self._entry_labels, self._seasonings
        ):
            selected = seasoning.name == self._selected_name
            frame.configure(
                style="SeasoningEntrySelected.TFrame"
                if selected
                else "SeasoningEntry.TFrame"
            )
            label.configure(
                style="SeasoningEntrySelected.TLabel"
                if selected
                else "SeasoningEntry.TLabel"
            )

    def _notify_selection(self) -> None:
        if self._on_select:
            self._on_select(self.get_selected_seasoning())

    def get_selected_seasoning(self) -> Optional[Seasoning]:
        if not self._selected_name:
            return None
        for seasoning in self._seasonings:
            if seasoning.name == self._selected_name:
                return seasoning
        return None

    def clear(self) -> None:
        self.set_seasonings([], 0)


class CardView(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        index: int,
        ingredient: Ingredient,
        chef_names: Sequence[str],
        recipe_hints: Sequence[str],
        cookbook_state: Optional[RecipeIconState],
        on_click: Optional[Callable[[int], None]] = None,
        *,
        compact: bool = False,
        quantity: Optional[int] = None,
        rot_info: Optional[Mapping[str, Any]] = None,
        locked: bool = False,
        is_rotten: bool = False,
    ) -> None:
        base_style = "CardDisabled.TFrame" if locked else "Card.TFrame"
        padding = (8, 6) if compact else (10, 8)
        super().__init__(master, style=base_style, padding=padding)
        self.index = index
        self.on_click = on_click
        self.selected = False
        self.cookbook_state = cookbook_state
        self.quantity = quantity
        self.locked = locked
        self.is_rotten = is_rotten
        self.compact = compact
        self.rot_label: Optional[ttk.Label] = None
        self._rot_color: Optional[str] = None
        self.cookbook_icon: Optional[tk.PhotoImage] = None
        self.taste_image: Optional[tk.PhotoImage] = None
        self.family_image: Optional[tk.PhotoImage] = None
        self.taste_label: Optional[ttk.Label] = None
        self.family_label: Optional[ttk.Label] = None
        self.chef_label: Optional[ttk.Label] = None
        self.recipe_label: Optional[ttk.Label] = None

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        display_name = getattr(ingredient, "display_name", None) or ingredient.name
        name_text = "Rotten" if locked else display_name
        if is_rotten and not locked:
            name_text += " (Rotten)"
        if quantity and quantity > 1 and not locked:
            name_text += f" ×{quantity}"

        indicator_state: Optional[RecipeIconState] = None
        if cookbook_state:
            indicator_state = cookbook_state
            if locked or is_rotten:
                indicator_state = "blocked"
        if indicator_state:
            self.cookbook_icon = _load_cookbook_indicator(indicator_state)
            if self.cookbook_icon is None and cookbook_state:
                name_text += " 📖"
        elif cookbook_state:
            name_text += " 📖"

        title_style = "CardTitleDisabled.TLabel" if locked else "CardTitle.TLabel"
        value_style = "CardValueDisabled.TLabel" if locked else "CardValue.TLabel"
        self.name_label = ttk.Label(
            self,
            text=name_text,
            style=title_style,
            anchor="w",
            justify="left",
            image=self.cookbook_icon,
            compound="left" if self.cookbook_icon else "none",
        )
        self.name_label.grid(row=0, column=0, sticky="w")
        if self.cookbook_icon:
            self.name_label.image = self.cookbook_icon

        if locked:
            value_text = "—"
        else:
            value_prefix = "+" if ingredient.Value >= 0 else ""
            value_text = f"{value_prefix}{ingredient.Value}"
        self.value_label = ttk.Label(
            self,
            text=value_text,
            style=value_style,
            anchor="e",
            justify="right",
        )
        self.value_label.grid(row=0, column=1, sticky="ne")

        separator = ttk.Separator(self, orient="horizontal")
        sep_padding = (4, 6) if compact else (6, 8)
        separator.grid(row=1, column=0, columnspan=2, sticky="ew", pady=sep_padding)

        row_index = 2

        if locked or is_rotten:
            self.ingredient_image = _load_rotten_image()
        else:
            self.ingredient_image = _load_ingredient_image(ingredient)
        self.ingredient_image_label = ttk.Label(
            self,
            image=self.ingredient_image,
            style="CardImage.TLabel",
        )
        self.ingredient_image_label.grid(
            row=row_index,
            column=0,
            columnspan=2,
            sticky="n",
            pady=(0, 6 if not compact else 4),
        )
        row_index += 1

        if rot_info is not None:
            rot_text, rot_color = self._format_rot_text(rot_info, ingredient)
            rot_style = "RotIndicatorDisabled.TLabel" if locked else "RotIndicator.TLabel"
            self.rot_label = ttk.Label(
                self,
                text=rot_text,
                style=rot_style,
                anchor="w",
                justify="left",
                wraplength=220,
            )
            if rot_color:
                self.rot_label.configure(foreground=rot_color)
            self._rot_color = rot_color
            self.rot_label.grid(row=row_index, column=0, columnspan=2, sticky="w", pady=(0, 6))
            row_index += 1

        body_style = "CardBodyDisabled.TLabel" if locked else "CardBody.TLabel"
        if compact:
            if locked:
                self.taste_image = None
                self.family_image = None
            else:
                self.taste_image = _load_icon("taste", ingredient.taste)
                self.family_image = _load_icon("family", ingredient.family)

            icon_frame = ttk.Frame(self)
            icon_frame.grid(row=row_index, column=0, columnspan=2, sticky="ew")
            icon_frame.columnconfigure(0, weight=1)
            icon_frame.columnconfigure(1, weight=1)

            if self.taste_image:
                self.taste_label = ttk.Label(
                    icon_frame,
                    image=self.taste_image,
                    style=body_style,
                )
                self.taste_label.grid(row=0, column=0, sticky="e", padx=(0, 6))

            if self.family_image:
                self.family_label = ttk.Label(
                    icon_frame,
                    image=self.family_image,
                    style=body_style,
                )
                self.family_label.grid(row=0, column=1, sticky="w", padx=(6, 0))

            row_index += 1
        else:
            if locked:
                self.taste_image = None
                taste_text = f"Original: {display_name}"
            else:
                self.taste_image = _load_icon("taste", ingredient.taste)
                taste_text = f"Taste: {ingredient.taste}"
            taste_compound = "left" if self.taste_image else "none"
            self.taste_label = ttk.Label(
                self,
                text=taste_text,
                style=body_style,
                image=self.taste_image,
                compound=taste_compound,
            )
            self.taste_label.grid(row=row_index, column=0, columnspan=2, sticky="w")
            row_index += 1

            if locked:
                self.family_image = None
                family_text = "Slot locked until next round."
            else:
                self.family_image = _load_icon("family", ingredient.family)
                family_text = f"Family: {ingredient.family}"
            family_compound = "left" if self.family_image else "none"
            self.family_label = ttk.Label(
                self,
                text=family_text,
                style=body_style,
                image=self.family_image,
                compound=family_compound,
            )
            self.family_label.grid(
                row=row_index, column=0, columnspan=2, sticky="w", pady=(2, 0)
            )
            row_index += 1

            marker_style = "CardMarkerDisabled.TLabel" if locked else "CardMarker.TLabel"
            if chef_names and not locked:
                first, *rest = chef_names
                lines = [f"Chef Key: {first}"]
                lines.extend(f"           {name}" for name in rest)
                self.chef_label = ttk.Label(
                    self,
                    text="\n".join(lines),
                    style=marker_style,
                    justify="left",
                )
                self.chef_label.grid(
                    row=row_index, column=0, columnspan=2, sticky="w", pady=(4, 0)
                )
                row_index += 1

            hint_style = "CardHintDisabled.TLabel" if locked else "CardHint.TLabel"
            if locked:
                hint_text = "Recipes: Unavailable while rotten."
            else:
                hint_text = ", ".join(recipe_hints) if recipe_hints else "(none)"
                hint_text = f"Recipes: {hint_text}"
                if is_rotten:
                    hint_text += "\nRotten ingredients ruin dishes."
            self.recipe_label = ttk.Label(
                self,
                text=hint_text,
                style=hint_style,
                wraplength=220,
                justify="left",
            )
            self.recipe_label.grid(
                row=row_index, column=0, columnspan=2, sticky="w", pady=(4, 0)
            )

        if self.on_click and not self.locked:
            self.bind("<Button-1>", self._handle_click)
            for child in self.winfo_children():
                child.bind("<Button-1>", self._handle_click)

    @staticmethod
    def _format_rot_text(
        rot_info: Mapping[str, Any], ingredient: Ingredient
    ) -> tuple[str, Optional[str]]:
        total = int(rot_info.get("total", 0) or 0)
        filled = int(rot_info.get("filled", 0) or 0)
        is_rotten = bool(rot_info.get("is_rotten"))

        # Treat extremely long decay tracks as effectively stable for display purposes.
        if total <= 0 or total >= 12 or ingredient.rotten_turns >= 900:
            return ("Shelf life: Stable", "#245c2f")

        cells = rot_info.get("cells") or []
        bar = "".join("■" if cell == "filled" else "□" for cell in cells[:12])
        progress = f" ({bar})" if bar else ""

        if is_rotten:
            return (f"Rotten — ruins dishes{progress}", "#8b1a1a")

        remaining = max(total - filled, 0)
        plural = "turn" if remaining == 1 else "turns"
        color = "#b45309" if remaining <= 1 else "#245c2f"
        return (f"Rot in {remaining} {plural}{progress}", color)

    def _handle_click(self, _event) -> None:
        if self.on_click and not self.locked:
            self.on_click(self.index)

    def set_selected(self, selected: bool) -> None:
        if self.locked:
            self.selected = False
            self.configure(style="CardDisabled.TFrame")
            return

        self.selected = selected
        style = "CardSelected.TFrame" if selected else "Card.TFrame"
        self.configure(style=style)
        title_style = "CardTitleSelected.TLabel" if selected else "CardTitle.TLabel"
        body_style = "CardBodySelected.TLabel" if selected else "CardBody.TLabel"
        marker_style = (
            "CardMarkerSelected.TLabel" if selected else "CardMarker.TLabel"
        )
        hint_style = "CardHintSelected.TLabel" if selected else "CardHint.TLabel"
        value_style = "CardValueSelected.TLabel" if selected else "CardValue.TLabel"
        image_style = "CardImageSelected.TLabel" if selected else "CardImage.TLabel"
        rot_style = (
            "RotIndicatorSelected.TLabel" if selected else "RotIndicator.TLabel"
        )
        self.name_label.configure(style=title_style)
        self.value_label.configure(style=value_style)
        self.ingredient_image_label.configure(style=image_style)
        if self.taste_label:
            self.taste_label.configure(style=body_style)
        if self.family_label:
            self.family_label.configure(style=body_style)
        if self.chef_label:
            self.chef_label.configure(style=marker_style)
        if self.recipe_label:
            self.recipe_label.configure(style=hint_style)
        if self.rot_label:
            self.rot_label.configure(style=rot_style)
            if self._rot_color:
                self.rot_label.configure(foreground=self._rot_color)


class DeckPopup(tk.Toplevel):
    def __init__(
        self,
        master: tk.Widget,
        session: GameSession,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master)
        self.session = session
        self._on_close = on_close
        self._card_views: List[CardView] = []

        self.title("Ingredient Pantry")
        self.geometry("760x560")
        self.minsize(520, 400)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        heading = ttk.Label(
            container,
            text="Remaining ingredients in your pantry",
            style="Header.TLabel",
            anchor="w",
        )
        heading.grid(row=0, column=0, sticky="w")

        self.count_var = tk.StringVar(value="")
        self.count_label = ttk.Label(
            container,
            textvariable=self.count_var,
            style="Info.TLabel",
            anchor="w",
            justify="left",
        )
        self.count_label.grid(row=1, column=0, sticky="w", pady=(6, 8))

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview
        )
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Escape>", lambda _e: self._handle_close())

        self.set_session(session)

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        self.destroy()

    def set_session(self, session: GameSession) -> None:
        self.session = session
        self._render_cards()

    def _render_cards(self) -> None:
        for child in self.cards_frame.winfo_children():
            child.destroy()
        self._card_views.clear()

        if not self.session:
            self.count_var.set("No run in progress.")
            return

        deck = list(self.session.get_remaining_deck())
        total = len(deck)
        self.count_var.set(f"Cards remaining: {total}")

        if not deck:
            ttk.Label(
                self.cards_frame,
                text="Your pantry is empty. Draw or start a new round to restock it.",
                style="Info.TLabel",
                wraplength=600,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        counts = Counter()
        samples: Dict[str, Ingredient] = {}
        for ingredient in deck:
            counts[ingredient.name] += 1
            samples.setdefault(ingredient.name, ingredient)

        sorted_names = sorted(counts.keys(), key=lambda name: name.lower())
        columns = 3
        for index, name in enumerate(sorted_names):
            ingredient = samples[name]
            count = counts[name]
            chef_names, cookbook_state = self.session.get_selection_markers(ingredient)
            recipe_hints = self.session.get_recipe_hints(ingredient)
            card = CardView(
                self.cards_frame,
                index=index,
                ingredient=ingredient,
                chef_names=chef_names,
                recipe_hints=recipe_hints,
                cookbook_state=cookbook_state,
                on_click=None,
                quantity=count,
            )
            row, column = divmod(index, columns)
            card.grid(row=row, column=column, sticky="n", padx=8, pady=8)
            self.cards_frame.columnconfigure(column, weight=1)
            self._card_views.append(card)


class CookbookPopup(tk.Toplevel):
    def __init__(
        self,
        master: tk.Widget,
        entries: Mapping[str, CookbookEntry],
        data: GameData,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master)
        self.data = data
        self._on_close = on_close
        self.entries: Dict[str, CookbookEntry] = {}
        self._image_refs: List[tk.PhotoImage] = []

        self.title("Cookbook")
        self.geometry("720x560")
        self.minsize(560, 420)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        heading = ttk.Label(
            container,
            text="Recipes you've discovered",
            style="Header.TLabel",
            anchor="w",
        )
        heading.grid(row=0, column=0, sticky="w")

        self.count_var = tk.StringVar(value="")
        self.count_label = ttk.Label(
            container,
            textvariable=self.count_var,
            style="Info.TLabel",
            anchor="w",
            justify="left",
        )
        self.count_label.grid(row=1, column=0, sticky="w", pady=(6, 8))

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview
        )
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.entries_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.entries_frame, anchor="nw")
        self.entries_frame.columnconfigure(0, weight=1)
        self.entries_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Escape>", lambda _e: self._handle_close())

        self.set_entries(entries)

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        self.destroy()

    def set_entries(self, entries: Mapping[str, CookbookEntry]) -> None:
        self.entries = {name: entry.clone() for name, entry in entries.items()}
        self._render_entries()

    def _render_entries(self) -> None:
        for child in self.entries_frame.winfo_children():
            child.destroy()
        self._image_refs.clear()

        total = len(self.entries)
        if total == 0:
            self.count_var.set("No recipes discovered yet.")
            ttk.Label(
                self.entries_frame,
                text="Cook dishes during a run to add recipes to your cookbook.",
                style="Info.TLabel",
                wraplength=620,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        plural = "recipe" if total == 1 else "recipes"
        self.count_var.set(f"{total} {plural} discovered.")

        sorted_entries = sorted(
            self.entries.items(), key=lambda item: item[1].display_name.lower()
        )
        for row_index, (name, entry) in enumerate(sorted_entries):
            wrapper = ttk.Frame(self.entries_frame, padding=(0, 0))
            wrapper.grid(row=row_index, column=0, sticky="ew", pady=(0, 14))
            wrapper.columnconfigure(0, weight=1)

            text_frame = ttk.Frame(wrapper)
            text_frame.grid(row=0, column=0, sticky="ew")
            text_frame.columnconfigure(0, weight=1)

            title_text = entry.display_name or name
            if entry.personal_discovery:
                title_text += " ⭐"
            ttk.Label(
                text_frame,
                text=title_text,
                style="Header.TLabel",
                anchor="w",
                justify="left",
            ).grid(row=0, column=0, sticky="w")

            times = "time" if entry.count == 1 else "times"
            info_parts = [
                f"Multiplier {format_multiplier(entry.multiplier)}",
                f"Cooked {entry.count} {times}",
            ]
            if entry.personal_discovery:
                info_parts.append("Personal discovery")
            ttk.Label(
                text_frame,
                text=" • ".join(info_parts),
                style="Info.TLabel",
                wraplength=640,
                justify="left",
            ).grid(row=1, column=0, sticky="w", pady=(4, 8))

            ingredients_frame = ttk.Frame(text_frame)
            ingredients_frame.grid(row=2, column=0, sticky="w")

            for col_index, ingredient_name in enumerate(entry.ingredients):
                ingredient = self.data.ingredients.get(ingredient_name)
                if ingredient:
                    display_name = (
                        getattr(ingredient, "display_name", None) or ingredient.name
                    )
                    image = _load_ingredient_image(ingredient, target_px=96)
                else:
                    display_name = ingredient_name
                    image = None

                if image is not None:
                    self._image_refs.append(image)

                label = ttk.Label(
                    ingredients_frame,
                    text=display_name,
                    image=image,
                    compound="top",
                    style="CardBody.TLabel",
                    anchor="center",
                    justify="center",
                    padding=(6, 0),
                )
                label.grid(row=0, column=col_index, padx=6)

            recipe_image = _load_recipe_image(
                name, entry.display_name, target_px=RECIPE_IMAGE_TARGET_PX
            )
            if recipe_image is not None:
                self._image_refs.append(recipe_image)
                image_label = ttk.Label(
                    wrapper,
                    image=recipe_image,
                    style="CardBody.TLabel",
                    anchor="center",
                    justify="center",
                )
                image_label.grid(row=0, column=1, rowspan=3, sticky="ne", padx=(12, 0))

        self.canvas.yview_moveto(0)

class SeasoningPopup(tk.Toplevel):
    def __init__(
        self,
        master: tk.Widget,
        seasonings: Sequence[Seasoning],
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_close = on_close
        self._seasonings: List[Seasoning] = []
        self._icon_refs: List[tk.PhotoImage] = []

        self.title("Seasoning Pantry")
        self.geometry("520x420")
        self.minsize(380, 320)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        heading = ttk.Label(
            container,
            text="Seasonings you've discovered",
            style="Header.TLabel",
            anchor="w",
        )
        heading.grid(row=0, column=0, sticky="w")

        self.count_var = tk.StringVar(value="")
        self.count_label = ttk.Label(
            container,
            textvariable=self.count_var,
            style="Info.TLabel",
            anchor="w",
            justify="left",
        )
        self.count_label.grid(row=1, column=0, sticky="w", pady=(6, 8))

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview
        )
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.entries_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.entries_frame, anchor="nw")
        self.entries_frame.columnconfigure(0, weight=1)
        self.entries_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Escape>", lambda _e: self._handle_close())

        self.set_seasonings(seasonings)

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        self.destroy()

    def set_seasonings(self, seasonings: Sequence[Seasoning]) -> None:
        self._seasonings = list(seasonings)
        self._render_entries()

    def _render_entries(self) -> None:
        for child in self.entries_frame.winfo_children():
            child.destroy()
        self._icon_refs.clear()

        total = len(self._seasonings)
        if total == 0:
            self.count_var.set("No seasonings collected yet.")
            ttk.Label(
                self.entries_frame,
                text="Finish a round to claim a seasoning wildcard for your pantry.",
                style="Info.TLabel",
                wraplength=440,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        plural = "seasoning" if total == 1 else "seasonings"
        self.count_var.set(f"{total} {plural} collected.")

        for row_index, seasoning in enumerate(
            sorted(
                self._seasonings,
                key=lambda s: (s.display_name or s.name).lower(),
            )
        ):
            frame = ttk.Frame(
                self.entries_frame,
                style="SeasoningEntry.TFrame",
                padding=(10, 8),
            )
            frame.grid(row=row_index, column=0, sticky="ew", pady=(0, 10))
            frame.columnconfigure(1, weight=1)

            icon = _load_seasoning_icon(seasoning, target_px=96)
            self._icon_refs.append(icon)
            icon_label = ttk.Label(frame, image=icon, style="SeasoningEntry.TLabel")
            icon_label.grid(row=0, column=0, rowspan=2, sticky="n")

            display_name = seasoning.display_name or seasoning.name
            name_label = ttk.Label(
                frame,
                text=display_name,
                style="Header.TLabel",
                anchor="w",
                justify="left",
            )
            name_label.grid(row=0, column=1, sticky="w")

            perk_text = seasoning.perk.strip() or "No perk description available."
            info_text = f"Taste: {seasoning.taste}\n{perk_text}"
            info_label = ttk.Label(
                frame,
                text=info_text,
                style="SeasoningEntry.TLabel",
                wraplength=360,
                justify="left",
            )
            info_label.grid(row=1, column=1, sticky="w", pady=(4, 0))

        self.canvas.yview_moveto(0)


class ChefTeamPopup(tk.Toplevel):
    """Display the active chef roster with their signature recipe boosts."""

    def __init__(
        self,
        master: tk.Widget,
        chefs: Sequence[Chef],
        data: GameData,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_close = on_close
        self._data = data
        self._chefs: List[Chef] = []
        self._image_refs: List[tk.PhotoImage] = []

        self.title("Chef Team")
        self.geometry("680x520")
        self.minsize(520, 420)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        heading = ttk.Label(
            container,
            text="Active chefs and their perked recipes",
            style="Header.TLabel",
            anchor="w",
        )
        heading.grid(row=0, column=0, sticky="w")

        self.count_var = tk.StringVar(value="")
        self.count_label = ttk.Label(
            container,
            textvariable=self.count_var,
            style="Info.TLabel",
            anchor="w",
            justify="left",
        )
        self.count_label.grid(row=1, column=0, sticky="w", pady=(6, 8))

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview
        )
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.entries_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.entries_frame, anchor="nw")
        self.entries_frame.columnconfigure(0, weight=1)
        self.entries_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.bind("<Escape>", lambda _e: self._handle_close())

        self.set_team(chefs)

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        self.destroy()

    def set_team(self, chefs: Sequence[Chef]) -> None:
        self._chefs = list(chefs)
        self._render_entries()

    def _render_entries(self) -> None:
        for child in self.entries_frame.winfo_children():
            child.destroy()
        self._image_refs.clear()

        total = len(self._chefs)
        if total == 0:
            self.count_var.set("No chefs recruited yet.")
            ttk.Label(
                self.entries_frame,
                text="Complete a cook to earn a new chef offer.",
                style="Info.TLabel",
                wraplength=520,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        plural = "chef" if total == 1 else "chefs"
        self.count_var.set(f"{total} {plural} on your team.")

        for row_index, chef in enumerate(self._chefs):
            chef_frame = ttk.Frame(
                self.entries_frame,
                style="Tile.TFrame",
                padding=(12, 10),
            )
            chef_frame.grid(row=row_index, column=0, sticky="ew", pady=(0, 12))
            chef_frame.columnconfigure(0, weight=1)

            name_label = ttk.Label(
                chef_frame,
                text=chef.name,
                style="Header.TLabel",
                anchor="w",
                justify="left",
            )
            name_label.grid(row=0, column=0, sticky="w")

            recipes = list(getattr(chef, "recipe_names", []))
            if not recipes:
                ttk.Label(
                    chef_frame,
                    text="No signature recipes listed.",
                    style="Body.TLabel",
                    anchor="w",
                    justify="left",
                ).grid(row=1, column=0, sticky="w", pady=(6, 0))
                continue

            multipliers: Dict[str, float] = {}
            if isinstance(chef.perks, Mapping):
                raw_perks = chef.perks.get("recipe_multipliers", {})
                if isinstance(raw_perks, Mapping):
                    for recipe_name, value in raw_perks.items():
                        try:
                            multipliers[recipe_name] = float(value)
                        except (TypeError, ValueError):
                            continue

            recipes_frame = ttk.Frame(chef_frame)
            recipes_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            recipes_frame.columnconfigure(1, weight=1)

            for recipe_index, recipe_name in enumerate(recipes):
                display_name = (
                    self._data.recipe_display_name(recipe_name) or recipe_name
                )
                multiplier = multipliers.get(recipe_name)
                if multiplier is None:
                    base = self._data.recipe_multipliers.get(recipe_name)
                    multiplier = float(base) if base is not None else 1.0
                formatted_multiplier = format_multiplier(multiplier)

                entry_frame = ttk.Frame(recipes_frame, padding=(0, 6))
                entry_frame.grid(row=recipe_index, column=0, sticky="ew")
                entry_frame.columnconfigure(1, weight=1)

                image = _load_recipe_image(recipe_name, display_name, target_px=132)
                if image is not None:
                    self._image_refs.append(image)
                    image_label = ttk.Label(entry_frame, image=image)
                else:
                    image_label = ttk.Label(
                        entry_frame,
                        text="No art",
                        style="Info.TLabel",
                        width=14,
                        anchor="center",
                    )
                image_label.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 12))

                recipe_label = ttk.Label(
                    entry_frame,
                    text=display_name,
                    style="Body.TLabel",
                    anchor="w",
                    justify="left",
                )
                recipe_label.grid(row=0, column=1, sticky="w")

                multiplier_label = ttk.Label(
                    entry_frame,
                    text=f"Multiplier: {formatted_multiplier}",
                    style="Info.TLabel",
                    anchor="w",
                    justify="left",
                )
                multiplier_label.grid(row=1, column=1, sticky="w", pady=(2, 0))

        self.canvas.yview_moveto(0)


class FoodGameApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Food Deck Simulator — Desktop Prototype")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)

        self.session: Optional[GameSession] = None
        self.card_views: Dict[int, CardView] = {}
        self.selected_indices: set[int] = set()
        self.applied_seasonings: Dict[str, int] = {}
        self.spinboxes: List[ttk.Spinbox] = []
        self.cookbook_tile: Optional[CookbookTile] = None
        self.team_tile: Optional[ChefTeamTile] = None
        self.seasoning_tile: Optional[SeasoningTile] = None
        self.active_popup: Optional[tk.Toplevel] = None
        self.cookbook_popup: Optional["CookbookPopup"] = None
        self.seasoning_popup: Optional["SeasoningPopup"] = None
        self.chef_popup: Optional["ChefTeamPopup"] = None
        self._pending_round_summary: Optional[Dict[str, object]] = None
        self._round_summary_shown = False
        self._round_reward_claimed = False
        self._pending_challenge_reward_score: Optional[int] = None
        self._deferring_round_summary = False
        self.recruit_dialog: Optional[tk.Toplevel] = None
        self.deck_popup: Optional["DeckPopup"] = None
        self.dish_dialog: Optional[DishMatrixDialog] = None

        self.cook_button: Optional[ttk.Button] = None
        self.return_button: Optional[ttk.Button] = None
        self.cookbook_button: Optional[ttk.Button] = None
        self.dish_matrix_button: Optional[ttk.Button] = None
        self.seasoning_button: Optional[ttk.Button] = None
        self.basket_button: Optional[ttk.Button] = None
        self.chef_button: Optional[ttk.Button] = None
        self.log_panel: Optional[ttk.Frame] = None
        self.log_toggle_button: Optional[ttk.Button] = None
        self.log_text: Optional[tk.Text] = None

        self.hand_sort_modes: Tuple[str, ...] = ("name", "family", "taste")
        self.hand_sort_index = 0
        self.hand_sort_var = tk.StringVar(
            value=self._format_sort_label(self._current_sort_mode())
        )

        self._resource_button_images: Dict[str, tk.PhotoImage] = {}
        self._action_button_images: Dict[str, tk.PhotoImage] = {}
        self._seasoning_hand_icons: List[tk.PhotoImage] = []
        self.log_collapsed = True
        self._run_completion_notified = False
        self._app_launch_time = datetime.now()
        self._log_start_time: Optional[datetime] = None
        self._lifetime_total_score = 0
        self._persistent_cookbook: Dict[str, CookbookEntry] = {}
        self._persistent_chefs: List[Chef] = []
        self._persistent_seasonings: List[Seasoning] = []
        self._last_run_config: Optional[Dict[str, int]] = None

        self.challenge_factory = BasketChallengeFactory(DATA, TARGET_SCORE_CONFIG)
        self.challenge_summary_var = tk.StringVar(
            value=self._default_challenge_message()
        )
        self.challenge_offers: Optional[Tuple[BasketChallenge, ...]] = None
        self.challenge_dialog: Optional[tk.Toplevel] = None
        self.pending_run_config: Optional[Dict[str, int]] = None
        self.pantry_card_ids: List[str] = []

        self._init_styles()
        self._build_layout()
        self._refresh_score_details()

    # ----------------- UI setup -----------------
    def _default_challenge_message(self) -> str:
        return "No basket selected. Start a run to choose from three baskets."

    def _init_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_bg = "#f5f5f5"
        selected_bg = "#e6edf7"
        disabled_bg = "#f0e7e1"
        ingredient_image_bg = "#f4ebd0"
        title_font = ("Helvetica", 12, "bold")
        body_font = ("Helvetica", 10)
        marker_font = ("Helvetica", 9, "italic")
        tile_bg = "#ffffff"

        style.configure("Card.TFrame", background=base_bg, borderwidth=1, relief="solid")
        style.configure(
            "CardSelected.TFrame",
            background=selected_bg,
            borderwidth=2,
            relief="solid",
        )
        style.configure(
            "CardDisabled.TFrame",
            background=disabled_bg,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "CardImage.TLabel",
            background=ingredient_image_bg,
            anchor="center",
        )
        style.configure(
            "CardImageSelected.TLabel",
            background=ingredient_image_bg,
            anchor="center",
        )
        style.configure(
            "CardTitle.TLabel",
            font=title_font,
            foreground="#2f2f2f",
            background=base_bg,
        )
        style.configure(
            "CardTitleDisabled.TLabel",
            font=title_font,
            foreground="#7a1d1d",
            background=disabled_bg,
        )
        style.configure(
            "CardBody.TLabel",
            font=body_font,
            foreground="#3a3a3a",
            background=base_bg,
        )
        style.configure(
            "CardBodyDisabled.TLabel",
            font=body_font,
            foreground="#6a5f5f",
            background=disabled_bg,
        )
        style.configure(
            "CardMarker.TLabel",
            font=marker_font,
            foreground="#5a5a5a",
            background=base_bg,
        )
        style.configure(
            "CardMarkerDisabled.TLabel",
            font=marker_font,
            foreground="#6a5f5f",
            background=disabled_bg,
        )
        style.configure(
            "CardHint.TLabel",
            font=("Helvetica", 9),
            foreground="#4a4a4a",
            background=base_bg,
        )
        style.configure(
            "CardHintDisabled.TLabel",
            font=("Helvetica", 9),
            foreground="#6a5f5f",
            background=disabled_bg,
        )
        style.configure(
            "RotIndicator.TLabel",
            font=("Helvetica", 9, "bold"),
            foreground="#245c2f",
            background=base_bg,
        )
        style.configure(
            "RotIndicatorDisabled.TLabel",
            font=("Helvetica", 9, "bold"),
            foreground="#7a1d1d",
            background=disabled_bg,
        )
        style.configure(
            "RotIndicatorSelected.TLabel",
            font=("Helvetica", 9, "bold"),
            foreground="#245c2f",
            background=selected_bg,
        )
        style.configure(
            "CardValue.TLabel",
            font=("Helvetica", 11, "bold"),
            foreground="#1f1f1f",
            background=base_bg,
        )
        style.configure(
            "CardValueDisabled.TLabel",
            font=("Helvetica", 11, "bold"),
            foreground="#6a5f5f",
            background=disabled_bg,
        )
        style.configure(
            "CardTitleSelected.TLabel",
            font=title_font,
            foreground="#1f1f1f",
            background=selected_bg,
        )
        style.configure(
            "CardBodySelected.TLabel",
            font=body_font,
            foreground="#2d2d2d",
            background=selected_bg,
        )
        style.configure(
            "CardMarkerSelected.TLabel",
            font=marker_font,
            foreground="#4a4a4a",
            background=selected_bg,
        )
        style.configure(
            "CardHintSelected.TLabel",
            font=("Helvetica", 9),
            foreground="#383838",
            background=selected_bg,
        )
        style.configure(
            "CardValueSelected.TLabel",
            font=("Helvetica", 11, "bold"),
            foreground="#1f1f1f",
            background=selected_bg,
        )

        style.configure("Info.TLabel", font=("Helvetica", 10), foreground="#2f2f2f")
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#1f1f1f")
        style.configure("Score.TLabel", font=("Helvetica", 18, "bold"), foreground="#1f1f1f")
        style.configure("Summary.TLabel", font=("Helvetica", 16, "bold"), foreground="#1c1c1c")
        style.configure(
            "Estimate.TLabel",
            font=("Helvetica", 13, "bold"),
            foreground="#185339",
        )
        style.configure(
            "Tile.TFrame",
            background=tile_bg,
            borderwidth=1,
            relief="ridge",
        )
        style.configure("TileBody.TFrame", background=tile_bg)
        style.configure(
            "TileHeader.TButton",
            font=("Helvetica", 12, "bold"),
            anchor="w",
            padding=(8, 6),
        )
        style.configure(
            "TileHeader.TLabel",
            font=("Helvetica", 12, "bold"),
            foreground="#1f1f1f",
            background=tile_bg,
        )
        style.configure(
            "TileSub.TLabel",
            font=("Helvetica", 10),
            foreground="#3a3a3a",
            background=tile_bg,
        )
        style.configure(
            "TileInfo.TLabel",
            font=("Helvetica", 10),
            foreground="#2a2a2a",
            background=tile_bg,
        )
        style.configure(
            "SeasoningEntry.TFrame",
            background=tile_bg,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "SeasoningEntrySelected.TFrame",
            background=selected_bg,
            borderwidth=2,
            relief="solid",
        )
        style.configure(
            "SeasoningEntry.TLabel",
            font=body_font,
            foreground="#3a3a3a",
            background=tile_bg,
        )
        style.configure(
            "SeasoningEntrySelected.TLabel",
            font=body_font,
            foreground="#1f1f1f",
            background=selected_bg,
        )
        style.configure(
            "ResourceButton.TButton",
            font=("Helvetica", 14, "bold"),
            padding=(8, 12),
        )
        style.configure(
            "TileRecipe.TButton",
            font=("Helvetica", 10),
            anchor="w",
            padding=(6, 4),
        )
        style.configure(
            "TileAction.TButton",
            font=("Helvetica", 11, "bold"),
            padding=(8, 6),
        )

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.control_frame = ttk.LabelFrame(main, text="Session Setup", padding=12)
        self.control_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 16))

        self.game_frame = ttk.Frame(main)
        self.game_frame.grid(row=0, column=1, sticky="nsew")
        self.game_frame.columnconfigure(0, weight=1)
        self.game_frame.rowconfigure(1, weight=1)
        self.game_frame.rowconfigure(3, weight=1)

        self._build_controls()
        self._build_game_panel()

    def _build_controls(self) -> None:
        ttk.Label(
            self.control_frame,
            text="Basket Challenge",
            style="Header.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.control_frame,
            textvariable=self.challenge_summary_var,
            style="Info.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        ttk.Label(
            self.control_frame,
            text="Start a run to draw three basket options. Each has a target score and reward.",
            style="Info.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        config_frame = ttk.Frame(self.control_frame)
        config_frame.pack(anchor="w", pady=(8, 0))

        self.cooks_var = tk.IntVar(value=DEFAULT_CONFIG.cooks)
        self.hand_var = tk.IntVar(value=DEFAULT_HAND_SIZE)
        self.pick_var = tk.IntVar(value=DEFAULT_PICK_SIZE)
        self.max_chefs_var = tk.IntVar(value=DEFAULT_MAX_CHEFS)

        self._add_spinbox(config_frame, "Cooks / Round", self.cooks_var, 1, 12)
        self._add_spinbox(config_frame, "Hand Size", self.hand_var, 3, 10)
        self._add_spinbox(config_frame, "Pick Size", self.pick_var, 1, 5)
        self._add_spinbox(
            config_frame,
            "Max Chefs",
            self.max_chefs_var,
            1,
            max(1, len(DATA.chefs)),
        )

        self.start_button = ttk.Button(
            self.control_frame, text="Start Run", command=self.start_run
        )
        self.start_button.pack(fill="x", pady=(12, 6))

        self.reset_button = ttk.Button(
            self.control_frame, text="Reset", command=self.reset_session, state="disabled"
        )
        self.reset_button.pack(fill="x")

    def _add_spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        minimum: int,
        maximum: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(anchor="w", pady=4, fill="x")
        ttk.Label(frame, text=label).pack(anchor="w")
        spin = ttk.Spinbox(
            frame,
            from_=minimum,
            to=maximum,
            textvariable=variable,
            width=6,
        )
        spin.pack(anchor="w", pady=(2, 0))
        self.spinboxes.append(spin)

    def _snapshot_run_config(self) -> Dict[str, int]:
        return {
            "cooks": int(self.cooks_var.get()),
            "hand_size": int(self.hand_var.get()),
            "pick_size": int(self.pick_var.get()),
            "max_chefs": int(self.max_chefs_var.get()),
        }

    def _build_game_panel(self) -> None:
        score_frame = ttk.Frame(self.game_frame)
        score_frame.grid(row=0, column=0, sticky="ew")
        score_frame.columnconfigure(0, weight=1)
        score_frame.columnconfigure(1, weight=1)

        ttk.Label(score_frame, text="Total Score", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.score_var = tk.StringVar(value="0")
        ttk.Label(score_frame, textvariable=self.score_var, style="Score.TLabel").grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        self.target_score_var = tk.StringVar(value="Target Score: —")
        ttk.Label(score_frame, textvariable=self.target_score_var, style="Info.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        self.run_score_detail_var = tk.StringVar(value="Run Score: 0")
        ttk.Label(
            score_frame,
            textvariable=self.run_score_detail_var,
            style="Info.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.lifetime_score_var = tk.StringVar(value="Lifetime Score: 0")
        ttk.Label(
            score_frame,
            textvariable=self.lifetime_score_var,
            style="Info.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.progress_var = tk.StringVar(value="Round 0 / 0 — Turn 0 / 0")
        ttk.Label(score_frame, textvariable=self.progress_var, style="Info.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        self.selection_var = tk.StringVar(value="Selection: 0")
        ttk.Label(score_frame, textvariable=self.selection_var, style="Info.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        self.selection_summary_var = tk.StringVar(value="Value × Dish = Score")
        self.selection_summary_label = ttk.Label(
            score_frame,
            textvariable=self.selection_summary_var,
            style="Summary.TLabel",
            anchor="center",
            justify="center",
        )
        self.selection_summary_label.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        self.estimated_score_var = tk.StringVar(value="Estimated Score: —")
        self.estimated_score_label = ttk.Label(
            score_frame,
            textvariable=self.estimated_score_var,
            style="Estimate.TLabel",
            anchor="center",
            justify="center",
        )
        self.estimated_score_label.grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        self.sort_button = ttk.Button(
            score_frame,
            textvariable=self.hand_sort_var,
            command=self.cycle_hand_sort_mode,
        )
        self.sort_button.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.chefs_var = tk.StringVar(
            value=f"Active chefs ({DEFAULT_MAX_CHEFS} max): —"
        )
        ttk.Label(score_frame, textvariable=self.chefs_var, style="Info.TLabel").grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        seasoning_frame = ttk.LabelFrame(score_frame, text="Seasoning Prep", padding=(10, 6))
        seasoning_frame.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        seasoning_frame.columnconfigure(0, weight=1)

        ttk.Label(
            seasoning_frame,
            text="Seasoning hand — click to add to the dish.",
            style="Info.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.seasoning_hand_frame = ttk.Frame(seasoning_frame)
        self.seasoning_hand_frame.grid(row=1, column=0, sticky="ew", pady=(4, 2))
        self.seasoning_hand_frame.columnconfigure(0, weight=1)

        ttk.Label(
            seasoning_frame,
            text="Applied to dish:",
            style="Info.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.applied_seasonings_frame = ttk.Frame(seasoning_frame)
        self.applied_seasonings_frame.grid(row=3, column=0, sticky="ew", pady=(4, 2))
        self.applied_seasonings_frame.columnconfigure(0, weight=1)

        self.clear_seasonings_button = ttk.Button(
            seasoning_frame,
            text="Clear Seasonings",
            command=self.clear_applied_seasonings,
            state="disabled",
        )
        self.clear_seasonings_button.grid(row=4, column=0, sticky="e", pady=(4, 0))

        hand_container = ttk.Frame(self.game_frame)
        hand_container.grid(row=1, column=0, sticky="nsew")
        hand_container.columnconfigure(0, weight=1)
        hand_container.rowconfigure(0, weight=1)

        self.hand_canvas = tk.Canvas(hand_container, borderwidth=0, highlightthickness=0)
        self.hand_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            hand_container, orient="horizontal", command=self.hand_canvas.xview
        )
        scrollbar.grid(row=1, column=0, sticky="ew")
        self.hand_canvas.configure(xscrollcommand=scrollbar.set)

        self.hand_frame = ttk.Frame(self.hand_canvas)
        self.hand_canvas.create_window((0, 0), window=self.hand_frame, anchor="nw")
        self.hand_frame.bind(
            "<Configure>",
            lambda _e: self.hand_canvas.configure(scrollregion=self.hand_canvas.bbox("all")),
        )

        self.action_frame = ttk.Frame(self.game_frame)
        self.action_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.action_frame.columnconfigure(0, weight=1)
        self.action_frame.columnconfigure(1, weight=1)

        self.cook_button = ttk.Button(
            self.action_frame,
            text="Cook",
            command=self.cook_selected,
            state="disabled",
            compound="left",
        )
        self.cook_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        cook_icon = _load_button_image("cook.png", target_px=52)
        if cook_icon is None:
            cook_icon = _generate_button_icon("cook", "CK", size=64)
        self._action_button_images["cook"] = cook_icon
        self.cook_button.configure(image=cook_icon)

        self.return_button = ttk.Button(
            self.action_frame,
            text="Return",
            command=self.return_selected,
            state="disabled",
            compound="left",
        )
        self.return_button.grid(row=0, column=1, sticky="ew")

        return_icon = _load_button_image("return.png", target_px=52)
        if return_icon is None:
            return_icon = _generate_button_icon("return", "RT", size=64)
        self._action_button_images["return"] = return_icon
        self.return_button.configure(image=return_icon)

        resource_frame = ttk.Frame(self.action_frame)
        resource_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for column in range(5):
            resource_frame.columnconfigure(column, weight=1)

        self.cookbook_count_var = tk.StringVar(value="Cookbook\n0 recipes")
        cookbook_icon = _load_button_image(
            "cookbook.png", target_px=RESOURCE_BUTTON_ICON_PX
        )
        if cookbook_icon is None:
            cookbook_icon = _generate_button_icon(
                "cookbook", "CB", size=RESOURCE_BUTTON_ICON_PX
            )
        self._resource_button_images["cookbook"] = cookbook_icon
        self.cookbook_button = ttk.Button(
            resource_frame,
            textvariable=self.cookbook_count_var,
            image=self._resource_button_images["cookbook"],
            compound="top",
            style="ResourceButton.TButton",
            command=self.show_cookbook_panel,
        )
        self.cookbook_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        dish_icon = _load_button_image(
            "dishmatrix.png", target_px=RESOURCE_BUTTON_ICON_PX
        )
        if dish_icon is None:
            dish_icon = _generate_button_icon(
                "dish", "DM", size=RESOURCE_BUTTON_ICON_PX
            )
        self._resource_button_images["dish"] = dish_icon
        self.dish_matrix_button = ttk.Button(
            resource_frame,
            text="Dish Matrix",
            image=self._resource_button_images["dish"],
            compound="top",
            style="ResourceButton.TButton",
            command=self.show_dish_matrix,
        )
        self.dish_matrix_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        seasoning_icon = _load_button_image(
            "seasoning.png", target_px=RESOURCE_BUTTON_ICON_PX
        )
        if seasoning_icon is None:
            seasoning_icon = _generate_button_icon(
                "seasoning", "SN", size=RESOURCE_BUTTON_ICON_PX
            )
        self._resource_button_images["seasoning"] = seasoning_icon
        self.seasoning_button = ttk.Button(
            resource_frame,
            text="Seasonings\nNone",
            image=self._resource_button_images["seasoning"],
            compound="top",
            style="ResourceButton.TButton",
            command=self.show_selected_seasoning_info,
            state="disabled",
        )
        self.seasoning_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))

        basket_icon = _load_button_image(
            "pantry.png", target_px=RESOURCE_BUTTON_ICON_PX
        )
        if basket_icon is None:
            basket_icon = _generate_button_icon(
                "pantry", "PT", size=RESOURCE_BUTTON_ICON_PX
            )
        self._resource_button_images["basket"] = basket_icon
        self.basket_count_var = tk.StringVar(value="Pantry\n0")
        self.basket_button = ttk.Button(
            resource_frame,
            textvariable=self.basket_count_var,
            image=self._resource_button_images["basket"],
            compound="top",
            style="ResourceButton.TButton",
            command=self.show_deck_popup,
            state="disabled",
        )
        self.basket_button.grid(row=0, column=3, sticky="ew", padx=(0, 6))

        chef_icon = _load_button_image(
            "chefs.png", target_px=RESOURCE_BUTTON_ICON_PX
        )
        if chef_icon is None:
            chef_icon = _generate_button_icon(
                "chef", "CF", size=RESOURCE_BUTTON_ICON_PX
            )
        self._resource_button_images["chef"] = chef_icon
        self.chef_button = ttk.Button(
            resource_frame,
            text="Chefs\n0/0",
            image=self._resource_button_images["chef"],
            compound="top",
            style="ResourceButton.TButton",
            command=self.show_chef_team,
            state="disabled",
        )
        self.chef_button.grid(row=0, column=4, sticky="ew")

        self.log_panel = ttk.Frame(self.game_frame)
        self.log_panel.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        self.log_panel.columnconfigure(0, weight=1)
        self.log_panel.rowconfigure(1, weight=1)

        log_header = ttk.Frame(self.log_panel)
        log_header.grid(row=0, column=0, sticky="ew")
        log_header.columnconfigure(0, weight=1)

        ttk.Label(log_header, text="Event Log", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        toggle_text = "Show Log ▼" if self.log_collapsed else "Hide Log ▲"
        self.log_toggle_button = ttk.Button(
            log_header,
            text=toggle_text,
            command=self.toggle_log_panel,
        )
        self.log_toggle_button.grid(row=0, column=1, sticky="e")

        self.log_text = tk.Text(
            self.log_panel,
            height=17,
            wrap="word",
            background="#ffffff",
            foreground="#202020",
            relief="solid",
            borderwidth=1,
            padx=12,
            pady=10,
            font=("Helvetica", 10),
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        if self.log_collapsed:
            self.log_text.grid_remove()
        self.log_text.configure(state="disabled")

        self._update_cookbook_button()
        self._update_seasoning_button(None)
        self._update_chef_button()

    def _refresh_score_details(self) -> None:
        target_text = "Target Score: —"
        run_total = 0
        if self.session:
            run_total = self.session.get_total_score()
            target = getattr(self.session, "challenge_target", None)
            if target is not None:
                basket_name = getattr(getattr(self.session, "challenge", None), "basket_name", None)
                if basket_name:
                    target_text = f"Target Score: {target} — {basket_name}"
                else:
                    target_text = f"Target Score: {target}"
        self.target_score_var.set(target_text)
        self.run_score_detail_var.set(f"Run Score: {run_total}")
        self.lifetime_score_var.set(f"Lifetime Score: {self._lifetime_total_score}")
        self._update_cookbook_button()
        self._update_seasoning_button(None)
        self._update_chef_button()

    # ----------------- Session management -----------------
    def _refresh_challenge_summary(self) -> None:
        if self.session and self.session.challenge:
            challenge = self.session.challenge
            reward_text = format_challenge_reward_text(challenge.reward)
            unique_count, total_count = challenge_ingredient_counts(DATA, challenge)
            ingredient_text = format_challenge_ingredient_text(
                unique_count, total_count
            )

            summary = (
                f"{challenge.basket_name} ({challenge.difficulty.title()}) — "
                f"Target {challenge.target_score} pts"
                f" · {ingredient_text}"
                f" · Reward: {reward_text}"
            )
            self.challenge_summary_var.set(summary)
        else:
            self.challenge_summary_var.set(self._default_challenge_message())

    def _close_challenge_dialog(self) -> None:
        if self.challenge_dialog and self.challenge_dialog.winfo_exists():
            try:
                self.challenge_dialog.grab_release()
            except tk.TclError:
                pass
            self.challenge_dialog.destroy()
        self.challenge_dialog = None
        self.challenge_offers = None

    def _challenge_run_state(self) -> Dict[str, object]:
        run_state = _blank_run_state()
        run_state["pantry"] = list(self.pantry_card_ids)
        if self.session:
            chef_ids: List[str] = []
            for chef in self.session.chefs:
                identifier = getattr(chef, "chef_id", None) or getattr(chef, "name", "")
                if identifier:
                    chef_ids.append(identifier)
            if chef_ids:
                run_state["chefs_active"] = chef_ids
            owned_ids = [
                seasoning.seasoning_id
                for seasoning in self.session.get_seasonings()
                if getattr(seasoning, "seasoning_id", None)
            ]
            if owned_ids:
                run_state["seasoning_owned"] = owned_ids
        return run_state

    def _prompt_basket_selection(self) -> None:
        self._close_challenge_dialog()

        if not self.pending_run_config:
            return

        offers = self.challenge_factory.three_offers(self._challenge_run_state())
        self.challenge_offers = offers

        dialog = tk.Toplevel(self.root)
        dialog.title("Choose a Basket Challenge")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        self.challenge_dialog = dialog

        container = ttk.Frame(dialog, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)

        ttk.Label(
            container,
            text="Select a basket to set your starting ingredients and reward.",
            style="Header.TLabel",
            wraplength=420,
            justify="center",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")

        tiles_frame = ttk.Frame(container)
        tiles_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        for column in range(3):
            tiles_frame.columnconfigure(column, weight=1)

        detail_windows: List[tk.Toplevel] = []

        def close_detail_windows() -> None:
            for window in list(detail_windows):
                if window.winfo_exists():
                    window.destroy()
            detail_windows.clear()

        def ingredient_preview(challenge: BasketChallenge) -> str:
            entries = DATA.baskets.get(challenge.basket_name, [])
            lines: List[str] = []
            for name, copies in entries[:6]:
                qty = f"×{copies}" if copies > 1 else ""
                lines.append(f"• {name} {qty}".rstrip())
            if len(entries) > 6:
                lines.append("• …")
            return "\n".join(lines) if lines else "No ingredients listed."

        def choose_challenge(challenge: BasketChallenge) -> None:
            close_detail_windows()
            self._close_challenge_dialog()
            self._finalize_start_run(challenge)

        def show_basket_details(challenge: BasketChallenge) -> None:
            entries = DATA.baskets.get(challenge.basket_name, [])
            detail = tk.Toplevel(dialog)
            detail.title(f"{challenge.basket_name} Ingredients")
            detail.transient(dialog)
            detail.resizable(False, False)
            detail_windows.append(detail)

            def handle_close() -> None:
                if detail in detail_windows:
                    detail_windows.remove(detail)
                if detail.winfo_exists():
                    detail.destroy()

            detail.protocol("WM_DELETE_WINDOW", handle_close)
            detail.bind("<Escape>", lambda _e: handle_close())

            detail.columnconfigure(0, weight=1)
            detail.rowconfigure(0, weight=1)

            wrapper = ttk.Frame(detail, padding=16)
            wrapper.grid(row=0, column=0, sticky="nsew")
            wrapper.columnconfigure(0, weight=1)
            wrapper.columnconfigure(1, weight=0)
            wrapper.columnconfigure(2, weight=0)
            wrapper.rowconfigure(3, weight=1)

            ttk.Label(
                wrapper,
                text=f"{challenge.basket_name} — {challenge.difficulty.title()}",
                style="Header.TLabel",
            ).grid(row=0, column=0, columnspan=2, sticky="w")

            icon = _load_challenge_tile_image(challenge.basket_name, target_px=120)
            if icon is not None:
                ttk.Label(wrapper, image=icon).grid(
                    row=1, column=0, sticky="w", pady=(8, 8)
                )

            reward_text = format_challenge_reward_text(challenge.reward)
            unique_count, total_count = challenge_ingredient_counts(DATA, challenge)
            ingredient_text = format_challenge_ingredient_text(
                unique_count, total_count
            )
            ttk.Label(
                wrapper,
                text=(
                    f"Target {challenge.target_score} pts"
                    f" · {ingredient_text}"
                    f" · Reward: {reward_text}"
                ),
                style="TileInfo.TLabel",
                wraplength=320,
                justify="left",
            ).grid(row=2, column=0, columnspan=2, sticky="w")

            if icon is not None:
                detail._image_refs = [icon]  # type: ignore[attr-defined]

            if entries:
                canvas = tk.Canvas(wrapper, borderwidth=0, highlightthickness=0)
                canvas.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
                scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
                scrollbar.grid(row=3, column=2, sticky="ns", pady=(12, 0))
                canvas.configure(yscrollcommand=scrollbar.set)

                cards_frame = ttk.Frame(canvas)
                canvas.create_window((0, 0), window=cards_frame, anchor="nw")
                cards_frame.bind(
                    "<Configure>",
                    lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
                )
                cards_frame.columnconfigure(0, weight=1)

                card_views: List[CardView] = []
                for index, (name, copies) in enumerate(entries):
                    ingredient = DATA.ingredients.get(name)
                    if not ingredient:
                        missing = ttk.Label(
                            cards_frame,
                            text=f"{name} ×{copies} (ingredient data unavailable)",
                            style="Info.TLabel",
                            justify="left",
                            wraplength=320,
                        )
                        missing.grid(row=index, column=0, sticky="ew", pady=(0, 8))
                        continue

                    card = CardView(
                        cards_frame,
                        index=index,
                        ingredient=ingredient,
                        chef_names=[],
                        recipe_hints=[],
                        cookbook_state=None,
                        on_click=None,
                        compact=True,
                        quantity=copies,
                    )
                    card.grid(row=index, column=0, sticky="ew", pady=(0, 12))
                    cards_frame.rowconfigure(index, weight=0)
                    card_views.append(card)

                cards_frame.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))
                detail._card_views = card_views  # type: ignore[attr-defined]
            else:
                ttk.Label(
                    wrapper,
                    text="No ingredients listed for this basket.",
                    style="Info.TLabel",
                ).grid(row=3, column=0, sticky="w", pady=(12, 0))

            ttk.Button(
                wrapper,
                text="Close",
                command=handle_close,
            ).grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))

            detail.update_idletasks()
            self._center_popup(detail)
            detail.focus_force()

        challenge_images: List[tk.PhotoImage] = []

        for column, challenge in enumerate(offers):
            tile = ttk.Frame(tiles_frame, style="Tile.TFrame", padding=(14, 12))
            tile.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 12, 0),
            )
            tile.columnconfigure(0, weight=1)

            ttk.Label(
                tile,
                text=f"{challenge.basket_name}\n{challenge.difficulty.title()}",
                style="TileHeader.TLabel",
                justify="center",
                anchor="center",
            ).grid(row=0, column=0, sticky="ew")

            challenge_image = _load_challenge_tile_image(
                challenge.basket_name, target_px=148
            )
            if challenge_image:
                ttk.Label(tile, image=challenge_image).grid(
                    row=1, column=0, sticky="n", pady=(8, 4)
                )
                challenge_images.append(challenge_image)

            unique_count, total_count = challenge_ingredient_counts(DATA, challenge)
            ingredient_text = format_challenge_ingredient_text(
                unique_count, total_count
            )
            ttk.Label(
                tile,
                text=(
                    f"Target {challenge.target_score} pts"
                    f" · {ingredient_text}"
                ),
                style="TileInfo.TLabel",
                justify="center",
                anchor="center",
                wraplength=220,
            ).grid(row=2, column=0, sticky="ew", pady=(8, 4))

            ttk.Label(
                tile,
                text=f"Reward: {format_challenge_reward_text(challenge.reward)}",
                style="TileSub.TLabel",
                justify="center",
                anchor="center",
                wraplength=220,
            ).grid(row=3, column=0, sticky="ew")

            ttk.Separator(tile, orient="horizontal").grid(
                row=4, column=0, sticky="ew", pady=(8, 8)
            )

            ttk.Label(
                tile,
                text=ingredient_preview(challenge),
                style="TileInfo.TLabel",
                justify="left",
                anchor="w",
            ).grid(row=5, column=0, sticky="ew")

            ttk.Button(
                tile,
                text="View Ingredients",
                command=lambda c=challenge: show_basket_details(c),
                style="TileAction.TButton",
            ).grid(row=6, column=0, sticky="ew", pady=(6, 0))

            ttk.Button(
                tile,
                text="Select",
                command=lambda c=challenge: choose_challenge(c),
                style="TileAction.TButton",
            ).grid(row=7, column=0, sticky="ew", pady=(10, 0))

        button_frame = ttk.Frame(container)
        button_frame.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        button_frame.columnconfigure(0, weight=1)

        def cancel_selection() -> None:
            self._log_action("Basket selection canceled.")
            self.pending_run_config = None
            self.challenge_summary_var.set(self._default_challenge_message())
            close_detail_windows()
            self._close_challenge_dialog()

        ttk.Button(
            button_frame,
            text="Cancel",
            command=cancel_selection,
        ).grid(row=0, column=0, padx=4)

        dialog.protocol("WM_DELETE_WINDOW", cancel_selection)
        dialog.bind("<Escape>", lambda _e: cancel_selection())
        dialog.grab_set()
        self._center_popup(dialog)
        dialog._image_refs = challenge_images  # type: ignore[attr-defined]

    def start_run(self) -> None:
        self._destroy_active_popup()
        if self.deck_popup and self.deck_popup.winfo_exists():
            self.deck_popup.destroy()
        self.deck_popup = None
        self._close_recruit_dialog()
        self._close_challenge_dialog()

        try:
            cooks = int(self.cooks_var.get())
            hand_size = int(self.hand_var.get())
            pick_size = int(self.pick_var.get())
            max_chefs = int(self.max_chefs_var.get())

            if pick_size > hand_size:
                raise ValueError("Pick size cannot exceed hand size.")
            if max_chefs <= 0:
                raise ValueError("Max chefs must be at least 1.")
        except Exception as exc:  # pragma: no cover - user feedback path
            self._log_action(f"Start Run failed: {exc}")
            messagebox.showerror("Cannot start run", str(exc))
            return

        self.pending_run_config = {
            "cooks": cooks,
            "hand_size": hand_size,
            "pick_size": pick_size,
            "max_chefs": max_chefs,
        }
        self.challenge_summary_var.set("Select a basket to begin this run.")
        self._log_action("Start Run button pressed. Awaiting basket selection.")
        self._prompt_basket_selection()

    def _finalize_start_run(self, challenge: BasketChallenge) -> None:
        config = self.pending_run_config
        self.pending_run_config = None
        if config is None:
            return

        new_pantry_ids = list(self.pantry_card_ids)
        new_pantry_ids.extend(challenge.added_ing_ids)

        starting_chefs = list(self._persistent_chefs)
        max_chefs = config["max_chefs"]
        if len(starting_chefs) > max_chefs:
            self._log_action(
                "Starting roster exceeds the max chef limit; extra chefs will sit out this run."
            )
            starting_chefs = starting_chefs[:max_chefs]

        starting_seasonings = list(self._persistent_seasonings)

        try:
            self.session = GameSession(
                DATA,
                basket_name=challenge.basket_name,
                chefs=starting_chefs,
                hand_size=config["hand_size"],
                pick_size=config["pick_size"],
                max_chefs=max_chefs,
                challenge=challenge,
                pantry_card_ids=new_pantry_ids,
                starting_cookbook=self._persistent_cookbook,
                starting_seasonings=starting_seasonings,
            )
            self._run_completion_notified = False
            self._last_run_config = dict(config)
            self.pantry_card_ids = new_pantry_ids
        except Exception as exc:  # pragma: no cover - user feedback path
            self._log_action(f"Start Run failed: {exc}")
            messagebox.showerror("Cannot start run", str(exc))
            self.session = None
            self._refresh_challenge_summary()
            return

        cookbook_entries = self.session.get_cookbook()
        self._persistent_cookbook = cookbook_entries
        if self.cookbook_tile:
            self.cookbook_tile.set_entries(cookbook_entries)
        self._update_cookbook_button()
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.set_entries(cookbook_entries)
        if self.team_tile:
            self.team_tile.set_team(self.session.chefs, self.session.max_chefs)
        if self.seasoning_tile:
            self.seasoning_tile.set_seasonings(
                self.session.get_seasonings(),
                len(self.session.available_seasonings()),
            )

        self._refresh_challenge_summary()

        self._set_controls_active(False)
        self._set_action_buttons_enabled(True)
        self.reset_button.configure(state="normal")
        self.applied_seasonings.clear()
        self.estimated_score_var.set("Estimated Score: —")
        self._update_seasoning_panels()
        self.selected_indices.clear()
        self.update_selection_label()
        self.render_hand()
        self.update_status()
        self.clear_events()
        self._pending_challenge_reward_score = None
        self._log_start_time = datetime.now()
        self._log_run_settings()
        self._log_action(
            "Basket selected: "
            f"{challenge.basket_name} ({challenge.difficulty.title()}) — "
            f"target {challenge.target_score} pts."
        )
        self._log_action("Run initialized.")
        self.append_events(self.session.consume_events())
        self._update_seasoning_button(None)
        self._update_chef_button()
        self.write_result("Run started. Select ingredients and press COOK!")
        if self.session.is_finished():
            self._handle_run_finished()

    def _check_challenge_completion(self) -> None:
        if not self.session or not self.session.challenge:
            return
        if getattr(self.session, "challenge_reward_claimed", False):
            return

        target = getattr(self.session, "challenge_target", None)
        if target is None:
            return

        total_score = self.session.get_total_score()
        if total_score < target:
            return

        if self._pending_challenge_reward_score is not None:
            self._pending_challenge_reward_score = max(
                self._pending_challenge_reward_score, total_score
            )
            return

        if (
            self.active_popup
            and self.active_popup.winfo_exists()
            and getattr(self.active_popup, "_popup_kind", None) == "turn_summary"
        ):
            self._pending_challenge_reward_score = total_score
            return

        self._present_challenge_reward(total_score)

    def _present_challenge_reward(self, total_score: int) -> None:
        if not self.session or not self.session.challenge:
            return

        self.session.challenge_reward_claimed = True
        self.session._post_run_reward_pending = True

        challenge = self.session.challenge
        reward = dict(getattr(self.session, "challenge_reward", {}) or {})
        reward_type, reward_obj, reward_id = self._resolve_challenge_reward_candidate(reward)

        if reward_id:
            reward["id"] = reward_id

        display_name, perk_text = self._describe_reward_details(reward_type, reward_obj, reward)
        reward["name"] = display_name
        self.session.challenge_reward = reward

        base_descriptor = format_challenge_reward_text(reward).capitalize()

        if isinstance(reward_obj, Seasoning):
            reward_image = _load_seasoning_icon(reward_obj, target_px=108)
        elif isinstance(reward_obj, Chef):
            reward_image = _load_chef_icon(target_px=108)
        else:
            reward_image = _load_button_image("reward.png", target_px=108)
            if reward_image is None:
                reward_image = _load_chef_icon(target_px=108)

        popup = tk.Toplevel(self.root)
        popup.title("Basket Reward Earned")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup._popup_kind = "challenge_reward"  # type: ignore[attr-defined]

        frame = ttk.Frame(popup, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(
            frame,
            text=f"You reached {total_score} points!",
            style="Header.TLabel",
            anchor="w",
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(
            frame,
            text=f"Reward earned: {display_name}",
            style="Summary.TLabel",
            anchor="w",
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(
            frame,
            text=base_descriptor,
            style="TileSub.TLabel",
            anchor="w",
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        image_label = ttk.Label(frame, image=reward_image)
        image_label.grid(row=3, column=0, sticky="nw", pady=(12, 0), padx=(0, 12))
        image_label.image = reward_image  # type: ignore[attr-defined]

        description = perk_text or "Perk details unavailable for this reward."
        ttk.Label(
            frame,
            text=description,
            style="Info.TLabel",
            wraplength=360,
            justify="left",
        ).grid(row=3, column=1, sticky="w", pady=(12, 0))

        summary_var = tk.StringVar(value="")
        summary_label = ttk.Label(
            frame,
            textvariable=summary_var,
            style="Info.TLabel",
            wraplength=420,
            justify="left",
            anchor="w",
        )
        summary_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(18, 0))
        summary_label.grid_remove()

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=5, column=0, columnspan=2, sticky="e", pady=(18, 0))

        reward_finalized = False

        def close_popup() -> None:
            if not popup.winfo_exists():
                return
            try:
                popup.grab_release()
            except tk.TclError:
                pass
            popup.destroy()
            if self.active_popup is popup:
                self.active_popup = None
            self.root.after(50, self._show_basket_clear_popup)

        def finalize_reward() -> None:
            nonlocal reward_finalized
            if reward_finalized:
                close_popup()
                return
            reward_finalized = True

            if not self.session:
                summary_var.set(
                    "Basket summary unavailable because the session has already closed."
                )
                summary_label.grid()
                continue_button.configure(text="Close", command=close_popup)
                popup.protocol("WM_DELETE_WINDOW", close_popup)
                popup.bind("<Escape>", lambda _e: close_popup())
                return

            error_message: Optional[str] = None

            if reward_type == "seasoning" and isinstance(reward_obj, Seasoning):
                try:
                    self.session.add_seasoning(reward_obj)
                except Exception as exc:  # pragma: no cover - user feedback path
                    error_message = str(exc)
                else:
                    self._store_persistent_seasoning(reward_obj)
            elif reward_type == "chef" and isinstance(reward_obj, Chef):
                try:
                    self.session.add_chef(reward_obj)
                except Exception as exc:  # pragma: no cover - user feedback path
                    error_message = str(exc)
                else:
                    self._store_persistent_chef(reward_obj)

            summary_lines: List[str] = []

            if self.session:
                self.session.challenge_reward = reward
                self.session.challenge_reward_claimed = True
                self.session._post_run_reward_pending = False
                self.session.finished = True

                self.append_events(self.session.consume_events())
                if self.team_tile:
                    self.team_tile.set_team(self.session.chefs, self.session.max_chefs)
                if self.seasoning_tile:
                    self.seasoning_tile.set_seasonings(
                        self.session.get_seasonings(),
                        len(self.session.available_seasonings()),
                    )
                self._update_chef_button()
                self._update_seasoning_button(None)
                self._update_seasoning_panels()

            summary_lines[:0] = [
                f"You reached {total_score} points and cleared the {challenge.basket_name} basket!",
                "",
                f"Reward earned: {display_name}.",
            ]
            if description:
                summary_lines.append("")
                summary_lines.append(description)

            if error_message:
                summary_lines.append("")
                summary_lines.append(f"⚠️ Reward could not be applied: {error_message}")

            message_text = "\n".join(summary_lines)

            if self.session:
                self._log_action(
                    f"Basket challenge completed at {total_score} points. Reward granted: {display_name}."
                )
                log_lines = [
                    f"Challenge complete! Reward earned: {display_name}.",
                    f"Total score: {total_score}",
                ]
                if description:
                    log_lines.append(description)
                if error_message:
                    log_lines.append(f"Reward could not be applied: {error_message}")
                self.write_result("\n".join(log_lines))

                self._handle_run_finished(show_dialog=False)
                self.challenge_summary_var.set(
                    "Basket cleared! Select your next challenge."
                )

                next_config = self._last_run_config or self._snapshot_run_config()
                self.pending_run_config = dict(next_config)
                self._prompt_basket_selection()

            summary_var.set(message_text)
            summary_label.grid()
            continue_button.configure(text="Close", command=close_popup)
            popup.protocol("WM_DELETE_WINDOW", close_popup)
            popup.bind("<Escape>", lambda _e: close_popup())

        continue_button = ttk.Button(
            button_frame,
            text="Continue",
            command=finalize_reward,
            width=18,
        )
        continue_button.grid(row=0, column=0, sticky="e")

        popup.protocol("WM_DELETE_WINDOW", lambda: finalize_reward())
        popup.bind("<Escape>", lambda _e: finalize_reward())

        popup.grab_set()
        popup._image_refs = [reward_image]  # type: ignore[attr-defined]
        self.active_popup = popup
        self._center_popup(popup)

    def _resolve_challenge_reward_candidate(
        self, reward: Mapping[str, str]
    ) -> Tuple[str, Optional[Union[Chef, Seasoning]], str]:
        reward_type = reward.get("type", "reward").strip().lower().replace(" ", "_") or "reward"

        if reward_type == "chef":
            owned_names = {chef.name for chef in self._persistent_chefs}
            if self.session:
                owned_names.update(chef.name for chef in self.session.chefs)
            candidates = [chef for chef in DATA.chefs if chef.name not in owned_names]
            if not candidates:
                candidates = list(DATA.chefs)
            if not candidates:
                return reward_type, None, ""
            rng = getattr(self.session, "rng", random.Random())
            chosen = rng.choice(candidates)
            return reward_type, chosen, getattr(chosen, "chef_id", chosen.name)

        if reward_type == "seasoning":
            owned_ids = {seasoning.seasoning_id for seasoning in self._persistent_seasonings}
            if self.session:
                owned_ids.update(
                    seasoning.seasoning_id for seasoning in self.session.get_seasonings()
                )
            candidates = [
                seasoning
                for seasoning in DATA.seasonings
                if seasoning.seasoning_id not in owned_ids
            ]
            if not candidates:
                candidates = list(DATA.seasonings)
            if not candidates:
                return reward_type, None, ""
            rng = getattr(self.session, "rng", random.Random())
            chosen = rng.choice(candidates)
            return reward_type, chosen, chosen.seasoning_id

        return reward_type, None, ""

    def _format_chef_reward_description(self, chef: Chef) -> str:
        lines: List[str] = []
        recipe_names = [
            DATA.recipe_display_name(name) or name for name in chef.recipe_names
        ]
        if recipe_names:
            lines.append("Signature recipes: " + ", ".join(recipe_names))

        perks = chef.perks.get("recipe_multipliers") if isinstance(chef.perks, Mapping) else None
        if isinstance(perks, Mapping):
            for recipe_name, multiplier in sorted(perks.items()):
                display = DATA.recipe_display_name(recipe_name) or recipe_name
                try:
                    value = float(multiplier)
                except (TypeError, ValueError):
                    continue
                lines.append(f"• {display} {format_multiplier(value)}")

        if not lines:
            lines.append("No perk description available.")
        return "\n".join(lines)

    def _describe_reward_details(
        self,
        reward_type: str,
        reward_obj: Optional[Union[Chef, Seasoning]],
        reward: Mapping[str, str],
    ) -> Tuple[str, str]:
        if reward_type == "seasoning" and isinstance(reward_obj, Seasoning):
            display_name = reward_obj.display_name or reward_obj.name
            perk_lines = [f"Taste: {reward_obj.taste}"]
            perk = reward_obj.perk.strip()
            if perk:
                perk_lines.append(perk)
            return display_name, "\n".join(perk_lines)

        if reward_type == "chef" and isinstance(reward_obj, Chef):
            display_name = reward_obj.name
            perk_text = self._format_chef_reward_description(reward_obj)
            return display_name, perk_text

        fallback_name = format_challenge_reward_text(reward).title()
        return fallback_name, "Perk details unavailable for this reward."

    def _store_persistent_chef(self, chef: Chef) -> None:
        if all(existing.name != chef.name for existing in self._persistent_chefs):
            self._persistent_chefs.append(chef)

    def _store_persistent_seasoning(self, seasoning: Seasoning) -> None:
        if all(
            existing.seasoning_id != seasoning.seasoning_id
            for existing in self._persistent_seasonings
        ):
            self._persistent_seasonings.append(seasoning)

    def reset_session(self) -> None:
        self._destroy_active_popup()
        if self.deck_popup and self.deck_popup.winfo_exists():
            self.deck_popup.destroy()
        self.deck_popup = None
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.destroy()
        self.cookbook_popup = None
        if self.seasoning_popup and self.seasoning_popup.winfo_exists():
            self.seasoning_popup.destroy()
        self.seasoning_popup = None
        self._pending_round_summary = None
        self._round_summary_shown = False
        self._round_reward_claimed = False
        self._deferring_round_summary = False
        self._run_completion_notified = False
        self._close_recruit_dialog()
        self.session = None
        self._persistent_cookbook.clear()
        self._persistent_chefs.clear()
        self._persistent_seasonings.clear()
        self.pantry_card_ids = []
        self._refresh_challenge_summary()
        self._update_basket_button()
        self.selected_indices.clear()
        self.update_selection_label()
        self.score_var.set("0")
        self.progress_var.set("Round 0 / 0 — Turn 0 / 0")
        self.chefs_var.set(
            f"Active chefs ({self.max_chefs_var.get()} max): —"
        )
        self.applied_seasonings.clear()
        self.estimated_score_var.set("Estimated Score: —")
        self._set_action_buttons_enabled(False)
        self.reset_button.configure(state="disabled")
        self._set_controls_active(True)
        self.clear_hand()
        self.clear_events()
        self._log_action("Reset button pressed. Session cleared.")
        self.write_result("Session reset. Configure options and start a new run.")
        if self.cookbook_tile:
            self.cookbook_tile.clear()
        self._update_cookbook_button()
        if self.team_tile:
            self.team_tile.set_team([], int(self.max_chefs_var.get()))
        if self.seasoning_tile:
            self.seasoning_tile.clear()
        self._update_seasoning_button(None)
        self._update_chef_button()
        self._update_seasoning_panels()

    def _set_controls_active(self, active: bool) -> None:
        state = "normal" if active else "disabled"
        for spin in self.spinboxes:
            spin.configure(state=state)
        self.start_button.configure(state="normal" if active else "disabled")

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (
            self.cook_button,
            self.return_button,
            self.basket_button,
            self.seasoning_button,
            self.chef_button,
        ):
            if button is not None:
                button.configure(state=state)

    # ----------------- UI updates -----------------
    def _refresh_deck_popup(self) -> None:
        if not self.deck_popup:
            return
        if self.deck_popup.winfo_exists():
            if self.session:
                self.deck_popup.set_session(self.session)
            else:
                self.deck_popup.destroy()
                self.deck_popup = None
        else:
            self.deck_popup = None

    def _refresh_seasoning_popup(self) -> None:
        if not self.seasoning_popup:
            return
        if self.seasoning_popup.winfo_exists():
            if self.session:
                self.seasoning_popup.set_seasonings(self.session.get_seasonings())
            else:
                self.seasoning_popup.destroy()
                self.seasoning_popup = None
        else:
            self.seasoning_popup = None

    def render_hand(self) -> None:
        self.clear_hand()
        if not self.session:
            self._update_basket_button()
            self._refresh_deck_popup()
            self._refresh_seasoning_popup()
            self._update_seasoning_panels()
            return

        self._refresh_seasoning_popup()
        if (
            not self.session.awaiting_new_round()
            and not self.session.is_finished()
        ):
            self._set_action_buttons_enabled(True)
        hand_cards = list(self.session.get_hand())
        self.selected_indices = {
            index for index in self.selected_indices if index < len(hand_cards)
        }

        hand_with_indices = list(enumerate(hand_cards))
        sorted_hand = self._sorted_hand(hand_with_indices)

        for column, (index, card) in enumerate(sorted_hand):
            ingredient = card.ingredient
            chef_names, cookbook_state = self.session.get_selection_markers(ingredient)
            recipe_hints: Sequence[str]
            recipe_hints = self.session.get_recipe_hints(ingredient)
            rot_info = rot_circles(card)
            view = CardView(
                self.hand_frame,
                index=index,
                ingredient=ingredient,
                chef_names=chef_names,
                recipe_hints=recipe_hints,
                cookbook_state=cookbook_state,
                on_click=self.toggle_card,
                rot_info=rot_info,
                is_rotten=card.is_rotten,
            )
            view.grid(row=0, column=column, sticky="nw", padx=8, pady=8)
            if index in self.selected_indices:
                view.set_selected(True)
            self.card_views[index] = view

        self.hand_frame.update_idletasks()
        self.hand_canvas.configure(scrollregion=self.hand_canvas.bbox("all"))
        self._refresh_deck_popup()
        self._update_basket_button()
        self._update_seasoning_panels()

    def _sorted_hand(
        self, hand_with_indices: Sequence[Tuple[int, IngredientCard]]
    ) -> List[Tuple[int, IngredientCard]]:
        mode = self._current_sort_mode()
        if mode == "name":
            key_func = lambda pair: (pair[1].ingredient.name.lower(), pair[0])
        elif mode == "family":
            key_func = lambda pair: (
                pair[1].ingredient.family.lower(),
                pair[1].ingredient.name.lower(),
                pair[0],
            )
        else:
            key_func = lambda pair: (
                pair[1].ingredient.taste.lower(),
                pair[1].ingredient.name.lower(),
                pair[0],
            )
        return sorted(hand_with_indices, key=key_func)

    def clear_hand(self) -> None:
        for view in self.card_views.values():
            view.destroy()
        self.card_views = {}

    def show_deck_popup(self) -> None:
        self._log_action("Pantry button pressed.")
        if not self.session:
            self._log_action("Pantry button pressed without an active session.")
            messagebox.showinfo(
                "No run in progress", "Start a run to view your ingredient pantry."
            )
            return

        if self.deck_popup and self.deck_popup.winfo_exists():
            self.deck_popup.set_session(self.session)
            self.deck_popup.lift()
            self.deck_popup.focus_force()
            self._log_action("Pantry popup focused.")
            return

        def handle_close() -> None:
            self.deck_popup = None

        self.deck_popup = DeckPopup(self.root, self.session, on_close=handle_close)
        self.deck_popup.transient(self.root)
        self.deck_popup.focus_force()
        self._center_popup(self.deck_popup)
        self._log_action("Pantry popup opened for the current run.")

    def show_cookbook_panel(self) -> None:
        self._log_action("Cookbook button pressed.")
        if not self.session:
            self._log_action("Cookbook button pressed without an active session.")
            messagebox.showinfo(
                "Cookbook", "Start a run to discover recipes and view your cookbook."
            )
            return
        entries = self.session.get_cookbook()
        self._update_cookbook_button()
        if not entries:
            self._log_action("Cookbook opened without any discovered recipes.")
            messagebox.showinfo(
                "Cookbook", "No recipes unlocked yet. Cook dishes to discover more."
            )
            return
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.set_entries(entries)
            self.cookbook_popup.lift()
            self.cookbook_popup.focus_force()
            self._log_action("Cookbook popup focused.")
            return

        def handle_close() -> None:
            self.cookbook_popup = None

        self.cookbook_popup = CookbookPopup(
            self.root, entries, DATA, on_close=handle_close
        )
        self.cookbook_popup.transient(self.root)
        self._center_popup(self.cookbook_popup)
        self.cookbook_popup.focus_force()
        self._log_action(
            f"Cookbook popup opened with {len(entries)} discovered recipes."
        )

    def show_dish_matrix(self) -> None:
        self._log_action("Dish matrix button pressed.")
        if self.dish_dialog and self.dish_dialog.winfo_exists():
            self.dish_dialog.lift()
            self.dish_dialog.focus_force()
            self._log_action("Dish matrix dialog focused.")
            return

        def handle_close() -> None:
            self.dish_dialog = None

        self.dish_dialog = DishMatrixDialog(
            self.root, DATA.dish_matrix, on_close=handle_close
        )
        self.dish_dialog.transient(self.root)
        self._center_popup(self.dish_dialog)
        self.dish_dialog.focus_force()
        self._log_action("Dish matrix dialog opened.")

    def show_selected_seasoning_info(self) -> None:
        self._log_action("Seasoning collection button pressed.")
        if not self.session:
            self._log_action(
                "Seasoning collection button pressed without an active session."
            )
            messagebox.showinfo(
                "Seasonings", "Start a run to collect seasonings for your pantry."
            )
            return

        seasonings = self.session.get_seasonings()
        if not seasonings:
            self._log_action("Seasoning collection viewed with no seasonings available.")
            messagebox.showinfo(
                "Seasonings", "No seasonings collected yet. Finish a round to claim one."
            )
            return

        if self.seasoning_popup and self.seasoning_popup.winfo_exists():
            self.seasoning_popup.set_seasonings(seasonings)
            self.seasoning_popup.lift()
            self.seasoning_popup.focus_force()
            self._log_action("Seasoning popup focused.")
            return

        def handle_close() -> None:
            self.seasoning_popup = None

        self.seasoning_popup = SeasoningPopup(
            self.root, seasonings, on_close=handle_close
        )
        self.seasoning_popup.transient(self.root)
        self._center_popup(self.seasoning_popup)
        self.seasoning_popup.focus_force()
        self._log_action(
            f"Seasoning popup opened with {len(seasonings)} collected items."
        )

    def show_chef_team(self) -> None:
        self._log_action("Chef team button pressed.")
        self._update_chef_button()
        if not self.session:
            self._log_action("Chef team button pressed without an active session.")
            messagebox.showinfo(
                "Chef Team", "Start a run to recruit chefs and view your roster."
            )
            return
        if self.session.can_recruit_chef():
            self._log_action("Chef recruitment dialog opened from chef team button.")
            self.show_recruit_dialog()
            return

        if not self.session.chefs:
            message = "No chefs recruited yet. Complete a cook to earn a new offer."
            self._log_action("Chef roster viewed with no active chefs.")
            messagebox.showinfo("Chef Team", message)
            return

        if self.chef_popup and self.chef_popup.winfo_exists():
            self.chef_popup.set_team(self.session.chefs)
            self.chef_popup.lift()
            self.chef_popup.focus_force()
            self._log_action("Chef popup focused.")
            return

        def handle_close() -> None:
            self.chef_popup = None

        self.chef_popup = ChefTeamPopup(
            self.root,
            self.session.chefs,
            DATA,
            on_close=handle_close,
        )
        self.chef_popup.transient(self.root)
        self._center_popup(self.chef_popup)
        self.chef_popup.focus_force()
        self._log_action(
            f"Chef popup opened with {len(self.session.chefs)} active chef(s)."
        )

    def _handle_seasoning_selected(self, seasoning: Optional[Seasoning]) -> None:
        self._update_seasoning_button(seasoning)

    def _update_cookbook_button(self) -> None:
        if not self.cookbook_button:
            return

        if not self.session:
            count = 0
        else:
            count = len(self.session.get_cookbook())

        plural = "recipe" if count == 1 else "recipes"
        self.cookbook_count_var.set(f"Cookbook\n{count} {plural}")

        icon = self._resource_button_images.get("cookbook")
        if icon is None:
            icon = _generate_button_icon(
                "cookbook", "CB", size=RESOURCE_BUTTON_ICON_PX
            )
            self._resource_button_images["cookbook"] = icon

        self.cookbook_button.configure(image=icon)

    def _update_basket_button(self) -> None:
        if not self.basket_button:
            return

        if not self.session:
            self.basket_count_var.set("Pantry\n0")
            return

        remaining, _ = self.session.get_basket_counts()
        self.basket_count_var.set(f"Pantry\n{remaining}")

    def _update_seasoning_button(self, seasoning: Optional[Seasoning] = None) -> None:
        if not self.seasoning_button:
            return

        session = self.session
        seasonings: Sequence[Seasoning] = ()
        if session:
            seasonings = session.get_seasonings()

        text: str
        if seasoning is None:
            if len(seasonings) == 1:
                seasoning = seasonings[0]
            elif seasonings:
                text = "Seasonings\nMultiple"
            else:
                text = "Seasonings\nNone"

        if seasoning is not None:
            display_name = seasoning.display_name or seasoning.name
            text = f"Seasonings\n{display_name}"
        elif seasonings:
            text = "Seasonings\nMultiple"
        else:
            text = "Seasonings\nNone"

        icon = self._resource_button_images.get("seasoning")
        if icon is None:
            icon = _generate_button_icon(
                "seasoning", "SN", size=RESOURCE_BUTTON_ICON_PX
            )
            self._resource_button_images["seasoning"] = icon

        state = "normal" if session and not session.is_finished() else "disabled"
        self.seasoning_button.configure(image=icon, text=text, state=state)

    def _seasoning_boost_summary(self, seasoning: Seasoning) -> str:
        parts = []
        for taste, boost in sorted(seasoning.boosts.items()):
            percent = int(round(float(boost) * 100))
            if percent == 0:
                continue
            parts.append(f"{taste}: {_format_percent(percent)}")
        return ", ".join(parts) if parts else "No boosts"

    def _can_apply_seasoning(self, seasoning: Seasoning) -> bool:
        if not self.session or self.session.is_finished():
            return False
        if not self.selected_indices:
            return False
        count = self.applied_seasonings.get(seasoning.seasoning_id, 0)
        if count >= max(1, seasoning.stack_limit):
            return False
        charges = self.session.get_seasoning_charges(seasoning.seasoning_id)
        if charges is not None and count >= charges:
            return False
        return True

    def _render_seasoning_hand(self) -> None:
        if not hasattr(self, "seasoning_hand_frame"):
            return
        for child in self.seasoning_hand_frame.winfo_children():
            child.destroy()
        self._seasoning_hand_icons = []

        if not self.session:
            ttk.Label(
                self.seasoning_hand_frame,
                text="Start a run to discover seasonings.",
                style="Info.TLabel",
                wraplength=320,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        active = self.session.get_active_seasonings()
        if not active:
            ttk.Label(
                self.seasoning_hand_frame,
                text="No seasonings ready. Collect more during the run.",
                style="Info.TLabel",
                wraplength=320,
                justify="left",
            ).grid(row=0, column=0, sticky="w")
            return

        sorted_active = sorted(
            active,
            key=lambda pair: (pair[0].display_name or pair[0].name).lower(),
        )
        for column, (seasoning, charges) in enumerate(sorted_active):
            display_name = seasoning.display_name or seasoning.name
            uses_text = "Uses ∞" if charges is None else f"Uses {charges}"
            stack_text = f"Stack {max(1, seasoning.stack_limit)}"
            info = self._seasoning_boost_summary(seasoning)
            lines = [f"{display_name}", info, f"{uses_text} · {stack_text}"]
            icon = _load_seasoning_icon(seasoning, target_px=72)
            if icon is not None:
                self._seasoning_hand_icons.append(icon)
            button = ttk.Button(
                self.seasoning_hand_frame,
                text="\n".join(lines),
                image=icon,
                compound="top",
                command=lambda s=seasoning: self.apply_seasoning_to_dish(s),
                style="TileAction.TButton",
            )
            button.grid(
                row=0,
                column=column,
                sticky="n",
                padx=(0 if column == 0 else 8, 0),
                pady=(0, 6),
            )
            self.seasoning_hand_frame.grid_columnconfigure(column, weight=1)
            if not self._can_apply_seasoning(seasoning):
                button.state(["disabled"])

    def _render_applied_seasonings(self) -> None:
        if not hasattr(self, "applied_seasonings_frame"):
            return
        for child in self.applied_seasonings_frame.winfo_children():
            child.destroy()

        if not self.session:
            ttk.Label(
                self.applied_seasonings_frame,
                text="No run in progress.",
                style="Info.TLabel",
            ).grid(row=0, column=0, sticky="w")
            if hasattr(self, "clear_seasonings_button"):
                self.clear_seasonings_button.configure(state="disabled")
            return

        if not self.applied_seasonings:
            ttk.Label(
                self.applied_seasonings_frame,
                text="No seasonings applied yet.",
                style="Info.TLabel",
            ).grid(row=0, column=0, sticky="w")
            if hasattr(self, "clear_seasonings_button"):
                self.clear_seasonings_button.configure(state="disabled")
            return

        if hasattr(self, "clear_seasonings_button"):
            self.clear_seasonings_button.configure(state="normal")

        sorted_items = sorted(
            self.applied_seasonings.items(),
            key=lambda item: (
                (self.session.data.seasoning_by_id.get(item[0]).display_name or
                 self.session.data.seasoning_by_id.get(item[0]).name)
                if self.session and item[0] in self.session.data.seasoning_by_id
                else item[0]
            ).lower(),
        )
        for row, (seasoning_id, count) in enumerate(sorted_items):
            seasoning = self.session.data.seasoning_by_id.get(seasoning_id)
            display_name = seasoning.display_name or seasoning.name if seasoning else seasoning_id
            charges = self.session.get_seasoning_charges(seasoning_id)
            if charges is not None:
                after_cook = max(charges - count, 0)
                label_text = f"{display_name} ×{count} (after cook: {after_cook} left)"
            else:
                label_text = f"{display_name} ×{count}"

            row_frame = ttk.Frame(self.applied_seasonings_frame)
            row_frame.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            row_frame.columnconfigure(0, weight=1)

            ttk.Label(row_frame, text=label_text, style="Info.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Button(
                row_frame,
                text="Remove",
                command=lambda sid=seasoning_id: self.remove_applied_seasoning(sid),
            ).grid(row=0, column=1, sticky="e", padx=(8, 0))

    def _update_seasoning_panels(self) -> None:
        self._render_seasoning_hand()
        self._render_applied_seasonings()

    def apply_seasoning_to_dish(self, seasoning: Seasoning) -> None:
        display_name = seasoning.display_name or seasoning.name
        if not self.session:
            self._log_action(
                f"Attempted to apply seasoning {display_name} without an active session."
            )
            messagebox.showinfo("No run in progress", "Start a run before seasoning a dish.")
            return
        if self.session.is_finished():
            self._log_action(
                f"Attempted to apply seasoning {display_name} after the run finished."
            )
            messagebox.showinfo("Run complete", "The run has finished. Start a new run to keep seasoning dishes.")
            return
        if not self.selected_indices:
            self._log_action(
                f"Attempted to apply seasoning {display_name} without selecting ingredients."
            )
            messagebox.showinfo(
                "No ingredients selected",
                "Select ingredients for the dish before adding seasonings.",
            )
            return

        current = self.applied_seasonings.get(seasoning.seasoning_id, 0)
        stack_limit = max(1, seasoning.stack_limit)
        if current >= stack_limit:
            self._log_action(
                f"Stack limit reached when applying seasoning {display_name}."
            )
            messagebox.showinfo(
                "Stack limit reached",
                f"{seasoning.display_name or seasoning.name} can only be applied {stack_limit} time(s) per dish.",
            )
            return
        charges = self.session.get_seasoning_charges(seasoning.seasoning_id)
        if charges is not None and current >= charges:
            self._log_action(
                f"No charges remaining when applying seasoning {display_name}."
            )
            messagebox.showinfo(
                "No charges remaining",
                f"{seasoning.display_name or seasoning.name} has no charges left for this run.",
            )
            return

        self.applied_seasonings[seasoning.seasoning_id] = current + 1
        self._update_seasoning_panels()
        self.update_selection_summary()
        new_total = self.applied_seasonings[seasoning.seasoning_id]
        charge_text = "∞" if charges is None else str(max(charges - current - 1, 0))
        self._log_action(
            f"Applied seasoning {display_name}. Count this dish: {new_total}. Remaining charges: {charge_text}."
        )

    def remove_applied_seasoning(self, seasoning_id: str) -> None:
        if seasoning_id not in self.applied_seasonings:
            self._log_action(
                f"Attempted to remove seasoning {seasoning_id} that was not applied."
            )
            return
        remaining = self.applied_seasonings.get(seasoning_id, 0)
        if remaining <= 1:
            self.applied_seasonings.pop(seasoning_id, None)
        else:
            self.applied_seasonings[seasoning_id] = remaining - 1
        self._update_seasoning_panels()
        self.update_selection_summary()
        seasoning_name = seasoning_id
        if self.session:
            seasoning = self.session.data.seasoning_by_id.get(seasoning_id)
            if seasoning:
                seasoning_name = seasoning.display_name or seasoning.name
        new_remaining = self.applied_seasonings.get(seasoning_id, 0)
        self._log_action(
            f"Removed seasoning {seasoning_name}. Remaining on dish: {new_remaining}."
        )

    def clear_applied_seasonings(self) -> None:
        if not self.applied_seasonings:
            self._log_action("Clear seasonings button pressed with no seasonings applied.")
            return
        self.applied_seasonings.clear()
        self._update_seasoning_panels()
        self.update_selection_summary()
        self._log_action("Cleared all applied seasonings from the dish preview.")

    def _update_chef_button(self) -> None:
        if not self.chef_button:
            return

        if not self.session:
            text = "Chefs\n0/0"
        else:
            count = len(self.session.chefs)
            max_slots = self.session.max_chefs
            if self.session.can_recruit_chef():
                text = "Chefs\nRecruit!"
            else:
                text = f"Chefs\n{count}/{max_slots}"

        icon = self._resource_button_images.get("chef")
        if icon is None:
            icon = _generate_button_icon(
                "chef", "CF", size=RESOURCE_BUTTON_ICON_PX
            )
            self._resource_button_images["chef"] = icon

        self.chef_button.configure(image=icon, text=text)

    def toggle_log_panel(self) -> None:
        if not self.log_text or not self.log_toggle_button:
            return
        self.log_collapsed = not self.log_collapsed
        if self.log_collapsed:
            self.log_text.grid_remove()
            self.log_toggle_button.configure(text="Show Log ▼")
        else:
            self.log_text.grid()
            self.log_toggle_button.configure(text="Hide Log ▲")

    def toggle_card(self, index: int) -> None:
        if not self.session:
            self._log_action(
                "Ingredient selection attempted without an active session."
            )
            return
        hand = self.session.get_hand()
        if index < 0 or index >= len(hand):
            self._log_action(
                f"Ingredient selection index {index} out of range for current hand."
            )
            return
        view = self.card_views.get(index)
        if not view:
            self._log_action(
                f"Ingredient view missing for index {index}; selection ignored."
            )
            return
        card = hand[index]
        ingredient = card.ingredient
        ingredient_name = getattr(ingredient, "display_name", None) or ingredient.name
        if index in self.selected_indices:
            self.selected_indices.remove(index)
            view.set_selected(False)
            self._log_action(f"Deselected ingredient: {ingredient_name}")
        else:
            if len(self.selected_indices) >= self.session.pick_size:
                messagebox.showinfo(
                    "Selection limit",
                    f"You may only pick {self.session.pick_size} cards per turn.",
                )
                self._log_action(
                    "Selection limit reached while choosing "
                    f"{ingredient_name}."
                )
                return
            self.selected_indices.add(index)
            view.set_selected(True)
            self._log_action(f"Selected ingredient: {ingredient_name}")
        self.update_selection_label()

    def _current_sort_mode(self) -> str:
        return self.hand_sort_modes[self.hand_sort_index]

    def _format_sort_label(self, mode: str) -> str:
        labels = {"name": "Name", "family": "Family", "taste": "Taste"}
        return f"Sort order: {labels.get(mode, mode.title())}"

    def cycle_hand_sort_mode(self) -> None:
        self.hand_sort_index = (self.hand_sort_index + 1) % len(self.hand_sort_modes)
        mode = self._current_sort_mode()
        self.hand_sort_var.set(self._format_sort_label(mode))
        self._log_action(f"Hand sort mode changed to {mode}.")
        self.render_hand()

    def update_selection_label(self) -> None:
        count = len(self.selected_indices)
        limit = self.session.pick_size if self.session else 0
        if limit:
            self.selection_var.set(f"Selection: {count} / {limit}")
        else:
            self.selection_var.set(f"Selection: {count}")
        self.update_selection_summary()

    def update_selection_summary(self) -> None:
        default_text = "Value × Dish = Score"
        default_estimate = "Estimated Score: —"
        if not self.session:
            self.selection_summary_var.set(default_text)
            self.estimated_score_var.set(default_estimate)
            self._update_seasoning_panels()
            return

        if not self.selected_indices:
            self.selection_summary_var.set(default_text)
            self.estimated_score_var.set(default_estimate)
            if self.applied_seasonings:
                self.applied_seasonings.clear()
            self._update_seasoning_panels()
            return

        hand_cards = self.session.get_hand()
        try:
            selected_cards = [hand_cards[index] for index in sorted(self.selected_indices)]
        except IndexError:
            self.selection_summary_var.set(default_text)
            self.estimated_score_var.set(default_estimate)
            self._update_seasoning_panels()
            return

        if any(card.is_rotten for card in selected_cards):
            self.selection_summary_var.set(default_text)
            self.estimated_score_var.set(default_estimate)
            self._update_seasoning_panels()
            return

        selected = [card.ingredient for card in selected_cards]

        Value = sum(ingredient.Value for ingredient in selected)
        dish_outcome = self.session.data.evaluate_dish(selected)
        multiplier = dish_outcome.dish_multiplier
        total = int(round(dish_outcome.dish_value))
        if dish_outcome.entry:
            summary = (
                f"Value {Value} × Dish {dish_outcome.entry.name} "
                f"({format_multiplier(multiplier)}) = {total}"
            )
        else:
            summary = (
                f"Value {Value} × Dish {format_multiplier(multiplier)} = {total}"
            )
        self.selection_summary_var.set(summary)

        self._update_seasoning_panels()
        self._update_estimated_score(selected, dish_outcome)

    def _update_estimated_score(
        self, selected: Sequence[Ingredient], dish_outcome: DishOutcome
    ) -> None:
        if not self.session:
            self.estimated_score_var.set("Estimated Score: —")
            return

        try:
            seasoning_calc = self.session.calculate_seasoning_adjustments(
                selected,
                dish_outcome.dish_value,
                self.applied_seasonings,
            )
        except ValueError as exc:
            self.estimated_score_var.set(f"Estimated Score: {exc}")
            return

        base_value = seasoning_calc.seasoned_score
        recipe_name = self.session.data.which_recipe(selected)
        recipe_multiplier = self.session.preview_recipe_multiplier(recipe_name)
        estimated_total = int(round(base_value * recipe_multiplier))

        multiplier_text = f"{recipe_multiplier:.2f}".rstrip("0").rstrip(".")
        if not multiplier_text:
            multiplier_text = "0"

        extras: List[str] = []
        boost_pct = int(round(seasoning_calc.total_boost_pct * 100))
        if boost_pct:
            extras.append(_format_percent(boost_pct))
        penalty_value = int(round(seasoning_calc.total_penalty))
        if penalty_value:
            extras.append(f"-{penalty_value}")
        extras_text = f" [{' ,'.join(extras)}]" if extras else ""
        ruined_text = " — Ruined" if seasoning_calc.ruined else ""

        self.estimated_score_var.set(
            f"Estimated Score: {estimated_total} "
            f"(Base {base_value} × Recipe {multiplier_text})"
            f"{extras_text}{ruined_text}"
        )

    def update_status(self) -> None:
        if not self.session:
            self._refresh_score_details()
            return
        round_text = f"Round {self.session.round_index}"
        if self.session.is_finished():
            turn_text = f"Turns completed: {self.session.turn_number}"
        else:
            turn_text = f"Next turn: {self.session.turn_number + 1}"
        self.progress_var.set(f"{round_text} — {turn_text}")
        self.score_var.set(str(self.session.get_total_score()))
        chef_names = ", ".join(chef.name for chef in self.session.chefs) or "None"
        self.chefs_var.set(
            f"Active chefs ({self.session.max_chefs} max): {chef_names}"
        )
        self._update_chef_button()
        self._update_seasoning_button()

    def _format_log_line(self, line: str) -> str:
        if not line.strip():
            return ""
        origin = self._log_start_time or self._app_launch_time
        elapsed = datetime.now() - origin
        total_seconds = elapsed.seconds + elapsed.days * 24 * 3600
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = elapsed.microseconds // 1000
        timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}]"
        return f"{timestamp} {line}"

    def _append_log_lines(self, lines: Iterable[str]) -> None:
        collected = list(lines)
        if not collected:
            return
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        for line in collected:
            formatted = self._format_log_line(line)
            if not formatted:
                continue
            self.log_text.insert("end", f"{formatted}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log_action(self, message: str) -> None:
        self._append_log_lines([message])

    def _log_run_settings(self) -> None:
        if not self.session:
            return
        seed_text = (
            str(self.session.seed)
            if getattr(self.session, "seed", None) is not None
            else "random"
        )
        chef_count = len(self.session.chefs)
        summary = (
            "Run settings — "
            f"Basket: {self.session.basket_name}; "
            f"Hand size: {self.session.hand_size}; "
            f"Pick size: {self.session.pick_size}; "
            f"Max chefs: {self.session.max_chefs}; "
            f"Starting chefs: {chef_count}; "
            f"RNG seed: {seed_text}"
        )
        position = (
            "Starting position — "
            f"Round {self.session.round_index}; "
            f"Next turn: {self.session.turn_number + 1}"
        )
        self._append_log_lines([summary, position])

    def append_events(self, messages: Iterable[str]) -> None:
        self._append_log_lines(messages)
        if not self.session:
            return
        if self.session.awaiting_new_round():
            summary = self.session.peek_basket_clear_summary() or {}
            if self._pending_round_summary is None:
                self._pending_round_summary = summary
                self._round_summary_shown = False
                self._round_reward_claimed = False
            else:
                self._pending_round_summary = summary or self._pending_round_summary
                if summary:
                    self._round_reward_claimed = False
            self._show_basket_clear_popup()
            return
        if not self.session.is_finished():
            self._set_action_buttons_enabled(True)
        if self.session.is_finished():
            self._handle_run_finished()

    def _show_round_summary_popup(self, summary: Mapping[str, object]) -> None:
        if not self.session:
            return
        if self.active_popup and self.active_popup.winfo_exists():
            popup_kind = getattr(self.active_popup, "_popup_kind", None)
            if popup_kind == "turn_summary":
                return
            if popup_kind == "round_summary":
                self.active_popup.lift()
                self.active_popup.focus_force()
                return
            self._destroy_active_popup()

        popup = tk.Toplevel(self.root)
        popup.title("Round Summary")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup._popup_kind = "round_summary"  # type: ignore[attr-defined]

        frame = ttk.Frame(popup, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        round_index = int(summary.get("round_index", self.session.round_index))
        round_points = int(summary.get("round_points", summary.get("round_score", 0)))
        total_score = int(summary.get("total_score", self.session.get_total_score()))
        dishes_cooked = int(summary.get("dishes_cooked", 0))
        recipes_completed = int(summary.get("recipes_completed", 0))
        unique_recipes = int(summary.get("unique_recipes", recipes_completed))
        rotten_used = int(summary.get("rotten_ingredients", 0))
        run_finished = bool(summary.get("run_finished", False))
        reason = str(summary.get("round_end_reason", "pantry_empty"))

        if run_finished:
            header_text = f"Round {round_index} complete"
        else:
            header_text = f"Round {round_index} ready"

        ttk.Label(
            frame,
            text=header_text,
            style="Header.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew")

        highlight_text = f"{round_points:+d} pts this round"
        highlight = tk.Label(
            frame,
            text=highlight_text,
            font=("Helvetica", 20, "bold"),
            fg="#1e2a33" if round_points >= 0 else "#611a15",
            bg="#ffe9a8" if round_points >= 0 else "#fdecea",
            padx=18,
            pady=10,
            anchor="center",
            justify="center",
        )
        highlight.grid(row=1, column=0, sticky="ew", pady=(8, 12))

        stats_lines = [
            f"Dishes cooked: {dishes_cooked}",
            f"Recipes completed: {recipes_completed} (unique {unique_recipes})",
            f"Rotten ingredients used: {rotten_used}",
            f"Total score so far: {total_score}",
        ]
        ttk.Label(
            frame,
            text="\n".join(stats_lines),
            justify="center",
            anchor="center",
        ).grid(row=2, column=0, sticky="ew")

        bonus_text: Optional[str]
        if run_finished:
            if reason == "target_reached":
                bonus_text = "Basket target reached! The run is complete."
            else:
                bonus_text = "Run complete."
        elif summary.get("bonus_choices_available", False):
            bonus_text = "Prize unlocked: Choose a bonus ingredient to start the next round."
        else:
            bonus_text = "Pantry will refill automatically for the next round."

        ttk.Label(
            frame,
            text=bonus_text,
            style="Info.TLabel",
            wraplength=420,
            justify="center",
        ).grid(row=3, column=0, sticky="ew", pady=(10, 16))

        ingredients_section = ttk.LabelFrame(
            frame, text="Ingredients cooked this round"
        )
        ingredients_section.grid(row=4, column=0, sticky="ew")
        ingredients_section.columnconfigure(0, weight=1)

        ingredient_entries = list(summary.get("ingredient_usage", []))
        image_refs: List[tk.PhotoImage] = []

        def resolve_ingredient(identifier: str, name: str) -> Optional[Ingredient]:
            if not self.session:
                return None
            ingredient = None
            if identifier:
                ingredient = self.session.data.ingredient_for_id(identifier)
            if not ingredient:
                lookup_name = name.strip()
                for candidate in self.session.data.ingredients.values():
                    display = getattr(candidate, "display_name", "") or candidate.name
                    if candidate.name == lookup_name or display == lookup_name:
                        ingredient = candidate
                        break
            return ingredient

        if ingredient_entries:
            grid = ttk.Frame(ingredients_section)
            grid.grid(row=0, column=0, sticky="ew", padx=4, pady=6)
            max_columns = 4
            for column in range(max_columns):
                grid.columnconfigure(column, weight=1)
            for index, entry in enumerate(ingredient_entries):
                identifier = str(entry.get("id", ""))
                name = str(entry.get("name", ""))
                count = int(entry.get("count", 0))
                ingredient = resolve_ingredient(identifier, name)
                card = ttk.Frame(grid, padding=6)
                row = index // max_columns
                column = index % max_columns
                card.grid(row=row, column=column, padx=4, pady=4, sticky="nsew")
                text = f"{name} ×{count}" if count else name
                if ingredient:
                    image = _load_ingredient_image(ingredient, target_px=72)
                else:
                    image = None
                if image is not None:
                    image_refs.append(image)
                    label = ttk.Label(
                        card,
                        image=image,
                        text=text,
                        compound="top",
                        justify="center",
                        wraplength=120,
                    )
                else:
                    label = ttk.Label(
                        card,
                        text=text,
                        justify="center",
                        wraplength=140,
                    )
                label.pack(anchor="center")
        else:
            ttk.Label(
                ingredients_section,
                text="No dishes were prepared this round.",
                style="Info.TLabel",
                anchor="center",
                justify="center",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=6)

        recipe_section = ttk.LabelFrame(frame, text="Recipes cooked")
        recipe_section.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        recipe_section.columnconfigure(0, weight=1)

        recipe_entries = list(summary.get("recipe_usage", []))
        if recipe_entries:
            lines = [
                f"• {str(entry.get('display') or entry.get('name'))} ×{int(entry.get('count', 0))}"
                for entry in recipe_entries
            ]
            ttk.Label(
                recipe_section,
                text="\n".join(lines),
                justify="left",
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=6)
        else:
            ttk.Label(
                recipe_section,
                text="No recipes completed this round.",
                style="Info.TLabel",
                anchor="center",
                justify="center",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=6)

        dish_section = ttk.LabelFrame(frame, text="Dish classifications")
        dish_section.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        dish_section.columnconfigure(0, weight=1)

        dish_entries = list(summary.get("dish_usage", []))
        if dish_entries:
            lines = []
            for entry in dish_entries:
                label = str(entry.get("label", "Dish"))
                count = int(entry.get("count", 0))
                family = str(entry.get("family_label", "")).strip()
                flavor = str(entry.get("flavor_label", "")).strip()
                details = " / ".join(
                    part for part in (family, flavor) if part
                )
                if details:
                    lines.append(f"• {label} ×{count} ({details})")
                else:
                    lines.append(f"• {label} ×{count}")
            ttk.Label(
                dish_section,
                text="\n".join(lines),
                justify="left",
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=6)
        else:
            ttk.Label(
                dish_section,
                text="No dish matrix bonuses recorded this round.",
                style="Info.TLabel",
                anchor="center",
                justify="center",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=6)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=7, column=0, sticky="e", pady=(16, 0))

        def close_popup() -> None:
            if popup.winfo_exists():
                try:
                    popup.grab_release()
                except tk.TclError:
                    pass
                popup.destroy()
            if self.active_popup is popup:
                self.active_popup = None
            self._round_summary_shown = True
            popup.after(50, self._show_basket_clear_popup)

        ttk.Button(button_frame, text="OK", command=close_popup, width=12).grid(
            row=0, column=0, sticky="e"
        )

        popup.bind("<Escape>", lambda _e: close_popup())
        popup.bind("<Return>", lambda _e: close_popup())
        popup.protocol("WM_DELETE_WINDOW", close_popup)

        popup._image_refs = image_refs  # type: ignore[attr-defined]
        popup.grab_set()
        self.active_popup = popup
        self._center_popup(popup)
        popup.focus_force()

    def _show_basket_clear_popup(self) -> None:
        if self._deferring_round_summary:
            return
        if not self.session:
            return

        if self.active_popup and self.active_popup.winfo_exists():
            popup_kind = getattr(self.active_popup, "_popup_kind", None)
            if popup_kind == "turn_summary":
                return
            if (
                self._pending_challenge_reward_score is not None
                and popup_kind == "challenge_reward"
            ):
                self.active_popup.lift()
                self.active_popup.focus_force()
                return

        if self._pending_challenge_reward_score is not None:
            score = self._pending_challenge_reward_score
            self._pending_challenge_reward_score = None
            self._present_challenge_reward(score)
            return

        summary = self._pending_round_summary or {}
        if not summary:
            summary = self.session.peek_basket_clear_summary() or {}
            self._pending_round_summary = summary or self._pending_round_summary

        if not summary:
            if (
                self.session
                and not self.session.awaiting_new_round()
                and not self.session.is_finished()
            ):
                self._set_action_buttons_enabled(True)
            return

        self._pending_round_summary = summary
        run_finished = bool(summary.get("run_finished", False))

        if run_finished:
            if not self._round_summary_shown:
                self._set_action_buttons_enabled(False)
                if (
                    self.active_popup
                    and self.active_popup.winfo_exists()
                    and getattr(self.active_popup, "_popup_kind", None) not in {"round_summary", "turn_summary"}
                ):
                    self._destroy_active_popup()
                self._show_round_summary_popup(summary)
                return

            if self.session.needs_cleanup_confirmation():
                self._set_action_buttons_enabled(False)
                if (
                    self.active_popup
                    and self.active_popup.winfo_exists()
                    and getattr(self.active_popup, "_popup_kind", None) == "cleanup"
                ):
                    self.active_popup.lift()
                    self.active_popup.focus_force()
                    return
                if (
                    self.active_popup
                    and self.active_popup.winfo_exists()
                    and getattr(self.active_popup, "_popup_kind", None) not in {"turn_summary"}
                ):
                    self._destroy_active_popup()
                self._show_cleanup_popup()
                return

            self._pending_round_summary = None
            self._round_summary_shown = False
            self._round_reward_claimed = False
            self._handle_run_finished()
            return

        if self.session.needs_cleanup_confirmation():
            self._set_action_buttons_enabled(False)
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) == "cleanup"
            ):
                self.active_popup.lift()
                self.active_popup.focus_force()
                return
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) not in {"turn_summary"}
            ):
                self._destroy_active_popup()
            self._show_cleanup_popup()
            return

        if not self._round_reward_claimed:
            self._set_action_buttons_enabled(False)
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) == "basket_clear"
            ):
                self.active_popup.lift()
                self.active_popup.focus_force()
                return
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) not in {"turn_summary"}
            ):
                self._destroy_active_popup()
            self._show_round_reward_popup(summary)
            return

        if not self._round_summary_shown:
            self._set_action_buttons_enabled(False)
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) == "round_summary"
            ):
                self.active_popup.lift()
                self.active_popup.focus_force()
                return
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) not in {"turn_summary"}
            ):
                self._destroy_active_popup()
            self._show_round_summary_popup(summary)
            return

        if self.session.needs_cleanup_confirmation():
            self._set_action_buttons_enabled(False)
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) == "cleanup"
            ):
                self.active_popup.lift()
                self.active_popup.focus_force()
                return
            if (
                self.active_popup
                and self.active_popup.winfo_exists()
                and getattr(self.active_popup, "_popup_kind", None) not in {"turn_summary"}
            ):
                self._destroy_active_popup()
            self._show_cleanup_popup()
            return

        self._pending_round_summary = None
        self._round_summary_shown = False
        self._round_reward_claimed = False
        if not self.session.is_finished():
            self._set_action_buttons_enabled(True)
        else:
            self._handle_run_finished()

    def _show_cleanup_popup(self) -> None:
        if not self.session:
            return

        cards = list(self.session.pending_cleanup_ingredients())
        if not cards:
            try:
                self.session.acknowledge_cleanup()
            except Exception as exc:  # pragma: no cover - user feedback path
                messagebox.showerror("Unable to confirm cleanup", str(exc))
                return
            self.append_events(self.session.consume_events())
            self._set_action_buttons_enabled(False)
            self.root.after(50, self._show_basket_clear_popup)
            return

        if (
            self.active_popup
            and self.active_popup.winfo_exists()
            and getattr(self.active_popup, "_popup_kind", None) not in {"turn_summary"}
        ):
            self._destroy_active_popup()

        popup = tk.Toplevel(self.root)
        popup.title("Clean Up Pantry")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup._popup_kind = "cleanup"  # type: ignore[attr-defined]

        frame = ttk.Frame(popup, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame,
            text="Clean up pantry from rotten ingredients.",
            style="Header.TLabel",
            justify="center",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")

        counts = Counter(
            getattr(ingredient, "display_name", None) or ingredient.name
            for ingredient in cards
        )
        lines = []
        for name, count in sorted(counts.items(), key=lambda item: item[0].lower()):
            if count > 1:
                lines.append(f"• {name} ×{count}")
            else:
                lines.append(f"• {name}")
        list_text = "\n".join(lines)

        ttk.Label(
            frame,
            text=list_text,
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(12, 8))

        def confirm_cleanup() -> None:
            if not self.session:
                return
            try:
                removed = self.session.acknowledge_cleanup()
            except Exception as exc:  # pragma: no cover - user feedback path
                messagebox.showerror("Unable to confirm cleanup", str(exc))
                return

            if popup.winfo_exists():
                try:
                    popup.grab_release()
                except tk.TclError:
                    pass
                popup.destroy()
            if self.active_popup is popup:
                self.active_popup = None

            removed_names = [
                getattr(ingredient, "display_name", None) or ingredient.name
                for ingredient in removed
            ]
            if removed_names:
                message = (
                    "Removed rotten ingredients from the pantry: "
                    + ", ".join(removed_names)
                    + "."
                )
            else:
                message = "Pantry contained no rotten ingredients to remove."
            self.write_result(message)
            self.append_events(self.session.consume_events())
            self._set_action_buttons_enabled(False)
            self.root.after(50, self._show_basket_clear_popup)

        button = ttk.Button(frame, text="OK", command=confirm_cleanup)
        button.grid(row=2, column=0, sticky="e", pady=(8, 0))

        popup.protocol("WM_DELETE_WINDOW", confirm_cleanup)
        popup.bind("<Return>", lambda _e: confirm_cleanup())
        popup.bind("<Escape>", lambda _e: confirm_cleanup())

        popup.update_idletasks()
        self._center_popup(popup)
        popup.grab_set()
        popup.focus_force()
        button.focus_set()
        self.active_popup = popup

    def _show_round_reward_popup(self, summary: Mapping[str, object]) -> None:
        if not self.session:
            return

        if self.active_popup and self.active_popup.winfo_exists():
            popup_kind = getattr(self.active_popup, "_popup_kind", None)
            if popup_kind == "basket_clear":
                self.active_popup.lift()
                self.active_popup.focus_force()
                return
            if popup_kind != "turn_summary":
                self._destroy_active_popup()

        popup = tk.Toplevel(self.root)
        popup.title("Round Reward")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup._popup_kind = "basket_clear"  # type: ignore[attr-defined]

        frame = ttk.Frame(popup, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        round_index = int(summary.get("round_index", self.session.round_index))
        round_points = int(summary.get("round_points", summary.get("round_score", 0)))
        total_score = int(summary.get("total_score", self.session.get_total_score()))
        dishes_cooked = int(summary.get("dishes_cooked", 0))
        rotten_used = int(summary.get("rotten_ingredients", 0))
        recipes_completed = int(summary.get("recipes_completed", 0))
        unique_recipes = int(summary.get("unique_recipes", recipes_completed))

        header_text = "Round complete! Claim your reward."
        stats_lines = [
            f"Round {round_index} highlights:",
            f"  • Dishes cooked: {dishes_cooked}",
            f"  • Recipes completed: {recipes_completed} (unique {unique_recipes})",
            f"  • Rotten ingredients used: {rotten_used}",
            f"  • Points earned this round: {round_points}",
            f"Total score so far: {total_score}",
        ]
        summary_text = "\n".join(stats_lines)

        ttk.Label(
            frame,
            text=header_text,
            style="Header.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(
            frame,
            text=summary_text,
            justify="center",
            anchor="center",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 12))

        choices = list(self.session.get_basket_bonus_choices())
        if not choices:
            ingredients = list(self.session.data.ingredients.values())
            if ingredients:
                sample_count = min(3, len(ingredients))
                choices = random.sample(ingredients, sample_count)

        instruction_text = (
            "Pick an ingredient to jump-start the next round:"
            if choices
            else "No ingredient bonus is available this time."
        )
        ttk.Label(
            frame,
            text=instruction_text,
            justify="center",
            anchor="center",
        ).grid(row=2, column=0, sticky="ew")

        reward_frame = ttk.Frame(frame)
        reward_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        reward_frame.columnconfigure(0, weight=1)

        image_refs: List[tk.PhotoImage] = []

        def finalize_reward() -> None:
            if not self.session:
                return
            self.selected_indices.clear()
            self.update_selection_label()
            if self.applied_seasonings:
                self.applied_seasonings.clear()
            self.estimated_score_var.set("Estimated Score: —")

            if popup.winfo_exists():
                popup.grab_release()
                popup.destroy()
            if self.active_popup is popup:
                self.active_popup = None

            self._round_reward_claimed = True

            self.render_hand()
            self.update_status()
            self._update_seasoning_panels()
            if self.team_tile:
                self.team_tile.set_team(self.session.chefs, self.session.max_chefs)
            if self.seasoning_tile:
                self.seasoning_tile.set_seasonings(
                    self.session.get_seasonings(),
                    len(self.session.available_seasonings()),
                )
            self._update_chef_button()
            self.append_events(self.session.consume_events())
            self._set_action_buttons_enabled(False)

            popup.after(50, self._show_basket_clear_popup)

        def handle_ingredient_choice(ingredient: Ingredient) -> None:
            if not self.session:
                return
            try:
                self.session.begin_next_round_from_empty_basket(ingredient)
            except Exception as exc:  # pragma: no cover - user feedback path
                messagebox.showerror("Unable to start new round", str(exc))
                return
            finalize_reward()

        if choices:
            ingredient_frame = ttk.LabelFrame(reward_frame, text="Add an ingredient")
            ingredient_frame.grid(row=0, column=0, sticky="ew")
            for column in range(len(choices)):
                ingredient_frame.columnconfigure(column, weight=1)
            for column, ingredient in enumerate(choices):
                image = _load_ingredient_image(ingredient, target_px=96)
                image_refs.append(image)
                display_name = getattr(ingredient, "display_name", None) or ingredient.name
                button = ttk.Button(
                    ingredient_frame,
                    text=display_name,
                    image=image,
                    compound="top",
                    command=lambda ing=ingredient: handle_ingredient_choice(ing),
                    width=20,
                )
                button.image = image  # type: ignore[attr-defined]
                button.grid(row=0, column=column, padx=6, pady=4)

        if not choices:
            def continue_without_reward() -> None:
                if not self.session:
                    return
                try:
                    self.session.begin_next_round_after_reward()
                except Exception as exc:  # pragma: no cover - user feedback path
                    messagebox.showerror("Unable to start new round", str(exc))
                    return
                finalize_reward()

            ttk.Button(
                reward_frame,
                text="Continue",
                command=continue_without_reward,
            ).grid(row=0, column=0, pady=6)

        popup.protocol("WM_DELETE_WINDOW", lambda: None)
        popup.bind("<Escape>", lambda _e: None)
        popup.grab_set()
        popup._image_refs = image_refs  # type: ignore[attr-defined]
        self.active_popup = popup
        self._center_popup(popup)
        popup.focus_force()

    def clear_events(self) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _handle_run_finished(
        self,
        *,
        custom_message: Optional[Tuple[str, str]] = None,
        show_dialog: bool = True,
    ) -> None:
        if not self.session:
            return

        self.pantry_card_ids = self.session.get_pantry_card_ids()
        self._persistent_cookbook = self.session.get_cookbook()

        self._set_action_buttons_enabled(False)
        self._set_controls_active(True)
        self._close_recruit_dialog()

        if custom_message is None and self._run_completion_notified and show_dialog:
            return

        title: str
        message: str
        if custom_message is not None:
            title, message = custom_message
        else:
            title = "Run complete"
            message = f"Final score: {self.session.get_total_score()}"

        self._run_completion_notified = True
        if show_dialog and custom_message is not None:
            messagebox.showinfo(title, message)
        summary_text = self._final_summary_text()
        if summary_text:
            self.write_result(summary_text)

    def _close_recruit_dialog(self) -> None:
        if self.recruit_dialog and self.recruit_dialog.winfo_exists():
            self.recruit_dialog.destroy()
        self.recruit_dialog = None

    def _destroy_active_popup(self) -> None:
        if self.active_popup and self.active_popup.winfo_exists():
            try:
                self.active_popup.grab_release()
            except tk.TclError:
                pass
            self.active_popup.destroy()
        self.active_popup = None

    def log_turn_points(self, outcome: TurnOutcome) -> None:
        notes: List[str] = []
        if outcome.dish_name:
            tier_text = f" {outcome.dish_tier}" if outcome.dish_tier else ""
            notes.append(
                f"dish {outcome.dish_name}{tier_text}"
                f" {format_multiplier(outcome.dish_multiplier)}"
            )
        else:
            notes.append(f"dish {format_multiplier(outcome.dish_multiplier)}")

        recipe_display = outcome.recipe_display_name or outcome.recipe_name
        if outcome.recipe_name:
            parts = [
                f"recipe {recipe_display} x{outcome.recipe_multiplier:.2f}"
            ]
            if outcome.discovered_recipe:
                if outcome.personal_discovery:
                    parts.append("personal discovery")
                else:
                    parts.append("new recipe")
            if outcome.times_cooked_total:
                parts.append(f"total cooks {outcome.times_cooked_total}")
            notes.append("; ".join(parts))

        if outcome.applied_seasonings:
            boost_pct = int(round(outcome.seasoning_boost_pct * 100))
            penalty_value = int(round(outcome.seasoning_penalty))
            seasoning_parts = [
                f"seasonings {outcome.base_score}->{outcome.seasoned_score}"
            ]
            if boost_pct:
                seasoning_parts.append(_format_percent(boost_pct))
            if penalty_value:
                seasoning_parts.append(f"-{penalty_value}")
            if self.session:
                applied = []
                for seasoning_id, count in outcome.applied_seasonings:
                    seasoning = self.session.data.seasoning_by_id.get(seasoning_id)
                    if seasoning:
                        display = seasoning.display_name or seasoning.name
                    else:
                        display = seasoning_id
                    applied.append(f"{display}×{count}")
                if applied:
                    seasoning_parts.append(", ".join(applied))
            notes.append("; ".join(seasoning_parts))

        note_text = f" ({'; '.join(notes)})" if notes else ""
        entry = (
            f"Turn {outcome.turn_number} {outcome.final_score:+d} pts{note_text}"
        )
        lines = [entry]
        if outcome.alerts:
            lines.extend(f"    ⚠️ {alert}" for alert in outcome.alerts)
        self._append_log_lines(lines)

    def _cookbook_summary_text(self) -> str:
        if not self.session:
            return ""
        entries = self.session.get_cookbook()
        if not entries:
            return "Cookbook:\n  (No recipes discovered yet.)"
        lines = []
        sorted_entries = sorted(
            entries.values(), key=lambda entry: entry.display_name.lower()
        )
        for entry in sorted_entries:
            times = "time" if entry.count == 1 else "times"
            lines.append(
                f"  {entry.display_name} — {format_multiplier(entry.multiplier)}; "
                f"cooked {entry.count} {times}: {', '.join(entry.ingredients)}"
            )
        return "Cookbook:\n" + "\n".join(lines)

    def _final_summary_text(self) -> str:
        if not self.session:
            return ""
        total = self.session.get_total_score()
        return f"Run complete!\nFinal score: {total}"

    def write_result(self, text: str) -> None:
        extra = self._cookbook_summary_text()
        lines: List[str] = []
        has_existing_text = False
        if self.log_text is not None:
            try:
                has_existing_text = self.log_text.index("end-1c") != "1.0"
            except tk.TclError:  # pragma: no cover - defensive guard
                has_existing_text = False
        if has_existing_text:
            lines.append("")
        lines.extend(text.splitlines())
        if extra:
            lines.append("")
            lines.extend(extra.splitlines())
        self._append_log_lines(lines)

    # ----------------- Gameplay actions -----------------
    def cook_selected(self) -> None:
        if not self.session:
            self._log_action("Cook button pressed without an active session.")
            return
        if not self.selected_indices:
            self._log_action("Cook button pressed with no ingredients selected.")
            messagebox.showwarning(
                "Incomplete selection",
                "Select at least one ingredient before cooking.",
            )
            return

        hand_cards = list(self.session.get_hand())
        indices = sorted(self.selected_indices)
        selected_names: List[str] = []
        for index in indices:
            if 0 <= index < len(hand_cards):
                ingredient = hand_cards[index].ingredient
                selected_names.append(
                    getattr(ingredient, "display_name", None) or ingredient.name
                )
        if self.applied_seasonings:
            seasoning_descriptions: List[str] = []
            for seasoning_id, count in sorted(self.applied_seasonings.items()):
                seasoning_name = seasoning_id
                if self.session:
                    seasoning = self.session.data.seasoning_by_id.get(seasoning_id)
                    if seasoning:
                        seasoning_name = seasoning.display_name or seasoning.name
                seasoning_descriptions.append(f"{seasoning_name}×{count}")
            seasonings_text = ", ".join(seasoning_descriptions)
        else:
            seasonings_text = "none"
        ingredients_text = ", ".join(selected_names) if selected_names else "none"
        self._log_action(
            "Cook button pressed with ingredients: "
            f"{ingredients_text}; seasonings: {seasonings_text}."
        )

        try:
            outcome = self.session.play_turn(indices, self.applied_seasonings)
        except InvalidDishSelection as exc:
            self.selected_indices.clear()
            self.update_selection_label()
            self.estimated_score_var.set("Estimated Score: —")
            self.render_hand()
            self.update_status()
            self.append_events(self.session.consume_events())
            self.write_result(str(exc))
            self._show_invalid_dish_popup(exc)
            self._log_action(f"Cook action failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - user feedback path
            self._log_action(f"Cook action failed with error: {exc}")
            messagebox.showerror("Unable to cook selection", str(exc))
            return

        self.selected_indices.clear()
        if self.applied_seasonings:
            self.applied_seasonings.clear()
        self.estimated_score_var.set("Estimated Score: —")
        self.update_selection_label()
        self._update_seasoning_panels()

        summary = self._format_outcome(outcome)
        self.write_result(summary)

        cookbook_entries = self.session.get_cookbook()
        self._persistent_cookbook = cookbook_entries
        if self.cookbook_tile:
            self.cookbook_tile.set_entries(cookbook_entries)
        self._update_cookbook_button()
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.set_entries(cookbook_entries)

        self.render_hand()
        self.update_status()
        self.log_turn_points(outcome)
        self._deferring_round_summary = True
        self.append_events(self.session.consume_events())
        self._deferring_round_summary = False
        self.show_turn_summary_popup(outcome)
        self.maybe_prompt_new_chef()
        self._lifetime_total_score += outcome.final_score
        self._refresh_score_details()
        self._check_challenge_completion()

        if self.session.is_finished():
            self._handle_run_finished()

    def _show_invalid_dish_popup(self, exc: InvalidDishSelection) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("No Dish Formed")
        popup.transient(self.root)
        popup.resizable(False, False)

        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        content = ttk.Frame(popup, padding=16)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)

        displayed_ingredients = list(getattr(exc, "ingredients", ()))
        message_row = 0

        if displayed_ingredients:
            cards_frame = ttk.Frame(content)
            cards_frame.grid(row=0, column=0, sticky="w", pady=(0, 12))
            for column, ingredient in enumerate(displayed_ingredients):
                holder = ttk.Frame(cards_frame)
                holder.grid(row=0, column=column, padx=(0 if column == 0 else 12, 0))

                image = _load_ingredient_image(ingredient, target_px=96)
                image_label = tk.Label(
                    holder,
                    image=image,
                    background="#f4ebd0",
                    borderwidth=1,
                    relief="solid",
                )
                image_label.image = image  # type: ignore[attr-defined]
                image_label.grid(row=0, column=0)

                name_text = getattr(ingredient, "display_name", None) or ingredient.name
                name_label = ttk.Label(
                    holder,
                    text=name_text,
                    style="Header.TLabel",
                    anchor="center",
                    wraplength=120,
                    justify="center",
                )
                name_label.grid(row=1, column=0, pady=(6, 0))

            message_row = 1
        else:
            ingredient = getattr(exc, "primary_ingredient", None)
            if ingredient is not None:
                single_frame = ttk.Frame(content)
                single_frame.grid(row=0, column=0, sticky="w", pady=(0, 12))
                image = _load_ingredient_image(ingredient, target_px=112)
                image_label = tk.Label(
                    single_frame,
                    image=image,
                    background="#f4ebd0",
                    borderwidth=1,
                    relief="solid",
                )
                image_label.image = image  # type: ignore[attr-defined]
                image_label.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 12))

                name_text = getattr(ingredient, "display_name", None) or ingredient.name
                name_label = ttk.Label(
                    single_frame,
                    text=name_text,
                    style="Header.TLabel",
                    anchor="w",
                )
                name_label.grid(row=0, column=1, sticky="w")

                message_row = 1

        message_label = ttk.Label(
            content,
            text=str(exc),
            style="Info.TLabel",
            wraplength=360,
            justify="left",
        )
        message_label.grid(row=message_row, column=0, sticky="w")

        button = ttk.Button(content, text="OK", command=popup.destroy)
        button.grid(row=message_row + 1, column=0, sticky="e", pady=(12, 0))

        popup.bind("<Return>", lambda _e: popup.destroy())
        popup.bind("<Escape>", lambda _e: popup.destroy())

        popup.update_idletasks()
        self._center_popup(popup)
        popup.grab_set()
        popup.focus_force()
        button.focus_set()

    def return_selected(self) -> None:
        if not self.session:
            self._log_action("Return button pressed without an active session.")
            return
        if not self.selected_indices:
            self._log_action("Return button pressed with no ingredients selected.")
            messagebox.showwarning(
                "No selection",
                "Select at least one ingredient to return.",
            )
            return

        indices = sorted(self.selected_indices)
        hand_cards = list(self.session.get_hand()) if self.session else []
        selected_names: List[str] = []
        for index in indices:
            if 0 <= index < len(hand_cards):
                ingredient = hand_cards[index].ingredient
                selected_names.append(
                    getattr(ingredient, "display_name", None) or ingredient.name
                )
        ingredients_text = ", ".join(selected_names) if selected_names else "none"
        self._log_action(f"Return button pressed with ingredients: {ingredients_text}.")
        try:
            removed, deck_refreshed = self.session.return_indices(indices)
        except Exception as exc:  # pragma: no cover - user feedback path
            self._log_action(f"Return action failed with error: {exc}")
            messagebox.showerror("Unable to return ingredient", str(exc))
            return

        self.selected_indices.clear()
        self.update_selection_label()
        if self.applied_seasonings:
            self.applied_seasonings.clear()
        self.estimated_score_var.set("Estimated Score: —")
        self._update_seasoning_panels()
        self.render_hand()
        self.update_status()

        events = self.session.consume_events()
        self.append_events(events)

        if removed:
            names = ", ".join(ingredient.name for ingredient in removed)
            if deck_refreshed:
                message = f"Returned {names}. Market pantry refreshed."
            else:
                replacement_text = "replacements" if len(removed) > 1 else "a replacement"
                message = f"Returned {names} to the pantry and drew {replacement_text}."
        else:
            message = "No ingredient was returned."
        self.write_result(message)
        if removed:
            self._log_action(
                f"Returned ingredients resolved: {', '.join(ingredient.name for ingredient in removed)}."
            )
        else:
            self._log_action("Return action completed with no ingredients removed.")

        if self.session.is_finished():
            self._handle_run_finished()

    def show_turn_summary_popup(self, outcome: TurnOutcome) -> None:
        self._destroy_active_popup()

        # Prevent additional turn actions until the summary is acknowledged.
        self._set_action_buttons_enabled(False)

        popup = tk.Toplevel(self.root)
        popup.title("Turn Summary")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup.configure(bg="#10151a")
        popup._popup_kind = "turn_summary"  # type: ignore[attr-defined]

        def close_popup() -> None:
            if popup.winfo_exists():
                try:
                    popup.grab_release()
                except tk.TclError:
                    pass
                popup.destroy()
            if self.active_popup is popup:
                self.active_popup = None
            pending_followups = bool(
                self._pending_challenge_reward_score is not None
                or self._pending_round_summary
                or (
                    self.session
                    and self.session.awaiting_new_round()
                )
            )
            if pending_followups:
                # Prevent the player from queuing another action while follow-up
                # dialogs prepare to display.
                self._set_action_buttons_enabled(False)
                popup.after(50, self._show_basket_clear_popup)
            elif (
                self.session
                and not self.session.awaiting_new_round()
                and not self.session.is_finished()
            ):
                # No additional dialogs will appear, so restore the cook button
                # immediately.
                self._set_action_buttons_enabled(True)

        popup.protocol("WM_DELETE_WINDOW", close_popup)
        popup.bind("<Escape>", lambda _e: close_popup())

        glow_frame = tk.Frame(popup, bg="#edf1f7", padx=12, pady=12)
        glow_frame.pack(fill="both", expand=True)

        base_bg = "#ffffff"
        content = tk.Frame(glow_frame, bg=base_bg, padx=18, pady=16)
        content.pack(fill="both", expand=True)

        heading = tk.Label(
            content,
            text=(
                f"Turn {outcome.turn_number}"
                f"  •  Round {outcome.round_index}"
            ),
            font=("Helvetica", 15, "bold"),
            fg="#15202b",
            bg=base_bg,
            justify="left",
            anchor="w",
        )
        heading.pack(anchor="w", pady=(0, 8))

        score_change = outcome.final_score
        score_text = f"{score_change:+d} pts this turn"
        highlight_bg = "#ffe9a8"
        highlight_fg = "#1e2a33"
        if score_change < 0:
            highlight_bg = "#fdecea"
            highlight_fg = "#611a15"

        score_highlight = tk.Label(
            content,
            text=score_text,
            font=("Helvetica", 22, "bold"),
            fg=highlight_fg,
            bg=highlight_bg,
            padx=24,
            pady=16,
            anchor="center",
            justify="center",
        )
        score_highlight.pack(fill="x", pady=(0, 12))

        indicator_text: Optional[str] = None
        indicator_bg = "#e8f5e9"
        indicator_fg = "#1e4520"
        if outcome.dish_name:
            multiplier_value = f"{outcome.dish_multiplier:.2f}".rstrip("0").rstrip(".")
            indicator_text = f'"{outcome.dish_name}" x {multiplier_value}'
        elif not math.isclose(outcome.dish_multiplier, 1.0):
            indicator_text = (
                f"Dish multiplier applied: {format_multiplier(outcome.dish_multiplier)}"
            )
            if outcome.dish_multiplier < 1.0:
                indicator_bg = "#fdecea"
                indicator_fg = "#611a15"

        if indicator_text:
            indicator_container = tk.Frame(
                content,
                bg=indicator_bg,
                padx=16,
                pady=10,
            )
            indicator_container.pack(fill="x", pady=(0, 12))
            tk.Label(
                indicator_container,
                text=indicator_text,
                font=("Helvetica", 13, "bold"),
                fg=indicator_fg,
                bg=indicator_bg,
                anchor="w",
                justify="left",
            ).pack(anchor="w")

        if outcome.alerts:
            alert_container = tk.Frame(
                content,
                bg="#fdecea",
                padx=16,
                pady=12,
            )
            alert_container.pack(fill="x", pady=(0, 12))
            tk.Label(
                alert_container,
                text="Alerts:",
                font=("Helvetica", 12, "bold"),
                fg="#611a15",
                bg="#fdecea",
                anchor="w",
                justify="left",
            ).pack(anchor="w")
            for alert in outcome.alerts:
                tk.Label(
                    alert_container,
                    text=f"⚠️ {alert}",
                    font=("Helvetica", 10),
                    fg="#611a15",
                    bg="#fdecea",
                    wraplength=360,
                    justify="left",
                    anchor="w",
                ).pack(anchor="w", pady=(2, 0))

        if outcome.recipe_name:
            recipe_display = outcome.recipe_display_name or outcome.recipe_name
            if outcome.discovered_recipe and outcome.personal_discovery:
                banner_text = (
                    f"✨ Personal discovery: {recipe_display}! ✨"
                    " Added to your cookbook."
                )
            else:
                banner_text = f"✨ Recipe completed: {recipe_display}! ✨"
                if outcome.discovered_recipe:
                    banner_text += " Added to your cookbook."
            recipe_banner = tk.Label(
                content,
                text=banner_text,
                font=("Helvetica", 12, "bold"),
                fg="#a35d00",
                bg=base_bg,
                anchor="w",
                justify="left",
            )
            recipe_banner.pack(anchor="w", pady=(0, 8))

            recipe_image = _load_recipe_image(
                outcome.recipe_name,
                recipe_display,
                target_px=RECIPE_IMAGE_TARGET_PX,
            )
            if recipe_image is not None:
                image_container = tk.Frame(content, bg=base_bg)
                image_container.pack(anchor="center", pady=(0, 12))
                image_label = tk.Label(
                    image_container,
                    image=recipe_image,
                    bg=base_bg,
                    bd=0,
                )
                image_label.image = recipe_image
                image_label.pack(anchor="center")

        tk.Label(
            content,
            text="Cooked ingredients:",
            font=("Helvetica", 11, "bold"),
            fg="#1d2730",
            bg=base_bg,
            anchor="w",
        ).pack(anchor="w")

        ingredients_frame = tk.Frame(content, bg=base_bg)
        ingredients_frame.pack(anchor="w", fill="x", pady=(0, 10))

        if outcome.selected:
            for ingredient in outcome.selected:
                row = tk.Frame(ingredients_frame, bg=base_bg)
                row.pack(anchor="w", fill="x", pady=(0, 4))

                icon_cluster = tk.Frame(row, bg=base_bg)
                icon_cluster.pack(side="left", padx=(0, 12))

                ingredient_icon = _load_ingredient_image(
                    ingredient, target_px=INGREDIENT_DIALOG_ICON_PX
                )
                ingredient_label = tk.Label(
                    icon_cluster, image=ingredient_icon, bg=base_bg
                )
                ingredient_label.image = ingredient_icon
                ingredient_label.pack(side="left")

                meta_icon_column: Optional[tk.Frame] = None

                def ensure_meta_column() -> tk.Frame:
                    nonlocal meta_icon_column
                    if meta_icon_column is None:
                        meta_icon_column = tk.Frame(icon_cluster, bg=base_bg)
                        meta_icon_column.pack(side="left", padx=(8, 0))
                    return meta_icon_column

                family_icon = _load_icon(
                    "family", ingredient.family, target_px=DIALOG_ICON_TARGET_PX
                )
                if family_icon is not None:
                    family_label = tk.Label(
                        ensure_meta_column(), image=family_icon, bg=base_bg
                    )
                    family_label.image = family_icon
                    family_label.pack(side="top")

                taste_icon = _load_icon(
                    "taste", ingredient.taste, target_px=DIALOG_ICON_TARGET_PX
                )
                if taste_icon is not None:
                    taste_label = tk.Label(
                        ensure_meta_column(), image=taste_icon, bg=base_bg
                    )
                    taste_label.image = taste_icon
                    taste_label.pack(side="top", pady=(4, 0))

                text_frame = tk.Frame(row, bg=base_bg)
                text_frame.pack(side="left", fill="x", expand=True)

                tk.Label(
                    text_frame,
                    text=f"{ingredient.name} — Taste {ingredient.taste}, Value {ingredient.Value}",
                    font=("Helvetica", 10, "bold"),
                    fg="#1d2730",
                    bg=base_bg,
                    anchor="w",
                    justify="left",
                ).pack(anchor="w")

                tk.Label(
                    text_frame,
                    text=f"Family: {ingredient.family}",
                    font=("Helvetica", 10),
                    fg="#26313a",
                    bg=base_bg,
                    anchor="w",
                    justify="left",
                ).pack(anchor="w")
        else:
            tk.Label(
                ingredients_frame,
                text="• —",
                font=("Helvetica", 10),
                fg="#26313a",
                bg=base_bg,
                anchor="w",
                justify="left",
            ).pack(anchor="w")

        family_desc = outcome.family_pattern.replace("_", " ")
        points_lines = [
            f"Ingredient Value: {outcome.Value}",
            f"Family profile: {outcome.family_label} ({family_desc})",
        ]
        if outcome.dish_name:
            tier_text = f" [{outcome.dish_tier}]" if outcome.dish_tier else ""
            points_lines.append(
                "Dish classification: "
                f"{outcome.dish_name}{tier_text} — Dish multiplier "
                f"{format_multiplier(outcome.dish_multiplier)}"
            )
        else:
            points_lines.append(
                "Dish classification: None — Dish multiplier "
                f"{format_multiplier(outcome.dish_multiplier)}"
            )

        points_lines.append(
            f"Dish value before recipes: {outcome.dish_value:.2f}"
        )
        if outcome.applied_seasonings:
            boost_pct = int(round(outcome.seasoning_boost_pct * 100))
            penalty_value = int(round(outcome.seasoning_penalty))
            summary = [
                f"Base {outcome.base_score} → {outcome.seasoned_score}",
                f"boost {boost_pct:+d}%",
            ]
            if penalty_value:
                summary.append(f"penalty -{penalty_value}")
            points_lines.append("Seasonings: " + ", ".join(summary))
            if self.session:
                applied = []
                for seasoning_id, count in outcome.applied_seasonings:
                    seasoning = self.session.data.seasoning_by_id.get(seasoning_id)
                    if seasoning:
                        display = seasoning.display_name or seasoning.name
                    else:
                        display = seasoning_id
                    applied.append(f"{display} ×{count}")
                if applied:
                    points_lines.append("Applied: " + ", ".join(applied))
            if outcome.ruined:
                points_lines.append("Seasonings ruined this dish — score reduced to zero.")
        if outcome.recipe_name:
            points_lines.append(
                f"Recipe multiplier: {format_multiplier(outcome.recipe_multiplier)}"
            )
            if outcome.discovered_recipe:
                if outcome.personal_discovery:
                    base_text = ""
                    if self.session and outcome.recipe_name in self.session.data.recipe_by_name:
                        recipe = self.session.data.recipe_by_name[outcome.recipe_name]
                        base_text = (
                            f" Base multiplier starts at {format_multiplier(recipe.base_multiplier)}."
                        )
                    points_lines.append(
                        "Personal discovery! Recipe added to your cookbook." + base_text
                    )
                else:
                    points_lines.append("New recipe added to your cookbook!")
            if outcome.times_cooked_total:
                times = (
                    "time"
                    if outcome.times_cooked_total == 1
                    else "times"
                )
                points_lines.append(
                    f"Cooked {outcome.recipe_display_name or outcome.recipe_name} "
                    f"{outcome.times_cooked_total} {times} total."
                )
        else:
            points_lines.append("No recipe bonus this turn.")
        points_lines.append(
            f"Score gained this turn: {outcome.final_score:+d}"
        )

        chef_hits = (
            f"Chef key ingredients used: {outcome.chef_hits}/"
            f"{max(len(outcome.selected), 1)}"
        )
        points_lines.append(chef_hits)

        if outcome.deck_refreshed:
            points_lines.append("Pantry refreshed for the next hand!")

        total_score = self.session.get_total_score() if self.session else outcome.final_score
        points_lines.append(f"Cumulative score: {total_score}")

        tk.Label(
            content,
            text="Points breakdown:",
            font=("Helvetica", 11, "bold"),
            fg="#1d2730",
            bg=base_bg,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            content,
            text="\n".join(points_lines),
            font=("Helvetica", 10),
            fg="#26313a",
            bg=base_bg,
            justify="left",
            anchor="w",
        ).pack(anchor="w", fill="x", pady=(0, 12))

        ttk.Separator(content, orient="horizontal").pack(fill="x", pady=(0, 10))

        footer = tk.Frame(content, bg=base_bg)
        footer.pack(fill="x")

        ttk.Button(footer, text="OK", command=close_popup).pack(pady=(6, 0))

        popup.update_idletasks()
        self._center_popup(popup)
        try:
            popup.grab_set()
        except tk.TclError:
            pass

        if outcome.recipe_name and outcome.discovered_recipe:
            glow_colors = ["#ffe8a8", "#ffd56f", "#fff3c4", "#ffdba0"]
            self._animate_popup_highlight(glow_frame, glow_colors, cycles=10, delay=120)

        self.active_popup = popup

    def _center_popup(self, popup: tk.Toplevel) -> None:
        popup.update_idletasks()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        popup_w = popup.winfo_width()
        popup_h = popup.winfo_height()
        x = root_x + (root_w - popup_w) // 2
        y = root_y + (root_h - popup_h) // 2
        popup.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _animate_popup_highlight(
        self, widget: tk.Widget, colors: Sequence[str], *, cycles: int = 6, delay: int = 140
    ) -> None:
        if not colors:
            return

        def step(index: int) -> None:
            if not widget.winfo_exists():
                return
            widget.configure(bg=colors[index % len(colors)])
            if index < cycles:
                widget.after(delay, lambda: step(index + 1))
            else:
                widget.configure(bg="#edf1f7")

        step(0)

    def maybe_prompt_new_chef(self) -> None:
        if not self.session or self.session.is_finished():
            self._update_chef_button()
            return
        if not self.session.can_recruit_chef():
            self._update_chef_button()
            return
        if self.recruit_dialog and self.recruit_dialog.winfo_exists():
            self._update_chef_button()
            return
        self._update_chef_button()
        self.show_recruit_dialog()

    def show_recruit_dialog(self) -> None:
        if not self.session:
            return
        available_chefs = self.session.available_chefs()
        available_seasonings = self.session.available_seasonings()
        if not (available_chefs or available_seasonings):
            self.session.skip_chef_recruitment()
            self.append_events(self.session.consume_events())
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Choose a Reward")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        self.recruit_dialog = dialog

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="Select a new chef or seasoning wildcard to boost your kitchen:",
            justify="left",
            wraplength=360,
        ).grid(row=0, column=0, sticky="w")

        tiles_frame = ttk.Frame(frame)
        tiles_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        tiles_frame.columnconfigure(0, weight=1)
        tiles_frame.columnconfigure(1, weight=1)

        chef_tile = ttk.Frame(tiles_frame, style="Tile.TFrame", padding=(14, 12))
        chef_tile.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        ttk.Label(
            chef_tile,
            text="👩‍🍳 Recruit Chef",
            style="TileHeader.TLabel",
            anchor="w",
            justify="left",
        ).pack(anchor="w")
        chef_desc = ttk.Label(
            chef_tile,
            text="Add a random chef to unlock new key ingredients and recipe boosts.",
            style="TileInfo.TLabel",
            wraplength=200,
            justify="left",
        )
        chef_desc.pack(anchor="w", pady=(8, 0))
        chef_count_var = tk.StringVar()
        chef_count_label = ttk.Label(
            chef_tile,
            textvariable=chef_count_var,
            style="TileSub.TLabel",
            wraplength=200,
            justify="left",
        )
        chef_count_label.pack(anchor="w", pady=(12, 0))

        seasoning_tile = ttk.Frame(tiles_frame, style="Tile.TFrame", padding=(14, 12))
        seasoning_tile.grid(row=0, column=1, sticky="nsew")

        ttk.Label(
            seasoning_tile,
            text="🧂 Claim Seasoning",
            style="TileHeader.TLabel",
            anchor="w",
            justify="left",
        ).pack(anchor="w")
        seasoning_desc = ttk.Label(
            seasoning_tile,
            text="Draw a random seasoning that can complete any missing taste in a combo.",
            style="TileInfo.TLabel",
            wraplength=200,
            justify="left",
        )
        seasoning_desc.pack(anchor="w", pady=(8, 0))
        seasoning_count_var = tk.StringVar()
        seasoning_count_label = ttk.Label(
            seasoning_tile,
            textvariable=seasoning_count_var,
            style="TileSub.TLabel",
            wraplength=200,
            justify="left",
        )
        seasoning_count_label.pack(anchor="w", pady=(12, 0))

        status_var = tk.StringVar(value="")
        status_label = ttk.Label(
            frame,
            textvariable=status_var,
            style="TileSub.TLabel",
            wraplength=360,
            justify="left",
        )
        status_label.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, sticky="e", pady=(12, 0))

        def close_dialog() -> None:
            if dialog.winfo_exists():
                dialog.destroy()
            if self.recruit_dialog is dialog:
                self.recruit_dialog = None

        def set_tile_cursor(tile: tk.Widget, enabled: bool) -> None:
            cursor = "hand2" if enabled else "X_cursor"
            try:
                tile.configure(cursor=cursor)
            except tk.TclError:
                pass
            for child in tile.winfo_children():
                try:
                    child.configure(cursor=cursor)
                except tk.TclError:
                    continue

        def refresh_tile_states() -> None:
            if not self.session:
                return
            chef_remaining = len(self.session.available_chefs())
            seasoning_remaining = len(self.session.available_seasonings())
            if chef_remaining:
                chef_count_var.set(f"{chef_remaining} chef{'s' if chef_remaining != 1 else ''} available.")
            else:
                chef_count_var.set("All chefs recruited.")
            if seasoning_remaining:
                seasoning_count_var.set(
                    f"{seasoning_remaining} seasoning{'s' if seasoning_remaining != 1 else ''} remaining."
                )
            else:
                seasoning_count_var.set("All seasonings collected.")
            set_tile_cursor(chef_tile, chef_remaining > 0)
            set_tile_cursor(seasoning_tile, seasoning_remaining > 0)

        def choose_chef() -> None:
            if not self.session:
                return
            chef = self.session.random_available_chef()
            if not chef:
                status_var.set("All chefs have already joined your team.")
                refresh_tile_states()
                return
            try:
                self.session.add_chef(chef)
            except Exception as exc:  # pragma: no cover - user feedback path
                messagebox.showerror("Cannot recruit chef", str(exc))
                close_dialog()
                return
            self.selected_indices.clear()
            self.update_selection_label()
            self.render_hand()
            self.update_status()
            if self.cookbook_tile:
                self.cookbook_tile.set_entries(self.session.get_cookbook())
            if self.team_tile:
                self.team_tile.set_team(self.session.chefs, self.session.max_chefs)
            if self.seasoning_tile:
                self.seasoning_tile.set_seasonings(
                    self.session.get_seasonings(),
                    len(self.session.available_seasonings()),
                )
            self.append_events(self.session.consume_events())
            self._update_chef_button()
            self._update_seasoning_button()
            messagebox.showinfo("Chef Recruited", f"{chef.name} joins your team!")
            close_dialog()

        def choose_seasoning() -> None:
            if not self.session:
                return
            seasoning = self.session.random_available_seasoning()
            if not seasoning:
                status_var.set("All seasonings have already been collected.")
                refresh_tile_states()
                return
            try:
                self.session.add_seasoning(seasoning)
            except Exception as exc:  # pragma: no cover - user feedback path
                messagebox.showerror("Cannot add seasoning", str(exc))
                close_dialog()
                return
            if self.seasoning_tile:
                self.seasoning_tile.set_seasonings(
                    self.session.get_seasonings(),
                    len(self.session.available_seasonings()),
                )
            self.append_events(self.session.consume_events())
            display_name = seasoning.display_name or seasoning.name
            perk_text = seasoning.perk.strip()
            if perk_text:
                message = f"You secured {display_name}!\n\n{perk_text}"
            else:
                message = f"You secured {display_name}!"
            self._update_seasoning_button(seasoning)
            self._refresh_seasoning_popup()
            self._update_chef_button()
            self._update_seasoning_panels()
            messagebox.showinfo("Seasoning Collected", message)
            close_dialog()

        def make_clickable(tile: tk.Widget, command: Callable[[], None]) -> None:
            def handler(_event: tk.Event) -> None:
                command()

            tile.bind("<Button-1>", handler)
            for child in tile.winfo_children():
                child.bind("<Button-1>", handler)

        def on_skip() -> None:
            if self.session:
                self.session.skip_chef_recruitment()
                self.append_events(self.session.consume_events())
                self._update_chef_button()
            close_dialog()

        make_clickable(chef_tile, choose_chef)
        make_clickable(seasoning_tile, choose_seasoning)

        ttk.Button(button_frame, text="Skip", command=on_skip).grid(row=0, column=0)

        refresh_tile_states()

        dialog.protocol("WM_DELETE_WINDOW", on_skip)
        dialog.bind("<Escape>", lambda _e: on_skip())
        dialog.grab_set()
        self._center_popup(dialog)

    def _format_outcome(self, outcome: TurnOutcome) -> str:
        parts = [
            f"Turn {outcome.turn_number} — "
            f"Round {outcome.round_index}",
            "",
        ]
        parts.append("Cooked selection:")
        for ingredient in outcome.selected:
            parts.append(
                f"  • {ingredient.name} (Taste: {ingredient.taste}, Value: {ingredient.Value})"
            )
        parts.append("")
        parts.append(f"Total Value: {outcome.Value}")
        parts.append(
            f"Family profile: {outcome.family_label}"
            f" ({outcome.family_pattern.replace('_', ' ')})"
        )
        if outcome.dish_name:
            tier_text = f" [{outcome.dish_tier}]" if outcome.dish_tier else ""
            parts.append(
                f"Dish classification: {outcome.dish_name}{tier_text}"
                f" — Dish multiplier {format_multiplier(outcome.dish_multiplier)}"
            )
        else:
            parts.append(
                "Dish classification: None — Dish multiplier "
                f"{format_multiplier(outcome.dish_multiplier)}"
            )
        parts.append(
            f"Dish value before recipe bonus: {outcome.dish_value:.2f}"
        )
        if outcome.applied_seasonings:
            boost_pct = int(round(outcome.seasoning_boost_pct * 100))
            penalty_value = int(round(outcome.seasoning_penalty))
            summary_bits = [
                f"{outcome.base_score} → {outcome.seasoned_score}",
                f"boost {boost_pct:+d}%",
            ]
            if penalty_value:
                summary_bits.append(f"penalty -{penalty_value}")
            applied_line = "Seasonings: " + ", ".join(summary_bits)
            parts.append(applied_line)
            if self.session:
                applied_names: List[str] = []
                for seasoning_id, count in outcome.applied_seasonings:
                    seasoning = self.session.data.seasoning_by_id.get(seasoning_id)
                    if seasoning:
                        display = seasoning.display_name or seasoning.name
                    else:
                        display = seasoning_id
                    applied_names.append(f"{display} ×{count}")
                if applied_names:
                    parts.append("Applied: " + ", ".join(applied_names))
            if outcome.ruined:
                parts.append("Seasonings ruined this dish — value clamped to zero.")
        if outcome.alerts:
            parts.append("")
            for alert in outcome.alerts:
                parts.append(f"⚠️  {alert}")
        if outcome.recipe_name:
            recipe_display = outcome.recipe_display_name or outcome.recipe_name
            parts.append(
                f"Recipe completed: {recipe_display} (x{outcome.recipe_multiplier:.2f})"
            )
            if outcome.discovered_recipe:
                if outcome.personal_discovery:
                    parts.append("Personal discovery added to your cookbook!")
                else:
                    parts.append("New recipe added to your cookbook!")
            if outcome.times_cooked_total:
                parts.append(
                    f"Total times cooked: {outcome.times_cooked_total}"
                )
        else:
            parts.append("No recipe completed this turn.")
        base_text = f"base Value: {outcome.base_score}"
        if outcome.applied_seasonings:
            base_text = (
                f"base Value: {outcome.base_score} → seasoned {outcome.seasoned_score}"
            )
        parts.append(f"Score gained: {outcome.final_score:+d} ({base_text})")
        parts.append(
            f"Chef key ingredients used: {outcome.chef_hits}/{max(len(outcome.selected), 1)}"
        )
        if outcome.deck_refreshed:
            parts.append("Pantry refreshed for the next turn.")

        parts.append("")
        parts.append(f"Cumulative score: {self.session.get_total_score()}")
        return "\n".join(parts)


def make_run_id() -> str:
    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


def _blank_run_state() -> Dict[str, object]:
    return {
        "run_id": make_run_id(),
        "turn_index": 1,
        "pantry": [],
        "chefs_active": [],
        "seasoning_owned": [],
        "lives": 0,
        "stats": {"total_score": 0, "baskets_cleared": 0},
    }


def _format_challenge_summary(data: GameData, challenge: BasketChallenge) -> str:
    reward = challenge.reward
    reward_type = reward.get("type", "?")
    reward_rarity = reward.get("rarity", "?")
    unique_count, total_count = challenge_ingredient_counts(data, challenge)
    ingredient_text = format_challenge_ingredient_text(unique_count, total_count)
    if ingredient_text != "No listed ingredients":
        ingredient_phrase = f"adds {ingredient_text}"
    else:
        ingredient_phrase = ingredient_text
    return (
        f"[{challenge.difficulty.upper()}] {challenge.basket_name} — "
        f"Target {challenge.target_score} pts; "
        f"{ingredient_phrase}; "
        f"reward {reward_type} ({reward_rarity})"
    )


def _debug_print_challenges(data: GameData, *, seed: Optional[int] = None) -> None:
    rng = random.Random(seed) if seed is not None else random.Random()
    run_state = _blank_run_state()
    factory = BasketChallengeFactory(data, TARGET_SCORE_CONFIG)
    offers = factory.three_offers(run_state, rng=rng)
    print("=== Basket Challenge Debug ===")
    if seed is not None:
        print(f"Seed: {seed}")
    else:
        print("Seed: random")
    for index, offer in enumerate(offers, start=1):
        print(f"{index}. {offer.id}")
        print(f"   {_format_challenge_summary(data, offer)}")
        added_preview = ", ".join(offer.added_ing_ids[:6])
        if len(offer.added_ing_ids) > 6:
            added_preview += ", …"
        if not added_preview:
            added_preview = "None"
        print(f"   Pantry adds ({len(offer.added_ing_ids)}): {added_preview}")
    print()


def main() -> None:
    if "--debug-baskets" in sys.argv:
        debug_seed: Optional[int] = None
        idx = sys.argv.index("--debug-baskets")
        if idx + 1 < len(sys.argv):
            try:
                debug_seed = int(sys.argv[idx + 1])
            except ValueError:
                debug_seed = None
        _debug_print_challenges(DATA, seed=debug_seed)
        return

    root = tk.Tk()
    app = FoodGameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
