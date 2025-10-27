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
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
    GameData,
    Ingredient,
    Seasoning,
    SimulationConfig,
    describe_family_pattern,
    describe_flavor_pattern,
    quantize_multiplier,
    build_market_deck,
)
ASSET_DIR = Path(__file__).resolve().parent
ICON_ASSET_DIR = ASSET_DIR / "icons"
INGREDIENT_ASSET_DIR = ASSET_DIR / "Ingredients"
RECIPE_ASSET_DIR = ASSET_DIR / "recipes"


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
DIALOG_ICON_TARGET_PX = 40
INGREDIENT_IMAGE_TARGET_PX = 160
RECIPE_IMAGE_TARGET_PX = 240

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow<9.1 fallback
    RESAMPLE_LANCZOS = Image.LANCZOS


_icon_cache: Dict[str, tk.PhotoImage] = {}
_ingredient_image_cache: Dict[str, tk.PhotoImage] = {}
_seasoning_icon_cache: Dict[str, tk.PhotoImage] = {}
_button_icon_cache: Dict[str, tk.PhotoImage] = {}
_recipe_image_cache: Dict[str, Optional[tk.PhotoImage]] = {}


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


def _find_recipe_image_path(
    recipe_name: str, display_name: Optional[str] = None
) -> Optional[Path]:
    if not RECIPE_ASSET_DIR.exists():
        return None

    candidates = _candidate_image_basenames((recipe_name, display_name))
    extensions = (".png", ".jpg", ".jpeg", ".gif")
    for base in candidates:
        for ext in extensions:
            path = RECIPE_ASSET_DIR / f"{base}{ext}"
            if path.exists():
                return path
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
    if image_path and image_path.exists():
        with Image.open(image_path) as source_image:
            working = source_image.convert("RGBA")
            working.thumbnail((target_px, target_px), RESAMPLE_LANCZOS)
            working = working.copy()
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
    total_rounds: int
    cook_number: int
    cooks_per_round: int
    turn_number: int
    total_turns: int
    deck_refreshed: bool
    discovered_recipe: bool
    personal_discovery: bool
    alerts: Tuple[str, ...] = ()


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


class GameSession:
    """Manage deck, hand, and scoring for a single run."""

    def __init__(
        self,
        data: GameData,
        basket_name: str,
        chefs: Sequence[Chef],
        rounds: int,
        cooks_per_round: int,
        hand_size: int,
        pick_size: int,
        deck_size: int = DEFAULT_DECK_SIZE,
        bias: float = DEFAULT_BIAS,
        max_chefs: int = DEFAULT_MAX_CHEFS,
        rng: Optional[random.Random] = None,
    ) -> None:
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        if cooks_per_round <= 0:
            raise ValueError("cooks_per_round must be positive")
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
        self.rounds = rounds
        self.cooks_per_round = cooks_per_round
        self.hand_size = hand_size
        self.pick_size = pick_size
        self.deck_size = deck_size
        self.bias = bias
        self.max_chefs = max_chefs
        self.rng = rng or random.Random()

        self.total_turns = rounds * cooks_per_round
        self.turn_number = 0
        self.round_index = 0
        self.cooks_completed_in_round = 0
        self.total_score = 0
        self.finished = False
        self.pending_new_chef_offer = False

        self.hand: List[Ingredient] = []
        self.deck: List[Ingredient] = []
        self.seasonings: List[Seasoning] = []
        self._events: List[str] = []

        self._cookbook_ingredients: set[str] = set()
        self._ingredient_recipe_map = {
            name: tuple(recipes)
            for name, recipes in self.data.ingredient_recipes.items()
        }

        self._refresh_chef_data()
        self.cookbook: Dict[str, CookbookEntry] = {}

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

    # ----------------- Round & hand management -----------------
    def _start_next_round(self, initial: bool = False) -> None:
        if self.round_index >= self.rounds and not initial:
            self.finished = True
            self._push_event("Run complete! No more rounds remaining.")
            return

        if initial:
            self.round_index = 1
        else:
            self.round_index += 1
            if self.round_index > self.rounds:
                self.finished = True
                self._push_event("Run complete! No more rounds remaining.")
                return

        self.cooks_completed_in_round = 0
        self.deck = build_market_deck(
            self.data,
            self.basket_name,
            self.chefs,
            deck_size=self.deck_size,
            bias=self.bias,
            rng=self.rng,
        )
        self.rng.shuffle(self.deck)
        self._current_deck_total = len(self.deck)
        self.hand.clear()
        self.pending_new_chef_offer = False
        self._push_event(
            f"Round {self.round_index}/{self.rounds} begins. Deck shuffled for the team."
        )
        if not initial and (self.available_chefs() or self.available_seasonings()):
            self.pending_new_chef_offer = True
            self._push_event(
                "You may recruit an additional chef or claim a seasoning wildcard before drawing the next hand."
            )
        self._refill_hand()

    def _refill_hand(self) -> bool:
        needed = self.hand_size - len(self.hand)
        deck_refreshed = False
        while needed > 0 and not self.finished:
            if not self.deck:
                self.deck = build_market_deck(
                    self.data,
                    self.basket_name,
                    self.chefs,
                    deck_size=self.deck_size,
                    bias=self.bias,
                    rng=self.rng,
                )
                self.rng.shuffle(self.deck)
                self._current_deck_total = len(self.deck)
                self._push_event("Market deck refreshed with new draws.")
                deck_refreshed = True
            if not self.deck:
                self.finished = True
                self._push_event("Deck exhausted; ending the run early.")
                return deck_refreshed
            self.hand.append(self.deck.pop())
            needed -= 1
        if len(self.hand) == 0:
            self.finished = True
            self._push_event("Not enough cards to continue this run.")
        return deck_refreshed

    def _rebuild_deck_for_new_chef(self) -> None:
        if self.finished:
            return
        self.deck = build_market_deck(
            self.data,
            self.basket_name,
            self.chefs,
            deck_size=self.deck_size,
            bias=self.bias,
            rng=self.rng,
        )
        self.rng.shuffle(self.deck)
        self._current_deck_total = len(self.deck)
        self.hand.clear()
        self._push_event("Deck refreshed to reflect your expanded chef lineup.")
        self._refill_hand()

    def _times_cooked(self, recipe_name: Optional[str]) -> int:
        if not recipe_name:
            return 0
        entry = self.cookbook.get(recipe_name)
        return entry.count if entry else 0

    # ----------------- Public API -----------------
    def get_hand(self) -> Sequence[Ingredient]:
        return list(self.hand)

    def get_remaining_deck(self) -> Sequence[Ingredient]:
        return list(self.deck)

    def get_basket_counts(self) -> Tuple[int, int]:
        return len(self.deck), self._current_deck_total

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
        if self.finished:
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
        self._rebuild_deck_for_new_chef()

    def add_seasoning(self, seasoning: Seasoning) -> None:
        if any(existing.name == seasoning.name for existing in self.seasonings):
            raise ValueError(f"{seasoning.display_name or seasoning.name} is already in your pantry.")
        if self.finished:
            raise RuntimeError("Cannot add seasonings after the run has finished.")
        self.seasonings.append(seasoning)
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

    def preview_recipe_multiplier(self, recipe_name: Optional[str]) -> float:
        return self.data.recipe_multiplier(
            recipe_name,
            chefs=self.chefs,
            times_cooked=self._times_cooked(recipe_name),
        )

    def get_selection_markers(self, ingredient: Ingredient) -> Tuple[List[str], bool]:
        markers = [
            chef.name
            for chef in self.chefs
            if ingredient.name in self._chef_key_map.get(chef.name, set())
        ]
        in_cookbook = ingredient.name in self._cookbook_ingredients
        return markers, in_cookbook

    def get_recipe_hints(self, ingredient: Ingredient) -> List[str]:
        hints: List[str] = []
        for recipe_name in self._ingredient_recipe_map.get(ingredient.name, ()):
            display = self.data.recipe_display_name(recipe_name)
            hints.append(display or recipe_name)
        return sorted(hints, key=lambda value: value.lower())

    def discard_indices(self, indices: Sequence[int]) -> Tuple[List[Ingredient], bool]:
        if self.finished:
            raise RuntimeError("The session has already finished.")
        if not indices:
            raise ValueError("You must select at least one card to discard.")
        if len(indices) > self.pick_size:
            raise ValueError(
                f"You may discard up to {self.pick_size} cards at a time."
            )

        unique = sorted(set(indices))
        if len(unique) != len(indices):
            raise ValueError("Selections contain duplicates.")

        if any(index < 0 or index >= len(self.hand) for index in unique):
            raise IndexError("Selection index out of range for the current hand.")

        removed = [self.hand[index] for index in unique]
        for offset, index in enumerate(unique):
            self.hand.pop(index - offset)

        deck_refreshed = self._refill_hand()

        if removed:
            if len(removed) == 1:
                name = removed[0].name
                self._push_event(f"Discarded {name} to draw a new ingredient.")
            else:
                names = ", ".join(ingredient.name for ingredient in removed)
                self._push_event(f"Discarded {names} and drew replacements.")

        return removed, deck_refreshed

    def play_turn(self, indices: Sequence[int]) -> TurnOutcome:
        if self.finished:
            raise RuntimeError("The session has already finished.")
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

        selected = [self.hand[index] for index in unique]
        dish = self.data.evaluate_dish(selected)
        if dish.alerts:
            for alert in dish.alerts:
                self._push_event(alert)
        Value = dish.base_value
        recipe_name = self.data.which_recipe(selected)
        recipe_display_name = (
            self.data.recipe_display_name(recipe_name) if recipe_name else None
        )
        times_cooked_before = self._times_cooked(recipe_name)
        recipe_multiplier = self.data.recipe_multiplier(
            recipe_name,
            chefs=self.chefs,
            times_cooked=times_cooked_before,
        )
        final_score = int(round(dish.dish_value * recipe_multiplier))
        chef_hits = sum(1 for ing in selected if ing.name in self._chef_key_set)

        discovered = False
        personal_discovery = False
        times_cooked_total = 0
        if recipe_name:
            recipe = self.data.recipe_by_name.get(recipe_name)
            if recipe:
                combo: Tuple[str, ...] = tuple(recipe.trio)
            else:
                combo = tuple(sorted(ingredient.name for ingredient in selected))
            entry = self.cookbook.get(recipe_name)
            chef_has_recipe = any(
                recipe_name in chef.recipe_names for chef in self.chefs
            )
            if not entry:
                display_name = recipe_display_name or recipe_name or ", ".join(combo)
                entry = CookbookEntry(combo, display_name)
                entry.personal_discovery = not chef_has_recipe
                self.cookbook[recipe_name] = entry
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
                recipe_name,
                chefs=self.chefs,
                times_cooked=display_times,
            )
            if discovered:
                event_name = recipe_display_name or recipe_name
                if personal_discovery:
                    self._push_event(
                        f"You personally discovered {event_name}! Added to your cookbook."
                    )
                else:
                    self._push_event(
                        f"{event_name} added to your cookbook thanks to your chef team."
                    )

        current_round = self.round_index
        current_cook = self.cooks_completed_in_round + 1
        current_turn = self.turn_number + 1

        self.total_score += final_score
        self.turn_number += 1
        self.cooks_completed_in_round += 1

        for ingredient in selected:
            self.hand.remove(ingredient)

        deck_refreshed = False

        if self.cooks_completed_in_round >= self.cooks_per_round:
            was_finished = self.finished
            self._start_next_round()
            deck_refreshed = not was_finished and not self.finished
        else:
            deck_refreshed = self._refill_hand() or deck_refreshed

        return TurnOutcome(
            selected=selected,
            Value=Value,
            dish_value=dish.dish_value,
            dish_multiplier=dish.dish_multiplier,
            dish_name=dish.name,
            dish_tier=dish.tier,
            family_label=dish.family_label,
            flavor_label=dish.flavor_label,
            family_pattern=dish.family_pattern,
            flavor_pattern=dish.flavor_pattern,
            recipe_name=recipe_name,
            recipe_display_name=recipe_display_name,
            recipe_multiplier=recipe_multiplier,
            final_score=final_score,
            times_cooked_total=times_cooked_total,
            base_score=int(round(dish.dish_value)),
            chef_hits=chef_hits,
            round_index=current_round,
            total_rounds=self.rounds,
            cook_number=current_cook,
            cooks_per_round=self.cooks_per_round,
            turn_number=current_turn,
            total_turns=self.total_turns,
            deck_refreshed=deck_refreshed,
            discovered_recipe=discovered,
            personal_discovery=personal_discovery,
            alerts=tuple(dish.alerts),
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
                self.subtitle_var.set("No seasonings collected yet. Finish rounds to claim one.")
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
        cookbook_hint: bool,
        on_click: Optional[Callable[[int], None]] = None,
        *,
        quantity: Optional[int] = None,
    ) -> None:
        super().__init__(master, style="Card.TFrame", padding=(10, 8))
        self.index = index
        self.on_click = on_click
        self.selected = False
        self.cookbook_hint = cookbook_hint
        self.quantity = quantity

        self.columnconfigure(0, weight=1)

        display_name = getattr(ingredient, "display_name", None) or ingredient.name
        name_text = display_name
        if quantity and quantity > 1:
            name_text += f" ×{quantity}"
        if cookbook_hint:
            name_text += " 📖"

        self.name_label = ttk.Label(
            self,
            text=name_text,
            style="CardTitle.TLabel",
            anchor="center",
            justify="center",
        )
        self.name_label.grid(row=0, column=0, sticky="ew")

        separator = ttk.Separator(self, orient="horizontal")
        separator.grid(row=1, column=0, sticky="ew", pady=(6, 8))

        row_index = 2

        self.ingredient_image = _load_ingredient_image(ingredient)
        self.ingredient_image_label = ttk.Label(
            self,
            image=self.ingredient_image,
            anchor="center",
        )
        self.ingredient_image_label.grid(row=row_index, column=0, sticky="n", pady=(0, 6))
        row_index += 1

        self.taste_image = _load_icon("taste", ingredient.taste)
        taste_text = f"Taste: {ingredient.taste}"
        self.taste_label = ttk.Label(
            self,
            text=taste_text,
            style="CardBody.TLabel",
            image=self.taste_image,
            compound="left",
        )
        self.taste_label.grid(row=row_index, column=0, sticky="w")
        row_index += 1

        self.family_image = _load_icon("family", ingredient.family)
        family_text = f"Family: {ingredient.family}"
        self.family_label = ttk.Label(
            self,
            text=family_text,
            style="CardBody.TLabel",
            image=self.family_image,
            compound="left",
        )
        self.family_label.grid(row=row_index, column=0, sticky="w", pady=(2, 0))
        row_index += 1

        self.Value_label = ttk.Label(
            self, text=f"Value: {ingredient.Value}", style="CardBody.TLabel"
        )
        self.Value_label.grid(row=row_index, column=0, sticky="w", pady=(2, 0))
        row_index += 1

        self.chef_label: Optional[ttk.Label] = None
        if chef_names:
            first, *rest = chef_names
            lines = [f"Chef Key: {first}"]
            lines.extend(f"           {name}" for name in rest)
            self.chef_label = ttk.Label(
                self,
                text="\n".join(lines),
                style="CardMarker.TLabel",
                justify="left",
            )
            self.chef_label.grid(row=row_index, column=0, sticky="w", pady=(4, 0))
            row_index += 1

        hint_text = ", ".join(recipe_hints) if recipe_hints else "(none)"
        self.recipe_label = ttk.Label(
            self,
            text=f"Recipes: {hint_text}",
            style="CardHint.TLabel",
            wraplength=220,
            justify="left",
        )
        self.recipe_label.grid(row=row_index, column=0, sticky="w", pady=(4, 0))

        if self.on_click:
            self.bind("<Button-1>", self._handle_click)
            for child in self.winfo_children():
                child.bind("<Button-1>", self._handle_click)

    def _handle_click(self, _event) -> None:
        if self.on_click:
            self.on_click(self.index)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        style = "CardSelected.TFrame" if selected else "Card.TFrame"
        self.configure(style=style)
        title_style = "CardTitleSelected.TLabel" if selected else "CardTitle.TLabel"
        body_style = "CardBodySelected.TLabel" if selected else "CardBody.TLabel"
        marker_style = (
            "CardMarkerSelected.TLabel" if selected else "CardMarker.TLabel"
        )
        hint_style = "CardHintSelected.TLabel" if selected else "CardHint.TLabel"
        self.name_label.configure(style=title_style)
        self.taste_label.configure(style=body_style)
        self.family_label.configure(style=body_style)
        self.Value_label.configure(style=body_style)
        if self.chef_label:
            self.chef_label.configure(style=marker_style)
        if self.recipe_label:
            self.recipe_label.configure(style=hint_style)


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

        self.title("Ingredient Basket")
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
            text="Remaining ingredients in your basket",
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
                text="Your basket is empty. Draw or start a new round to refill it.",
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
            chef_names, cookbook_hint = self.session.get_selection_markers(ingredient)
            recipe_hints = self.session.get_recipe_hints(ingredient)
            card = CardView(
                self.cards_frame,
                index=index,
                ingredient=ingredient,
                chef_names=chef_names,
                recipe_hints=recipe_hints,
                cookbook_hint=cookbook_hint,
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


class FoodGameApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Food Deck Simulator — Desktop Prototype")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)

        self.session: Optional[GameSession] = None
        self.card_views: Dict[int, CardView] = {}
        self.selected_indices: set[int] = set()
        self.spinboxes: List[ttk.Spinbox] = []
        self.cookbook_tile: Optional[CookbookTile] = None
        self.team_tile: Optional[ChefTeamTile] = None
        self.seasoning_tile: Optional[SeasoningTile] = None
        self.active_popup: Optional[tk.Toplevel] = None
        self.cookbook_popup: Optional["CookbookPopup"] = None
        self.seasoning_popup: Optional["SeasoningPopup"] = None
        self.recruit_dialog: Optional[tk.Toplevel] = None
        self.deck_popup: Optional["DeckPopup"] = None
        self.dish_dialog: Optional[DishMatrixDialog] = None

        self.hand_sort_modes: Tuple[str, ...] = ("name", "family", "taste")
        self.hand_sort_index = 0
        self.hand_sort_var = tk.StringVar(
            value=self._format_sort_label(self._current_sort_mode())
        )

        self._resource_button_images: Dict[str, tk.PhotoImage] = {}
        self._action_button_images: Dict[str, tk.PhotoImage] = {}

        self._init_styles()
        self._build_layout()

    # ----------------- UI setup -----------------
    def _init_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_bg = "#f5f5f5"
        selected_bg = "#e6edf7"
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
            "CardTitle.TLabel",
            font=title_font,
            foreground="#2f2f2f",
            background=base_bg,
        )
        style.configure(
            "CardBody.TLabel",
            font=body_font,
            foreground="#3a3a3a",
            background=base_bg,
        )
        style.configure(
            "CardMarker.TLabel",
            font=marker_font,
            foreground="#5a5a5a",
            background=base_bg,
        )
        style.configure(
            "CardHint.TLabel",
            font=("Helvetica", 9),
            foreground="#4a4a4a",
            background=base_bg,
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

        style.configure("Info.TLabel", font=("Helvetica", 10), foreground="#2f2f2f")
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#1f1f1f")
        style.configure("Score.TLabel", font=("Helvetica", 18, "bold"), foreground="#1f1f1f")
        style.configure("Summary.TLabel", font=("Helvetica", 16, "bold"), foreground="#1c1c1c")
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

        self._build_controls()
        self._build_game_panel()

    def _build_controls(self) -> None:
        ttk.Label(self.control_frame, text="Basket", style="Header.TLabel").pack(anchor="w")
        self.basket_var = tk.StringVar()
        basket_names = sorted(DATA.baskets.keys())
        if "Basic" in basket_names:
            basket_names = ["Basic"] + [name for name in basket_names if name != "Basic"]
        self.basket_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.basket_var,
            values=basket_names,
            state="readonly",
            width=28,
        )
        if basket_names:
            self.basket_combo.current(0)
        self.basket_combo.pack(anchor="w", pady=(4, 12))

        ttk.Label(
            self.control_frame,
            text="You'll be prompted to recruit chefs or claim seasonings once a run begins.",
            style="Info.TLabel",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        config_frame = ttk.Frame(self.control_frame)
        config_frame.pack(anchor="w", pady=(8, 0))

        self.round_var = tk.IntVar(value=DEFAULT_CONFIG.rounds)
        self.cooks_var = tk.IntVar(value=DEFAULT_CONFIG.cooks)
        self.hand_var = tk.IntVar(value=DEFAULT_HAND_SIZE)
        self.pick_var = tk.IntVar(value=DEFAULT_PICK_SIZE)
        self.max_chefs_var = tk.IntVar(value=DEFAULT_MAX_CHEFS)

        self._add_spinbox(config_frame, "Rounds", self.round_var, 1, 10)
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

        self.progress_var = tk.StringVar(value="Round 0 / 0 — Turn 0 / 0")
        ttk.Label(score_frame, textvariable=self.progress_var, style="Info.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        self.selection_var = tk.StringVar(value="Selection: 0")
        ttk.Label(score_frame, textvariable=self.selection_var, style="Info.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
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
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        self.sort_button = ttk.Button(
            score_frame,
            textvariable=self.hand_sort_var,
            command=self.cycle_hand_sort_mode,
        )
        self.sort_button.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.chefs_var = tk.StringVar(
            value=f"Active chefs ({DEFAULT_MAX_CHEFS} max): —"
        )
        ttk.Label(score_frame, textvariable=self.chefs_var, style="Info.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

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

        action_frame = ttk.Frame(self.game_frame)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        self.cook_button = ttk.Button(
            action_frame,
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

        self.discard_button = ttk.Button(
            action_frame,
            text="Discard",
            command=self.discard_selected,
            state="disabled",
            compound="left",
        )
        self.discard_button.grid(row=0, column=1, sticky="ew")

        discard_icon = _load_button_image("discard.png", target_px=52)
        if discard_icon is None:
            discard_icon = _generate_button_icon("discard", "DC", size=64)
        self._action_button_images["discard"] = discard_icon
        self.discard_button.configure(image=discard_icon)

        resource_frame = ttk.Frame(action_frame)
        resource_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for column in range(5):
            resource_frame.columnconfigure(column, weight=1)

        self.cookbook_count_var = tk.StringVar(value="Cookbook\n0 recipes")
        cookbook_icon = _load_button_image("cookbook.png")
        if cookbook_icon is None:
            cookbook_icon = _generate_button_icon("cookbook", "CB")
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
        self._update_cookbook_button()

        dish_icon = _load_button_image("dishmatrix.png")
        if dish_icon is None:
            dish_icon = _generate_button_icon("dish", "DM")
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

        seasoning_icon = _load_button_image("seasoning.png")
        if seasoning_icon is None:
            seasoning_icon = _generate_button_icon("seasoning", "SN")
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

        basket_icon = _load_button_image("basket.png")
        if basket_icon is None:
            basket_icon = _generate_button_icon("basket", "BK")
        self._resource_button_images["basket"] = basket_icon
        self.basket_count_var = tk.StringVar(value="Basket\n0/0")
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

        chef_icon = _load_button_image("chefs.png")
        if chef_icon is None:
            chef_icon = _generate_button_icon("chef", "CF")
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
        self._update_seasoning_button(None)
        self._update_chef_button()

        self.log_text = tk.Text(
            self.game_frame,
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
        self.log_text.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.log_text.configure(state="disabled")

    # ----------------- Session management -----------------
    def start_run(self) -> None:
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.destroy()
        self.active_popup = None
        if self.deck_popup and self.deck_popup.winfo_exists():
            self.deck_popup.destroy()
        self.deck_popup = None
        self._close_recruit_dialog()
        try:
            basket = self.basket_var.get()
            if not basket:
                raise ValueError("Select a basket before starting a run.")

            rounds = int(self.round_var.get())
            cooks = int(self.cooks_var.get())
            hand_size = int(self.hand_var.get())
            pick_size = int(self.pick_var.get())
            max_chefs = int(self.max_chefs_var.get())

            if pick_size > hand_size:
                raise ValueError("Pick size cannot exceed hand size.")
            if max_chefs <= 0:
                raise ValueError("Max chefs must be at least 1.")

            self.session = GameSession(
                DATA,
                basket_name=basket,
                chefs=[],
                rounds=rounds,
                cooks_per_round=cooks,
                hand_size=hand_size,
                pick_size=pick_size,
                max_chefs=max_chefs,
            )
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Cannot start run", str(exc))
            return

        cookbook_entries = self.session.get_cookbook()
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

        self._set_controls_active(False)
        self.cook_button.configure(state="normal")
        self.discard_button.configure(state="normal")
        self.basket_button.configure(state="normal")
        self.seasoning_button.configure(state="normal")
        self.chef_button.configure(state="normal")
        self.reset_button.configure(state="normal")
        self.selected_indices.clear()
        self.update_selection_label()
        self.render_hand()
        self.update_status()
        self.clear_events()
        self.append_events(self.session.consume_events())
        self._update_seasoning_button(None)
        self._update_chef_button()
        if not self.session.chefs and (
            self.session.available_chefs() or self.session.available_seasonings()
        ):
            self.session.pending_new_chef_offer = True
            self._update_chef_button()
            self.append_events(["Choose a chef or seasoning to begin your run."])
            self.show_recruit_dialog()
        self.write_result("Run started. Select ingredients and press COOK!")

    def reset_session(self) -> None:
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.destroy()
        self.active_popup = None
        if self.deck_popup and self.deck_popup.winfo_exists():
            self.deck_popup.destroy()
        self.deck_popup = None
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.destroy()
        self.cookbook_popup = None
        if self.seasoning_popup and self.seasoning_popup.winfo_exists():
            self.seasoning_popup.destroy()
        self.seasoning_popup = None
        self._close_recruit_dialog()
        self.session = None
        self._update_basket_button()
        self.selected_indices.clear()
        self.update_selection_label()
        self.score_var.set("0")
        self.progress_var.set("Round 0 / 0 — Turn 0 / 0")
        self.chefs_var.set(
            f"Active chefs ({self.max_chefs_var.get()} max): —"
        )
        self.cook_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.basket_button.configure(state="disabled")
        self.seasoning_button.configure(state="disabled")
        self.chef_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self._set_controls_active(True)
        self.clear_hand()
        self.clear_events()
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

    def _set_controls_active(self, active: bool) -> None:
        state = "normal" if active else "disabled"
        self.basket_combo.configure(state="readonly" if active else "disabled")
        for spin in self.spinboxes:
            spin.configure(state=state)
        self.start_button.configure(state="normal" if active else "disabled")

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
            return

        self._refresh_seasoning_popup()
        hand_with_indices = list(enumerate(self.session.get_hand()))
        sorted_hand = self._sorted_hand(hand_with_indices)

        for column, (index, ingredient) in enumerate(sorted_hand):
            chef_names, cookbook_hint = self.session.get_selection_markers(ingredient)
            recipe_hints = self.session.get_recipe_hints(ingredient)
            view = CardView(
                self.hand_frame,
                index=index,
                ingredient=ingredient,
                chef_names=chef_names,
                recipe_hints=recipe_hints,
                cookbook_hint=cookbook_hint,
                on_click=self.toggle_card,
            )
            view.grid(row=0, column=column, sticky="nw", padx=8, pady=8)
            if index in self.selected_indices:
                view.set_selected(True)
            self.card_views[index] = view

        self.hand_frame.update_idletasks()
        self.hand_canvas.configure(scrollregion=self.hand_canvas.bbox("all"))
        self._refresh_deck_popup()
        self._update_basket_button()

    def _sorted_hand(
        self, hand_with_indices: Sequence[Tuple[int, Ingredient]]
    ) -> List[Tuple[int, Ingredient]]:
        mode = self._current_sort_mode()
        if mode == "name":
            key_func = lambda pair: (pair[1].name.lower(), pair[0])
        elif mode == "family":
            key_func = lambda pair: (
                pair[1].family.lower(),
                pair[1].name.lower(),
                pair[0],
            )
        else:
            key_func = lambda pair: (
                pair[1].taste.lower(),
                pair[1].name.lower(),
                pair[0],
            )
        return sorted(hand_with_indices, key=key_func)

    def clear_hand(self) -> None:
        for view in self.card_views.values():
            view.destroy()
        self.card_views = {}

    def show_deck_popup(self) -> None:
        if not self.session:
            messagebox.showinfo(
                "No run in progress", "Start a run to view your ingredient basket."
            )
            return

        if self.deck_popup and self.deck_popup.winfo_exists():
            self.deck_popup.set_session(self.session)
            self.deck_popup.lift()
            self.deck_popup.focus_force()
            return

        def handle_close() -> None:
            self.deck_popup = None

        self.deck_popup = DeckPopup(self.root, self.session, on_close=handle_close)
        self.deck_popup.transient(self.root)
        self.deck_popup.focus_force()
        self._center_popup(self.deck_popup)

    def show_cookbook_panel(self) -> None:
        if not self.session:
            messagebox.showinfo(
                "Cookbook", "Start a run to discover recipes and view your cookbook."
            )
            return
        entries = self.session.get_cookbook()
        self._update_cookbook_button()
        if not entries:
            messagebox.showinfo(
                "Cookbook", "No recipes unlocked yet. Cook dishes to discover more."
            )
            return
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.set_entries(entries)
            self.cookbook_popup.lift()
            self.cookbook_popup.focus_force()
            return

        def handle_close() -> None:
            self.cookbook_popup = None

        self.cookbook_popup = CookbookPopup(
            self.root, entries, DATA, on_close=handle_close
        )
        self.cookbook_popup.transient(self.root)
        self._center_popup(self.cookbook_popup)
        self.cookbook_popup.focus_force()

    def show_dish_matrix(self) -> None:
        if self.dish_dialog and self.dish_dialog.winfo_exists():
            self.dish_dialog.lift()
            self.dish_dialog.focus_force()
            return

        def handle_close() -> None:
            self.dish_dialog = None

        self.dish_dialog = DishMatrixDialog(
            self.root, DATA.dish_matrix, on_close=handle_close
        )
        self.dish_dialog.transient(self.root)
        self._center_popup(self.dish_dialog)
        self.dish_dialog.focus_force()

    def show_selected_seasoning_info(self) -> None:
        if not self.session:
            messagebox.showinfo(
                "Seasonings", "Start a run to collect seasonings for your pantry."
            )
            return

        seasonings = self.session.get_seasonings()
        if not seasonings:
            messagebox.showinfo(
                "Seasonings", "No seasonings collected yet. Finish a round to claim one."
            )
            return

        if self.seasoning_popup and self.seasoning_popup.winfo_exists():
            self.seasoning_popup.set_seasonings(seasonings)
            self.seasoning_popup.lift()
            self.seasoning_popup.focus_force()
            return

        def handle_close() -> None:
            self.seasoning_popup = None

        self.seasoning_popup = SeasoningPopup(
            self.root, seasonings, on_close=handle_close
        )
        self.seasoning_popup.transient(self.root)
        self._center_popup(self.seasoning_popup)
        self.seasoning_popup.focus_force()

    def show_chef_team(self) -> None:
        self._update_chef_button()
        if not self.session:
            messagebox.showinfo(
                "Chef Team", "Start a run to recruit chefs and view your roster."
            )
            return
        if self.session.can_recruit_chef():
            self.show_recruit_dialog()
            return

        if not self.session.chefs:
            message = "No chefs recruited yet. Complete a cook to earn a new offer."
        else:
            lines = ["Active chefs:"]
            for chef in self.session.chefs:
                recipes = (
                    ", ".join(chef.recipe_names)
                    if getattr(chef, "recipe_names", None)
                    else "No signature recipes"
                )
                lines.append(f"• {chef.name} — {recipes}")
            message = "\n".join(lines)

        messagebox.showinfo("Chef Team", message)

    def _handle_seasoning_selected(self, seasoning: Optional[Seasoning]) -> None:
        self._update_seasoning_button(seasoning)

    def _update_cookbook_button(self) -> None:
        if not hasattr(self, "cookbook_button"):
            return

        if not self.session:
            count = 0
        else:
            count = len(self.session.get_cookbook())

        plural = "recipe" if count == 1 else "recipes"
        self.cookbook_count_var.set(f"Cookbook\n{count} {plural}")

        icon = self._resource_button_images.get("cookbook")
        if icon is None:
            icon = _generate_button_icon("cookbook", "CB")
            self._resource_button_images["cookbook"] = icon

        self.cookbook_button.configure(image=icon)

    def _update_basket_button(self) -> None:
        if not hasattr(self, "basket_button"):
            return

        if not self.session:
            self.basket_count_var.set("Basket\n0/0")
            return

        remaining, total = self.session.get_basket_counts()
        self.basket_count_var.set(f"Basket\n{remaining}/{total}")

    def _update_seasoning_button(self, seasoning: Optional[Seasoning] = None) -> None:
        if not hasattr(self, "seasoning_button"):
            return
        if seasoning is None and self.session:
            seasonings = self.session.get_seasonings()
            if len(seasonings) == 1:
                seasoning = seasonings[0]
            elif seasonings:
                seasoning = None
                text = "Seasonings\nMultiple"
            else:
                seasoning = None
                text = "Seasonings\nNone"
        if seasoning is not None:
            display_name = seasoning.display_name or seasoning.name
            text = f"Seasonings\n{display_name}"
        elif self.session and self.session.get_seasonings():
            text = "Seasonings\nMultiple"
        else:
            text = "Seasonings\nNone"

        icon = self._resource_button_images.get("seasoning")
        if icon is None:
            icon = _generate_button_icon("seasoning", "SN")
            self._resource_button_images["seasoning"] = icon

        self.seasoning_button.configure(image=icon, text=text)

    def _update_chef_button(self) -> None:
        if not hasattr(self, "chef_button"):
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
            icon = _generate_button_icon("chef", "CF")
            self._resource_button_images["chef"] = icon

        self.chef_button.configure(image=icon, text=text)

    def toggle_card(self, index: int) -> None:
        if not self.session:
            return
        view = self.card_views.get(index)
        if not view:
            return
        if index in self.selected_indices:
            self.selected_indices.remove(index)
            view.set_selected(False)
        else:
            if len(self.selected_indices) >= self.session.pick_size:
                messagebox.showinfo(
                    "Selection limit",
                    f"You may only pick {self.session.pick_size} cards per turn.",
                )
                return
            self.selected_indices.add(index)
            view.set_selected(True)
        self.update_selection_label()

    def _current_sort_mode(self) -> str:
        return self.hand_sort_modes[self.hand_sort_index]

    def _format_sort_label(self, mode: str) -> str:
        labels = {"name": "Name", "family": "Family", "taste": "Taste"}
        return f"Sort order: {labels.get(mode, mode.title())}"

    def cycle_hand_sort_mode(self) -> None:
        self.hand_sort_index = (self.hand_sort_index + 1) % len(self.hand_sort_modes)
        self.hand_sort_var.set(self._format_sort_label(self._current_sort_mode()))
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
        if not self.session or not self.selected_indices:
            self.selection_summary_var.set(default_text)
            return

        hand = self.session.get_hand()
        try:
            selected = [hand[index] for index in sorted(self.selected_indices)]
        except IndexError:
            self.selection_summary_var.set(default_text)
            return

        Value = sum(card.Value for card in selected)
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

    def update_status(self) -> None:
        if not self.session:
            return
        round_text = (
            f"Round {self.session.round_index} / {self.session.rounds}"
        )
        turn_text = (
            f"Turn {self.session.turn_number + 1} / {self.session.total_turns}"
            if not self.session.is_finished()
            else f"Turns completed: {self.session.turn_number}"
        )
        self.progress_var.set(f"{round_text} — {turn_text}")
        self.score_var.set(str(self.session.get_total_score()))
        chef_names = ", ".join(chef.name for chef in self.session.chefs) or "None"
        self.chefs_var.set(
            f"Active chefs ({self.session.max_chefs} max): {chef_names}"
        )
        self._update_chef_button()
        self._update_seasoning_button()

    def _append_log_lines(self, lines: Iterable[str]) -> None:
        collected = list(lines)
        if not collected:
            return
        self.log_text.configure(state="normal")
        for line in collected:
            self.log_text.insert("end", f"{line}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def append_events(self, messages: Iterable[str]) -> None:
        self._append_log_lines(f"• {message}" for message in messages)

    def clear_events(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _close_recruit_dialog(self) -> None:
        if self.recruit_dialog and self.recruit_dialog.winfo_exists():
            self.recruit_dialog.destroy()
        self.recruit_dialog = None

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
            return
        if not self.selected_indices:
            messagebox.showwarning(
                "Incomplete selection",
                "Select at least one ingredient before cooking.",
            )
            return

        try:
            indices = sorted(self.selected_indices)
            outcome = self.session.play_turn(indices)
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Unable to cook selection", str(exc))
            return

        self.selected_indices.clear()
        self.update_selection_label()

        summary = self._format_outcome(outcome)
        self.write_result(summary)

        cookbook_entries = self.session.get_cookbook()
        if self.cookbook_tile:
            self.cookbook_tile.set_entries(cookbook_entries)
        self._update_cookbook_button()
        if self.cookbook_popup and self.cookbook_popup.winfo_exists():
            self.cookbook_popup.set_entries(cookbook_entries)

        self.render_hand()
        self.update_status()
        self.log_turn_points(outcome)
        self.append_events(self.session.consume_events())
        self.show_turn_summary_popup(outcome)
        self.maybe_prompt_new_chef()

        if self.session.is_finished():
            self.cook_button.configure(state="disabled")
            self.discard_button.configure(state="disabled")
            self.basket_button.configure(state="disabled")
            self.seasoning_button.configure(state="disabled")
            self.chef_button.configure(state="disabled")
            self._set_controls_active(True)
            self._close_recruit_dialog()
            messagebox.showinfo(
                "Run complete",
                f"Final score: {self.session.get_total_score()}",
            )
            summary_text = self._final_summary_text()
            if summary_text:
                self.write_result(summary_text)

    def discard_selected(self) -> None:
        if not self.session:
            return
        if not self.selected_indices:
            messagebox.showwarning(
                "No selection",
                "Select at least one ingredient to discard.",
            )
            return

        indices = sorted(self.selected_indices)
        try:
            removed, deck_refreshed = self.session.discard_indices(indices)
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Unable to discard ingredient", str(exc))
            return

        self.selected_indices.clear()
        self.update_selection_label()
        self.render_hand()
        self.update_status()

        events = self.session.consume_events()
        self.append_events(events)

        if removed:
            names = ", ".join(ingredient.name for ingredient in removed)
            if deck_refreshed:
                message = f"Discarded {names}. Market deck refreshed."
            else:
                replacement_text = "replacements" if len(removed) > 1 else "a replacement"
                message = f"Discarded {names} and drew {replacement_text}."
        else:
            message = "No ingredient was discarded."
        self.write_result(message)

        if self.session.is_finished():
            self.cook_button.configure(state="disabled")
            self.discard_button.configure(state="disabled")
            self.basket_button.configure(state="disabled")
            self.seasoning_button.configure(state="disabled")
            self.chef_button.configure(state="disabled")
            self._set_controls_active(True)
            self._close_recruit_dialog()

    def show_turn_summary_popup(self, outcome: TurnOutcome) -> None:
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.destroy()
        self.active_popup = None

        popup = tk.Toplevel(self.root)
        popup.title("Turn Summary")
        popup.transient(self.root)
        popup.resizable(False, False)
        popup.configure(bg="#10151a")

        def close_popup() -> None:
            if popup.winfo_exists():
                popup.destroy()
            if self.active_popup is popup:
                self.active_popup = None

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
                f"Turn {outcome.turn_number}/{outcome.total_turns}"
                f"  •  Round {outcome.round_index}/{outcome.total_rounds}"
                f"  •  Cook {outcome.cook_number}/{outcome.cooks_per_round}"
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

            if outcome.discovered_recipe:
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

                icon_container: Optional[tk.Frame] = None
                taste_label: Optional[tk.Label] = None

                taste_icon = _load_icon(
                    "taste", ingredient.taste, target_px=DIALOG_ICON_TARGET_PX
                )
                if taste_icon is not None:
                    if icon_container is None:
                        icon_container = tk.Frame(row, bg=base_bg)
                        icon_container.pack(side="left", padx=(0, 8))
                    taste_label = tk.Label(
                        icon_container, image=taste_icon, bg=base_bg
                    )
                    taste_label.image = taste_icon
                    taste_label.pack(side="top")

                family_icon = _load_icon(
                    "family", ingredient.family, target_px=DIALOG_ICON_TARGET_PX
                )
                if family_icon is not None:
                    if icon_container is None:
                        icon_container = tk.Frame(row, bg=base_bg)
                        icon_container.pack(side="left", padx=(0, 8))
                    family_label = tk.Label(
                        icon_container, image=family_icon, bg=base_bg
                    )
                    family_label.image = family_icon
                    family_label.pack(side="top")
                    if taste_label is not None:
                        taste_label.pack_configure(pady=(0, 4))

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
        flavor_desc = outcome.flavor_pattern.replace("_", " ")
        points_lines = [
            f"Ingredient Value: {outcome.Value}",
            f"Family profile: {outcome.family_label} ({family_desc})",
            f"Taste profile: {outcome.flavor_label} ({flavor_desc})",
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
            points_lines.append("Deck refreshed for the next hand!")

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
            f"Turn {outcome.turn_number}/{outcome.total_turns} — "
            f"Round {outcome.round_index}/{outcome.total_rounds}, "
            f"Cook {outcome.cook_number}/{outcome.cooks_per_round}",
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
        parts.append(
            f"Taste profile: {outcome.flavor_label}"
            f" ({outcome.flavor_pattern.replace('_', ' ')})"
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
        parts.append(
            f"Score gained: {outcome.final_score:+d} ({base_text})"
        )
        parts.append(
            f"Chef key ingredients used: {outcome.chef_hits}/{max(len(outcome.selected), 1)}"
        )
        if outcome.deck_refreshed:
            parts.append("Deck refreshed for the next turn.")

        parts.append("")
        parts.append(f"Cumulative score: {self.session.get_total_score()}")
        return "\n".join(parts)


def main() -> None:
    root = tk.Tk()
    app = FoodGameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
