"""Desktop GUI harness for Food Deck Simulator.

This module provides a Tkinter-based application that reuses the shared
`food_api` rules so designers can explore the card game with a richer visual
interface.  The gameplay loop mirrors the CLI version in ``food_game.py``: pick
chefs, choose a market theme, draw ingredient hands, and cook trios to chase high
scores.  No networking is involved—everything runs in-process on top of the
existing data files.
"""
from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Iterable, List, Optional, Sequence, Tuple

from tkinter import ttk

from food_api import (
    DEFAULT_HAND_SIZE,
    DEFAULT_PICK_SIZE,
    DEFAULT_CHEFS_JSON,
    DEFAULT_INGREDIENTS_JSON,
    DEFAULT_RECIPES_JSON,
    DEFAULT_TASTE_JSON,
    DEFAULT_THEMES_JSON,
    Chef,
    GameData,
    Ingredient,
    SimulationConfig,
    build_market_deck,
)

DEFAULT_CONFIG = SimulationConfig()
DEFAULT_DECK_SIZE = DEFAULT_CONFIG.deck_size
DEFAULT_BIAS = DEFAULT_CONFIG.bias

ASSET_DIR = Path(__file__).resolve().parent


def _load_game_data() -> GameData:
    """Load the shared JSON data using paths relative to this file."""

    return GameData.from_json(
        ingredients_path=str(ASSET_DIR / DEFAULT_INGREDIENTS_JSON),
        recipes_path=str(ASSET_DIR / DEFAULT_RECIPES_JSON),
        chefs_path=str(ASSET_DIR / DEFAULT_CHEFS_JSON),
        taste_path=str(ASSET_DIR / DEFAULT_TASTE_JSON),
        themes_path=str(ASSET_DIR / DEFAULT_THEMES_JSON),
    )


DATA = _load_game_data()


def chef_marker(chef: Chef) -> str:
    parts = [part for part in chef.name.split() if part]
    for part in parts:
        if part.lower() != "chef":
            return part[0].upper()
    return parts[0][0].upper() if parts else "?"


@dataclass
class TurnOutcome:
    selected: Sequence[Ingredient]
    chips: int
    taste_sum: int
    taste_multiplier: int
    recipe_name: Optional[str]
    recipe_multiplier: float
    contributions: List[Tuple[str, float]]
    total_multiplier: float
    final_score: int
    base_score: int
    chef_hits: int
    round_index: int
    total_rounds: int
    cook_number: int
    cooks_per_round: int
    turn_number: int
    total_turns: int
    deck_refreshed: bool


class GameSession:
    """Manage deck, hand, and scoring for a single run."""

    def __init__(
        self,
        data: GameData,
        theme_name: str,
        chefs: Sequence[Chef],
        rounds: int,
        cooks_per_round: int,
        hand_size: int,
        pick_size: int,
        deck_size: int = DEFAULT_DECK_SIZE,
        bias: float = DEFAULT_BIAS,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not chefs:
            raise ValueError("At least one chef must be selected for a run.")
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

        self.data = data
        self.theme_name = theme_name
        self.chefs = list(chefs)
        self.rounds = rounds
        self.cooks_per_round = cooks_per_round
        self.hand_size = hand_size
        self.pick_size = pick_size
        self.deck_size = deck_size
        self.bias = bias
        self.rng = rng or random.Random()

        self.total_turns = rounds * cooks_per_round
        self.turn_number = 0
        self.round_index = 0
        self.cooks_completed_in_round = 0
        self.total_score = 0
        self.finished = False

        self.hand: List[Ingredient] = []
        self.deck: List[Ingredient] = []
        self._events: List[str] = []

        self._chef_key_map = {
            chef.name: data.chef_key_ingredients(chef) for chef in self.chefs
        }
        self._chef_key_set = data.chefs_key_ingredients(self.chefs)

        self._start_next_round(initial=True)

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
            self.theme_name,
            self.chefs,
            deck_size=self.deck_size,
            bias=self.bias,
            rng=self.rng,
        )
        self.rng.shuffle(self.deck)
        self.hand.clear()
        self._push_event(
            f"Round {self.round_index}/{self.rounds} begins. Deck shuffled for the team."
        )
        self._refill_hand()

    def _refill_hand(self) -> bool:
        needed = self.hand_size - len(self.hand)
        deck_refreshed = False
        while needed > 0 and not self.finished:
            if not self.deck:
                self.deck = build_market_deck(
                    self.data,
                    self.theme_name,
                    self.chefs,
                    deck_size=self.deck_size,
                    bias=self.bias,
                    rng=self.rng,
                )
                self.rng.shuffle(self.deck)
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

    # ----------------- Public API -----------------
    def get_hand(self) -> Sequence[Ingredient]:
        return list(self.hand)

    def get_total_score(self) -> int:
        return self.total_score

    def get_selection_markers(self, ingredient: Ingredient) -> str:
        markers = [
            chef_marker(chef)
            for chef in self.chefs
            if ingredient.name in self._chef_key_map.get(chef.name, set())
        ]
        return "".join(markers)

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
        base_score, chips, taste_sum, taste_multiplier = self.data.trio_score(selected)
        recipe_name = self.data.which_recipe(selected)
        recipe_multiplier, contributions = self._team_recipe_multiplier(recipe_name)
        total_multiplier = taste_multiplier * recipe_multiplier
        final_score = int(round(base_score * recipe_multiplier))
        chef_hits = sum(1 for ing in selected if ing.name in self._chef_key_set)

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
            chips=chips,
            taste_sum=taste_sum,
            taste_multiplier=taste_multiplier,
            recipe_name=recipe_name,
            recipe_multiplier=recipe_multiplier,
            contributions=contributions,
            total_multiplier=total_multiplier,
            final_score=final_score,
            base_score=base_score,
            chef_hits=chef_hits,
            round_index=current_round,
            total_rounds=self.rounds,
            cook_number=current_cook,
            cooks_per_round=self.cooks_per_round,
            turn_number=current_turn,
            total_turns=self.total_turns,
            deck_refreshed=deck_refreshed,
        )

    def _team_recipe_multiplier(
        self, recipe_name: Optional[str]
    ) -> Tuple[float, List[Tuple[str, float]]]:
        if not recipe_name:
            return 1.0, []
        total = 1.0
        contributions: List[Tuple[str, float]] = []
        for chef in self.chefs:
            multiplier = self._chef_recipe_multiplier(chef, recipe_name)
            total *= multiplier
            if not math.isclose(multiplier, 1.0):
                contributions.append((chef.name, multiplier))
        return total, contributions

    @staticmethod
    def _chef_recipe_multiplier(chef: Chef, recipe_name: Optional[str]) -> float:
        if not recipe_name:
            return 1.0
        multipliers = chef.perks.get("recipe_multipliers", {})
        try:
            return float(multipliers.get(recipe_name, 1.0))
        except (TypeError, ValueError):
            return 1.0

    def is_finished(self) -> bool:
        return self.finished


class ChefTile(ttk.Frame):
    def __init__(self, master: tk.Widget, data: GameData) -> None:
        super().__init__(master, style="Tile.TFrame", padding=(14, 12))
        self.data = data
        self.chef: Optional[Chef] = None
        self.expanded = False
        self._current_recipe_names: list[str] = []
        self._recipe_buttons: list[ttk.Button] = []
        self._empty_label: Optional[ttk.Label] = None

        self.columnconfigure(0, weight=1)

        self.name_var = tk.StringVar(value="Chef preview")
        self.subtitle_var = tk.StringVar(
            value="Select a chef from the list to preview their recipes."
        )

        self.header_button = ttk.Button(
            self,
            textvariable=self.name_var,
            style="TileHeader.TButton",
            command=self.toggle_recipes,
        )
        self.header_button.grid(row=0, column=0, sticky="ew")

        self.subtitle_label = ttk.Label(
            self,
            textvariable=self.subtitle_var,
            style="TileSub.TLabel",
            wraplength=260,
            justify="left",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.recipe_container = ttk.Frame(self, style="TileBody.TFrame")
        self.recipe_container.columnconfigure(0, weight=1)
        self.recipe_container.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.recipe_container.grid_remove()

        self.recipe_buttons_frame = ttk.Frame(
            self.recipe_container, style="TileBody.TFrame"
        )
        self.recipe_buttons_frame.grid(row=0, column=0, sticky="ew")
        self.recipe_buttons_frame.columnconfigure(0, weight=1)

        self.ingredients_var = tk.StringVar(value="")
        self.ingredients_label = ttk.Label(
            self.recipe_container,
            textvariable=self.ingredients_var,
            style="TileInfo.TLabel",
            wraplength=260,
            justify="left",
        )
        self.ingredients_label.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.set_chef(None)

    def set_chef(self, chef: Optional[Chef]) -> None:
        self.chef = chef
        self.expanded = False
        self._clear_recipe_buttons()
        self.recipe_container.grid_remove()
        self._current_recipe_names = []
        self.ingredients_var.set("")

        if not chef:
            self.name_var.set("Chef preview")
            self.subtitle_var.set(
                "Select a chef from the list to preview their recipes."
            )
            self.header_button.state(["disabled"])
            return

        self.header_button.state(["!disabled"])
        self.name_var.set(chef.name)
        self._current_recipe_names = sorted(set(chef.recipe_names))
        self._populate_recipe_buttons()
        self._update_collapsed_hint()

    def toggle_recipes(self) -> None:
        if not self.chef:
            return
        if self.expanded:
            self.recipe_container.grid_remove()
            self.expanded = False
            self._update_collapsed_hint()
        else:
            self.recipe_container.grid()
            self.expanded = True
            if self._current_recipe_names:
                self.subtitle_var.set("Select a recipe to view its ingredients.")
            else:
                self.subtitle_var.set("No signature recipes for this chef.")

    def show_ingredients(self, recipe_name: str) -> None:
        if not self.chef:
            return
        recipe = self.data.recipe_by_name.get(recipe_name)
        if not recipe:
            self.ingredients_var.set("Recipe data unavailable.")
            return
        if not self.expanded:
            self.toggle_recipes()
        lines = "\n".join(f"• {ingredient}" for ingredient in recipe.trio)
        self.ingredients_var.set(f"Ingredients:\n{lines}")
        self.subtitle_var.set(f"{recipe_name} ingredients:")

    def _clear_recipe_buttons(self) -> None:
        for button in self._recipe_buttons:
            button.destroy()
        self._recipe_buttons.clear()
        if self._empty_label:
            self._empty_label.destroy()
            self._empty_label = None

    def _populate_recipe_buttons(self) -> None:
        if not self.chef:
            return
        if not self._current_recipe_names:
            self._empty_label = ttk.Label(
                self.recipe_buttons_frame,
                text="This chef does not host any signature recipes.",
                style="TileInfo.TLabel",
                wraplength=260,
                justify="left",
            )
            self._empty_label.grid(row=0, column=0, sticky="ew")
            return

        for row, recipe_name in enumerate(self._current_recipe_names):
            button = ttk.Button(
                self.recipe_buttons_frame,
                text=recipe_name,
                style="TileRecipe.TButton",
                command=lambda name=recipe_name: self.show_ingredients(name),
            )
            button.grid(row=row, column=0, sticky="ew", pady=2)
            self._recipe_buttons.append(button)
        self.ingredients_var.set("Select a recipe to view its ingredients.")

    def _update_collapsed_hint(self) -> None:
        if not self.chef:
            self.subtitle_var.set(
                "Select a chef from the list to preview their recipes."
            )
            return
        count = len(self._current_recipe_names)
        if count:
            plural = "recipe" if count == 1 else "recipes"
            self.subtitle_var.set(f"{count} {plural}. Click to view.")
        else:
            self.subtitle_var.set("No signature recipes for this chef.")


class CardView(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        index: int,
        ingredient: Ingredient,
        marker_text: str,
        on_click,
    ) -> None:
        super().__init__(master, style="Card.TFrame", padding=(10, 8))
        self.index = index
        self.on_click = on_click
        self.selected = False

        self.columnconfigure(0, weight=1)

        self.name_label = ttk.Label(
            self,
            text=ingredient.name,
            style="CardTitle.TLabel",
            anchor="center",
            justify="center",
        )
        self.name_label.grid(row=0, column=0, sticky="ew")

        separator = ttk.Separator(self, orient="horizontal")
        separator.grid(row=1, column=0, sticky="ew", pady=(6, 8))

        self.taste_label = ttk.Label(
            self, text=f"Taste: {ingredient.taste}", style="CardBody.TLabel"
        )
        self.taste_label.grid(row=2, column=0, sticky="w")

        self.chips_label = ttk.Label(
            self, text=f"Chips: {ingredient.chips}", style="CardBody.TLabel"
        )
        self.chips_label.grid(row=3, column=0, sticky="w", pady=(2, 0))

        self.marker_label: Optional[ttk.Label]
        self.marker_label = None
        if marker_text:
            self.marker_label = ttk.Label(
                self,
                text=f"Chef Key: {marker_text}",
                style="CardMarker.TLabel",
            )
            self.marker_label.grid(row=4, column=0, sticky="w", pady=(4, 0))

        self.bind("<Button-1>", self._handle_click)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._handle_click)

    def _handle_click(self, _event) -> None:
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
        self.name_label.configure(style=title_style)
        self.taste_label.configure(style=body_style)
        self.chips_label.configure(style=body_style)
        if self.marker_label:
            self.marker_label.configure(style=marker_style)


class FoodGameApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Food Deck Simulator — Desktop Prototype")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)

        self.session: Optional[GameSession] = None
        self.card_views: List[CardView] = []
        self.selected_indices: set[int] = set()
        self.spinboxes: List[ttk.Spinbox] = []
        self.chef_tile: Optional[ChefTile] = None
        self.active_popup: Optional[tk.Toplevel] = None

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

        style.configure("Info.TLabel", font=("Helvetica", 10), foreground="#2f2f2f")
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground="#1f1f1f")
        style.configure("Score.TLabel", font=("Helvetica", 18, "bold"), foreground="#1f1f1f")
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
            "TileRecipe.TButton",
            font=("Helvetica", 10),
            anchor="w",
            padding=(6, 4),
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
        self.game_frame.rowconfigure(2, weight=1)

        self._build_controls()
        self._build_game_panel()

    def _build_controls(self) -> None:
        ttk.Label(self.control_frame, text="Theme", style="Header.TLabel").pack(anchor="w")
        self.theme_var = tk.StringVar()
        theme_names = sorted(DATA.themes.keys())
        self.theme_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.theme_var,
            values=theme_names,
            state="readonly",
            width=28,
        )
        if theme_names:
            self.theme_combo.current(0)
        self.theme_combo.pack(anchor="w", pady=(4, 12))

        ttk.Label(self.control_frame, text="Chefs", style="Header.TLabel").pack(anchor="w")
        self.chef_list = tk.Listbox(
            self.control_frame,
            selectmode="multiple",
            exportselection=False,
            width=28,
            height=10,
        )
        for chef in DATA.chefs:
            self.chef_list.insert("end", chef.name)
        self.chef_list.pack(anchor="w", pady=(4, 12))
        self.chef_tile = ChefTile(self.control_frame, DATA)
        self.chef_tile.pack(fill="x", pady=(0, 12))
        self.chef_list.bind("<<ListboxSelect>>", self.on_chef_select)

        config_frame = ttk.Frame(self.control_frame)
        config_frame.pack(anchor="w", pady=(8, 0))

        self.round_var = tk.IntVar(value=DEFAULT_CONFIG.rounds)
        self.cooks_var = tk.IntVar(value=DEFAULT_CONFIG.cooks)
        self.hand_var = tk.IntVar(value=DEFAULT_HAND_SIZE)
        self.pick_var = tk.IntVar(value=DEFAULT_PICK_SIZE)

        self._add_spinbox(config_frame, "Rounds", self.round_var, 1, 10)
        self._add_spinbox(config_frame, "Cooks / Round", self.cooks_var, 1, 12)
        self._add_spinbox(config_frame, "Hand Size", self.hand_var, 3, 10)
        self._add_spinbox(config_frame, "Pick Size", self.pick_var, 1, 5)

        self.start_button = ttk.Button(
            self.control_frame, text="Start Run", command=self.start_run
        )
        self.start_button.pack(fill="x", pady=(12, 6))

        self.reset_button = ttk.Button(
            self.control_frame, text="Reset", command=self.reset_session, state="disabled"
        )
        self.reset_button.pack(fill="x")

        ttk.Label(
            self.control_frame,
            text="Hold Ctrl/Cmd to select multiple chefs.",
            style="Info.TLabel",
        ).pack(anchor="w", pady=(12, 0))

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

        self.selection_summary_var = tk.StringVar(value="TASTE X CHIPS = POINTS")
        ttk.Label(
            score_frame,
            textvariable=self.selection_summary_var,
            style="Info.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        self.chefs_var = tk.StringVar(value="Active chefs: —")
        ttk.Label(score_frame, textvariable=self.chefs_var, style="Info.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self.events_text = tk.Text(
            self.game_frame,
            height=5,
            wrap="word",
            background="#f8f8f8",
            foreground="#242424",
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=8,
            font=("Helvetica", 10),
        )
        self.events_text.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        self.events_text.configure(state="disabled")

        hand_container = ttk.Frame(self.game_frame)
        hand_container.grid(row=2, column=0, sticky="nsew")
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
        action_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        action_frame.columnconfigure(0, weight=1)

        self.cook_button = ttk.Button(
            action_frame,
            text="COOK",
            command=self.cook_selected,
            state="disabled",
        )
        self.cook_button.grid(row=0, column=0, sticky="ew")

        self.result_text = tk.Text(
            self.game_frame,
            height=12,
            wrap="word",
            background="#ffffff",
            foreground="#202020",
            relief="solid",
            borderwidth=1,
            padx=12,
            pady=10,
            font=("Helvetica", 10),
        )
        self.result_text.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.result_text.configure(state="disabled")

    # ----------------- Session management -----------------
    def start_run(self) -> None:
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.destroy()
        self.active_popup = None
        try:
            theme = self.theme_var.get()
            if not theme:
                raise ValueError("Select a theme before starting a run.")

            selected_indices = list(self.chef_list.curselection())
            if not selected_indices:
                raise ValueError("Select at least one chef (Ctrl/Cmd + click for multiples).")
            chefs = [DATA.chefs[index] for index in selected_indices]

            rounds = int(self.round_var.get())
            cooks = int(self.cooks_var.get())
            hand_size = int(self.hand_var.get())
            pick_size = int(self.pick_var.get())

            if pick_size > hand_size:
                raise ValueError("Pick size cannot exceed hand size.")

            self.session = GameSession(
                DATA,
                theme_name=theme,
                chefs=chefs,
                rounds=rounds,
                cooks_per_round=cooks,
                hand_size=hand_size,
                pick_size=pick_size,
            )
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Cannot start run", str(exc))
            return

        self._set_controls_active(False)
        self.cook_button.configure(state="normal")
        self.reset_button.configure(state="normal")
        self.selected_indices.clear()
        self.update_selection_label()
        self.render_hand()
        self.update_status()
        self.append_events(self.session.consume_events())
        self.write_result("Run started. Select ingredients and press COOK!")

    def reset_session(self) -> None:
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.destroy()
        self.active_popup = None
        self.session = None
        self.selected_indices.clear()
        self.update_selection_label()
        self.score_var.set("0")
        self.progress_var.set("Round 0 / 0 — Turn 0 / 0")
        self.chefs_var.set("Active chefs: —")
        self.cook_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self._set_controls_active(True)
        self.clear_hand()
        self.clear_events()
        self.write_result("Session reset. Configure options and start a new run.")

    def _set_controls_active(self, active: bool) -> None:
        state = "normal" if active else "disabled"
        self.theme_combo.configure(state="readonly" if active else "disabled")
        self.chef_list.configure(state=state)
        for spin in self.spinboxes:
            spin.configure(state=state)
        self.start_button.configure(state="normal" if active else "disabled")

    def on_chef_select(self, _event: Optional[tk.Event] = None) -> None:
        if not self.chef_tile:
            return
        selection = self.chef_list.curselection()
        chef: Optional[Chef] = None
        if selection:
            try:
                active_index = int(self.chef_list.index("active"))
            except (tk.TclError, ValueError):
                active_index = selection[-1]
            index = active_index if active_index in selection else selection[-1]
            try:
                chef = DATA.chefs[index]
            except IndexError:
                chef = None
        self.chef_tile.set_chef(chef)

    # ----------------- UI updates -----------------
    def render_hand(self) -> None:
        self.clear_hand()
        if not self.session:
            return

        hand = self.session.get_hand()
        for index, ingredient in enumerate(hand):
            markers = self.session.get_selection_markers(ingredient)
            view = CardView(
                self.hand_frame,
                index=index,
                ingredient=ingredient,
                marker_text=markers,
                on_click=self.toggle_card,
            )
            view.grid(row=0, column=index, sticky="nw", padx=8, pady=8)
            self.card_views.append(view)

        self.hand_frame.update_idletasks()
        self.hand_canvas.configure(scrollregion=self.hand_canvas.bbox("all"))

    def clear_hand(self) -> None:
        for view in self.card_views:
            view.destroy()
        self.card_views.clear()

    def toggle_card(self, index: int) -> None:
        if not self.session:
            return
        if index in self.selected_indices:
            self.selected_indices.remove(index)
            self.card_views[index].set_selected(False)
        else:
            if len(self.selected_indices) >= self.session.pick_size:
                messagebox.showinfo(
                    "Selection limit",
                    f"You may only pick {self.session.pick_size} cards per turn.",
                )
                return
            self.selected_indices.add(index)
            self.card_views[index].set_selected(True)
        self.update_selection_label()

    def update_selection_label(self) -> None:
        count = len(self.selected_indices)
        limit = self.session.pick_size if self.session else 0
        if limit:
            self.selection_var.set(f"Selection: {count} / {limit}")
        else:
            self.selection_var.set(f"Selection: {count}")
        self.update_selection_summary()

    def update_selection_summary(self) -> None:
        default_text = "TASTE X CHIPS = POINTS"
        if not self.session or not self.selected_indices:
            self.selection_summary_var.set(default_text)
            return

        hand = self.session.get_hand()
        try:
            selected = [hand[index] for index in sorted(self.selected_indices)]
        except IndexError:
            self.selection_summary_var.set(default_text)
            return

        base_score, chips, _taste_sum, taste_multiplier = self.session.data.trio_score(
            selected
        )
        summary = f"TASTE {taste_multiplier} X CHIPS {chips} = {base_score}"
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
        chef_names = ", ".join(chef.name for chef in self.session.chefs)
        self.chefs_var.set(f"Active chefs: {chef_names}")

    def append_events(self, messages: Iterable[str]) -> None:
        if not messages:
            return
        self.events_text.configure(state="normal")
        for message in messages:
            self.events_text.insert("end", f"• {message}\n")
        self.events_text.see("end")
        self.events_text.configure(state="disabled")

    def clear_events(self) -> None:
        self.events_text.configure(state="normal")
        self.events_text.delete("1.0", "end")
        self.events_text.configure(state="disabled")

    def write_result(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

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

        self.render_hand()
        self.update_status()
        self.append_events(self.session.consume_events())
        self.show_turn_summary_popup(outcome)

        if self.session.is_finished():
            self.cook_button.configure(state="disabled")
            self._set_controls_active(True)
            messagebox.showinfo(
                "Run complete",
                f"Final score: {self.session.get_total_score()}",
            )

    @staticmethod
    def _format_multiplier(multiplier: float) -> str:
        if math.isclose(multiplier, round(multiplier)):
            return f"x{int(round(multiplier))}"
        return f"x{multiplier:.2f}"

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

        if outcome.recipe_name:
            recipe_banner = tk.Label(
                content,
                text=f"✨ Recipe completed: {outcome.recipe_name}! ✨",
                font=("Helvetica", 12, "bold"),
                fg="#a35d00",
                bg=base_bg,
                anchor="w",
                justify="left",
            )
            recipe_banner.pack(anchor="w", pady=(0, 8))

        ingredient_lines = [
            f"• {ingredient.name} — Taste {ingredient.taste}, Chips {ingredient.chips}"
            for ingredient in outcome.selected
        ]
        ingredients_text = "\n".join(ingredient_lines) or "• —"

        tk.Label(
            content,
            text="Cooked ingredients:",
            font=("Helvetica", 11, "bold"),
            fg="#1d2730",
            bg=base_bg,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            content,
            text=ingredients_text,
            font=("Helvetica", 10),
            fg="#26313a",
            bg=base_bg,
            justify="left",
            anchor="w",
        ).pack(anchor="w", fill="x", pady=(0, 10))

        points_lines = [
            f"Chips: {outcome.chips}",
            f"Taste sum: {outcome.taste_sum}",
            f"Taste multiplier: {self._format_multiplier(outcome.taste_multiplier)}",
            f"Recipe multiplier: {self._format_multiplier(outcome.recipe_multiplier)}",
            f"Total multiplier: {self._format_multiplier(outcome.total_multiplier)}",
            f"Base score (Taste × Chips): {outcome.base_score}",
            f"Score gained this turn: {outcome.final_score}",
        ]

        if outcome.contributions:
            points_lines.append("")
            points_lines.append("Recipe bonus breakdown:")
            for name, multiplier in outcome.contributions:
                points_lines.append(
                    f"  • {name}: {self._format_multiplier(multiplier)}"
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

        ttk.Button(footer, text="Close", command=close_popup).pack(
            side="right", pady=(6, 0)
        )

        popup.update_idletasks()
        self._center_popup(popup)

        if outcome.recipe_name:
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
                f"  • {ingredient.name} (Taste: {ingredient.taste}, Chips: {ingredient.chips})"
            )
        parts.extend(
            [
                "",
                f"Total chips: {outcome.chips}",
                f"Taste synergy sum: {outcome.taste_sum}",
                f"Taste multiplier: x{outcome.taste_multiplier}",
                (
                    "TASTE X CHIPS = "
                    f"{outcome.taste_multiplier} X {outcome.chips} = {outcome.base_score}"
                ),
            ]
        )
        if outcome.recipe_name:
            parts.append(f"Recipe completed: {outcome.recipe_name}")
        else:
            parts.append("No recipe completed this turn.")
        parts.append(
            f"Score gained: {outcome.final_score} (base before recipe bonus: {outcome.base_score})"
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
