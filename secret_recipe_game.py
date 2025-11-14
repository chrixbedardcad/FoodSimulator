"""Prototype mini-game for matching secret recipes by ingredients.

This lightweight Tkinter experience reuses the Food Simulator data model to
build a short guessing game. Players receive a hand of eight ingredient cards,
try to spot which ingredient belongs to an unseen recipe, and press the Cook
button to test their guess. Finding three recipes ends the game and presents a
summary screen with attempt and timing stats.
"""
from __future__ import annotations

import random
import time
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Set

import tkinter as tk

from food_api import GameData, Ingredient, Recipe

HAND_SIZE = 8
RECIPES_TO_FIND = 3
APP_VERSION = "V0.02"
APP_WIDTH = 960
APP_HEIGHT = 780
TARGET_WRAP_LENGTH = 760
STATUS_WRAP_LENGTH = 760
SUMMARY_NAME_WRAP = 200
DETAIL_WRAP_LENGTH = 440
DETAIL_WINDOW_WIDTH = 560
DETAIL_WINDOW_HEIGHT = 520
ASSETS_ROOT = Path(__file__).resolve().parent
INGREDIENT_ICON_DIR = ASSETS_ROOT / "Ingredients"
RECIPE_ICON_DIR = ASSETS_ROOT / "recipes"
PLACEHOLDER_RECIPE_IMAGE = RECIPE_ICON_DIR / "emptydish.png"
IMAGE_TARGET_SIZE = 180


def _load_pil_modules() -> tuple[Optional[object], Optional[object]]:
    """Load Pillow modules if available without raising import errors."""

    image_spec = importlib.util.find_spec("PIL.Image")
    imagetk_spec = importlib.util.find_spec("PIL.ImageTk")
    if image_spec is None or imagetk_spec is None:
        return None, None
    image_module = importlib.import_module("PIL.Image")
    imagetk_module = importlib.import_module("PIL.ImageTk")
    return image_module, imagetk_module


PIL_Image, PIL_ImageTk = _load_pil_modules()
if PIL_Image:
    resampling_attr = getattr(PIL_Image, "Resampling", None)
    if resampling_attr is not None:
        PIL_RESAMPLE = getattr(resampling_attr, "LANCZOS", getattr(resampling_attr, "BICUBIC", None))
    else:
        PIL_RESAMPLE = getattr(PIL_Image, "LANCZOS", getattr(PIL_Image, "BICUBIC", None))
else:
    PIL_RESAMPLE = None


def _normalize_key(name: str) -> str:
    """Normalize names so they align with asset file stems."""

    return "".join(char.lower() for char in name if char.isalnum())


class Tooltip:
    """Simple tooltip helper for Tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str = "") -> None:
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def set_text(self, text: str) -> None:
        self.text = text

    def _show(self, event: Optional[tk.Event[tk.Misc]] = None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        self.tipwindow.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tipwindow,
            text=self.text,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        label.pack(ipadx=4)

    def _hide(self, event: Optional[tk.Event[tk.Misc]] = None) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class SecretRecipeGame:
    """Manage state and UI for the secret recipe guessing prototype."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Secret Recipe Hunt {APP_VERSION}")
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.root.resizable(False, False)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.data = GameData.from_json()
        self.rng = random.Random()
        self.pending_recipes: List[Recipe] = self._pick_recipes()
        self.current_recipe: Optional[Recipe] = None
        self.current_trio: set[str] = set()
        self.current_hand: List[str] = []
        self.selected_indices: Set[int] = set()
        self.attempts = 0
        self.start_time = time.perf_counter()

        self.ingredient_images: Dict[str, tk.PhotoImage] = {}
        self.recipe_images: Dict[str, tk.PhotoImage] = {}
        self.ingredient_image_paths = self._scan_asset_directory(INGREDIENT_ICON_DIR)
        self.recipe_image_paths = self._scan_asset_directory(RECIPE_ICON_DIR)
        self.blank_card_image = self._build_placeholder_image(size=IMAGE_TARGET_SIZE)
        self.placeholder_recipe_image = self._load_photo(PLACEHOLDER_RECIPE_IMAGE)

        self.status_var = tk.StringVar(value="Select ingredients and press Cook!")
        self.target_recipe_var = tk.StringVar(value="Find this recipe: ???")
        self.recipe_name_vars: List[tk.StringVar] = []
        self.found_recipes: List[Optional[Recipe]] = [None] * RECIPES_TO_FIND
        self._build_layout()
        self._start_next_round()

    # --- UI assembly helpers -------------------------------------------------
    def _build_layout(self) -> None:
        wrapper = tk.Frame(self.root, padx=16, pady=16)
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.grid_columnconfigure(0, weight=1)

        cards_frame = tk.Frame(wrapper)
        cards_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(0, 16))

        summary_header = tk.Label(
            cards_frame,
            text="Recipes Found",
            font=("Segoe UI", 14, "bold"),
        )
        summary_header.grid(row=0, column=0, columnspan=4, pady=(0, 6))

        summary_frame = tk.Frame(cards_frame)
        summary_frame.grid(row=1, column=0, columnspan=4, pady=(0, 8))
        self.recipe_slots: List[tk.Label] = []
        for index in range(RECIPES_TO_FIND):
            slot_container = tk.Frame(summary_frame)
            slot_container.grid(row=0, column=index, padx=8)
            slot = tk.Label(
                slot_container,
                image=self.placeholder_recipe_image,
                width=IMAGE_TARGET_SIZE,
                height=IMAGE_TARGET_SIZE,
            )
            slot.image = self.placeholder_recipe_image
            slot.grid(row=0, column=0)
            slot.bind("<Button-1>", lambda event, idx=index: self._show_recipe_details(idx))
            name_var = tk.StringVar(value="???")
            name_label = tk.Label(
                slot_container,
                textvariable=name_var,
                font=("Segoe UI", 11, "bold"),
                wraplength=SUMMARY_NAME_WRAP,
                justify="center",
            )
            name_label.grid(row=1, column=0, pady=(6, 0))
            self.recipe_slots.append(slot)
            self.recipe_name_vars.append(name_var)

        card_grid = tk.Frame(cards_frame)
        card_grid.grid(row=2, column=0, columnspan=4, pady=(12, 0))
        self.card_buttons: List[tk.Button] = []
        self.card_tooltips: List[Tooltip] = []
        self.default_highlight = "#bcbcbc"
        self.selected_highlight = "#4caf50"
        self.disabled_highlight = "#c8c8c8"

        for idx in range(HAND_SIZE):
            button = tk.Button(
                card_grid,
                image=self.blank_card_image,
                width=IMAGE_TARGET_SIZE,
                height=IMAGE_TARGET_SIZE,
                relief=tk.RAISED,
                borderwidth=2,
                highlightthickness=1,
                highlightbackground=self.default_highlight,
                highlightcolor=self.default_highlight,
                command=lambda index=idx: self._select_card(index),
            )
            button.image = self.blank_card_image
            row, column = divmod(idx, 4)
            button.grid(row=row, column=column, padx=6, pady=6)
            tooltip = Tooltip(button)
            self.card_buttons.append(button)
            self.card_tooltips.append(tooltip)

        target_frame = tk.Frame(wrapper)
        target_frame.grid(row=0, column=0, columnspan=4, pady=(0, 16))
        target_label = tk.Label(
            target_frame,
            textvariable=self.target_recipe_var,
            font=("Segoe UI", 20, "bold"),
            wraplength=TARGET_WRAP_LENGTH,
        )
        target_label.grid(row=0, column=0)

        controls = tk.Frame(wrapper)
        controls.grid(row=2, column=0, columnspan=4, pady=(4, 0), sticky="ew")
        controls.grid_columnconfigure(0, weight=1)

        self.cook_button = tk.Button(
            controls,
            text="Cook",
            font=("Segoe UI", 12, "bold"),
            command=self._cook_selected,
            state=tk.DISABLED,
            width=12,
        )
        self.cook_button.grid(row=0, column=0, padx=6, pady=(0, 6), sticky="w")

        self.status_label = tk.Label(
            controls,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=STATUS_WRAP_LENGTH,
        )
        self.status_label.grid(row=1, column=0, padx=6, sticky="ew")

    # --- Asset helpers -------------------------------------------------------
    def _scan_asset_directory(self, directory: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        if not directory.exists():
            return paths
        for image_path in directory.glob("*.png"):
            key = _normalize_key(image_path.stem)
            paths[key] = image_path
        return paths

    def _build_placeholder_image(self, size: int = IMAGE_TARGET_SIZE) -> tk.PhotoImage:
        image = tk.PhotoImage(master=self.root, width=size, height=size)
        image.put("#d9d9d9", to=(0, 0, size, size))
        return image

    def _load_photo(self, path: Path) -> tk.PhotoImage:
        if not path or not path.exists():
            return self.blank_card_image
        if PIL_Image and PIL_ImageTk:
            pil_image = PIL_Image.open(path).convert("RGBA")
            resample = PIL_RESAMPLE if PIL_RESAMPLE is not None else 1
            pil_image.thumbnail((IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), resample)
            return PIL_ImageTk.PhotoImage(pil_image, master=self.root)
        image = tk.PhotoImage(master=self.root, file=str(path))
        width, height = image.width(), image.height()
        scale = max(width // IMAGE_TARGET_SIZE, height // IMAGE_TARGET_SIZE, 1)
        if scale > 1:
            image = image.subsample(scale, scale)
        return image

    def _ingredient_image(self, name: str) -> tk.PhotoImage:
        key = _normalize_key(name)
        if key not in self.ingredient_images:
            path = self.ingredient_image_paths.get(key)
            self.ingredient_images[key] = self._load_photo(path) if path else self.blank_card_image
        return self.ingredient_images[key]

    def _recipe_image(self, name: str) -> tk.PhotoImage:
        key = _normalize_key(name)
        if key not in self.recipe_images:
            path = self.recipe_image_paths.get(key, PLACEHOLDER_RECIPE_IMAGE)
            self.recipe_images[key] = self._load_photo(path)
        return self.recipe_images[key]

    # --- Game flow -----------------------------------------------------------
    def _pick_recipes(self) -> List[Recipe]:
        pool = [recipe for recipe in self.data.recipes if len(recipe.trio) <= HAND_SIZE]
        if len(pool) < RECIPES_TO_FIND:
            raise RuntimeError("Not enough recipes to start the game.")
        return self.rng.sample(pool, RECIPES_TO_FIND)

    def _start_next_round(self) -> None:
        if not self.pending_recipes:
            self._finish_game()
            return
        self.current_recipe = self.pending_recipes.pop(0)
        self.current_trio = set(self.current_recipe.trio)
        self._deal_hand(self._build_hand(self.current_recipe))
        ingredient_total = len(self.current_recipe.trio)
        self.status_var.set(
            f"A new secret recipe awaits. Gather the {ingredient_total} key ingredients."
        )
        display_name = self.current_recipe.display_name or self.current_recipe.name
        self.target_recipe_var.set(
            f"Find this recipe: {display_name}"
            f" ({ingredient_total} ingredients)"
        )

    def _build_hand(self, recipe: Recipe) -> List[str]:
        required = list(recipe.trio)
        extras = [name for name in self.data.ingredients if name not in required]
        self.rng.shuffle(extras)
        needed = HAND_SIZE - len(required)
        if needed < 0:
            raise RuntimeError("Recipe requires more ingredients than the hand allows.")
        selected_extras = extras[:needed]
        full_hand = required + selected_extras
        self.rng.shuffle(full_hand)
        return full_hand

    def _deal_hand(self, hand: List[str]) -> None:
        self.current_hand = hand
        self.selected_indices.clear()
        self.cook_button.config(state=tk.DISABLED)
        for idx, button in enumerate(self.card_buttons):
            if idx < len(hand):
                name = hand[idx]
                image = self._ingredient_image(name)
                button.config(
                    image=image,
                    state=tk.NORMAL,
                    relief=tk.RAISED,
                    borderwidth=2,
                    highlightbackground=self.default_highlight,
                    highlightcolor=self.default_highlight,
                    highlightthickness=1,
                )
                button.image = image
                ingredient = self.data.ingredients.get(name)
                display = (
                    ingredient.display_name if isinstance(ingredient, Ingredient) else name
                )
                self.card_tooltips[idx].set_text(display)
            else:
                button.config(
                    image=self.blank_card_image,
                    state=tk.DISABLED,
                    relief=tk.RAISED,
                    borderwidth=2,
                    highlightbackground=self.disabled_highlight,
                    highlightcolor=self.disabled_highlight,
                    highlightthickness=1,
                )
                button.image = self.blank_card_image
                self.card_tooltips[idx].set_text("")
            self._update_card_visual(idx)

    def _reshuffle_current_hand(self, failure_message: Optional[str] = None) -> None:
        if not self.current_recipe:
            return
        new_hand = self._build_hand(self.current_recipe)
        self._deal_hand(new_hand)
        follow_up = (
            f"Fresh ingredients drawn for {self.current_recipe.display_name}. Try again!"
        )
        if failure_message:
            self.status_var.set(f"{failure_message} {follow_up}")
        else:
            self.status_var.set(follow_up)

    def _select_card(self, index: int) -> None:
        if index >= len(self.current_hand):
            return
        button = self.card_buttons[index]
        if str(button["state"]) == tk.DISABLED:
            return
        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.add(index)
        self._update_card_visual(index)
        self._update_status_message()
        self.cook_button.config(state=tk.NORMAL if self.selected_indices else tk.DISABLED)

    def _cook_selected(self) -> None:
        if not self.selected_indices or not self.current_recipe:
            self.status_var.set("Pick at least one ingredient before cooking.")
            return
        selected_names = {self.current_hand[idx] for idx in self.selected_indices}
        self.attempts += 1
        extras = selected_names - self.current_trio
        missing = self.current_trio - selected_names
        if not extras and not missing:
            joined = self._format_display_list(
                [self._display_name(name) for name in sorted(selected_names)]
            )
            self.status_var.set(
                f"Perfect! {joined} complete {self.current_recipe.display_name}."
            )
            self._record_success()
            return

        messages: List[str] = []
        if extras:
            extras_display = self._format_display_list(
                [self._display_name(name) for name in sorted(extras)]
            )
            plural = "s" if len(extras) > 1 else ""
            messages.append(f"Extra ingredient{plural}: {extras_display}.")
        if missing:
            missing_display = self._format_display_list(
                [self._display_name(name) for name in sorted(missing)]
            )
            plural = "s" if len(missing) > 1 else ""
            messages.append(f"Missing ingredient{plural}: {missing_display}.")
        failure_message = " ".join(messages)
        self._reshuffle_current_hand(failure_message)

    def _record_success(self) -> None:
        if not self.current_recipe:
            return
        slot_index = sum(1 for slot in self.recipe_slots if getattr(slot, "found", False))
        if slot_index < len(self.recipe_slots):
            slot = self.recipe_slots[slot_index]
            image = self._recipe_image(self.current_recipe.name)
            slot.config(image=image, cursor="hand2")
            slot.image = image
            setattr(slot, "found", True)
            self.found_recipes[slot_index] = self.current_recipe
            display_name = self.current_recipe.display_name or self.current_recipe.name
            self.recipe_name_vars[slot_index].set(display_name)
        self.selected_indices.clear()
        for idx, button in enumerate(self.card_buttons):
            button.config(
                state=tk.DISABLED,
                relief=tk.RAISED,
                borderwidth=2,
                highlightbackground=self.disabled_highlight,
                highlightcolor=self.disabled_highlight,
                highlightthickness=1,
            )
            self._update_card_visual(idx)
        self.cook_button.config(state=tk.DISABLED)
        if slot_index + 1 >= RECIPES_TO_FIND:
            self._finish_game()
        else:
            self.root.after(900, self._start_next_round)

    def _finish_game(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.status_var.set("All recipes discovered! Great job.")
        self.target_recipe_var.set("All recipes discovered!")
        self.cook_button.config(state=tk.DISABLED)
        self.selected_indices.clear()
        for idx, button in enumerate(self.card_buttons):
            button.config(
                state=tk.DISABLED,
                relief=tk.RAISED,
                borderwidth=2,
                highlightbackground=self.disabled_highlight,
                highlightcolor=self.disabled_highlight,
                highlightthickness=1,
            )
            self._update_card_visual(idx)
        self._show_summary(elapsed)

    def _show_summary(self, elapsed_seconds: float) -> None:
        minutes, seconds = divmod(int(round(elapsed_seconds)), 60)
        summary = tk.Toplevel(self.root)
        summary.title("Session Summary")
        summary.resizable(False, False)
        wrapper = tk.Frame(summary, padx=16, pady=16)
        wrapper.pack()
        tk.Label(
            wrapper,
            text="Recipes Completed!",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=(0, 12))
        tk.Label(wrapper, text=f"Total attempts: {self.attempts}", font=("Segoe UI", 12)).pack()
        tk.Label(wrapper, text=f"Time: {minutes:02d}:{seconds:02d}", font=("Segoe UI", 12)).pack(pady=(0, 12))
        tk.Button(wrapper, text="Close", command=summary.destroy, width=12).pack()

    def _update_card_visual(self, index: int) -> None:
        if index >= len(self.card_buttons):
            return
        button = self.card_buttons[index]
        if str(button["state"]) == tk.DISABLED and index not in self.selected_indices:
            button.config(
                relief=tk.RAISED,
                borderwidth=2,
                highlightbackground=self.disabled_highlight,
                highlightcolor=self.disabled_highlight,
                highlightthickness=1,
            )
            return
        if index in self.selected_indices:
            button.config(
                relief=tk.SUNKEN,
                borderwidth=3,
                highlightbackground=self.selected_highlight,
                highlightcolor=self.selected_highlight,
                highlightthickness=3,
            )
        else:
            button.config(
                relief=tk.RAISED,
                borderwidth=2,
                highlightbackground=self.default_highlight,
                highlightcolor=self.default_highlight,
                highlightthickness=1,
            )

    def _update_status_message(self) -> None:
        if not self.selected_indices:
            if self.current_recipe:
                needed = len(self.current_recipe.trio)
                self.status_var.set(
                    f"Select up to {needed} ingredients, then press Cook!"
                )
            else:
                self.status_var.set("Select ingredients and press Cook!")
            return
        names = [
            self._display_name(self.current_hand[idx]) for idx in sorted(self.selected_indices)
        ]
        formatted = self._format_display_list(names)
        required = len(self.current_recipe.trio) if self.current_recipe else 0
        selected_count = len(names)
        plural = "s" if required != 1 else ""
        prefix = f"Selected {selected_count}/{required} ingredient{plural}: "
        if selected_count == 1:
            self.status_var.set(
                f"{prefix}{formatted}. Choose more or press Cook."
            )
        else:
            self.status_var.set(f"{prefix}{formatted}. Press Cook when ready.")

    def _show_recipe_details(self, index: int) -> None:
        if index >= len(self.found_recipes):
            return
        recipe = self.found_recipes[index]
        if not recipe:
            return

        detail = tk.Toplevel(self.root)
        detail.title(recipe.display_name or recipe.name)
        detail.geometry(f"{DETAIL_WINDOW_WIDTH}x{DETAIL_WINDOW_HEIGHT}")
        detail.resizable(False, False)
        detail.transient(self.root)

        wrapper = tk.Frame(detail, padx=20, pady=20)
        wrapper.pack(fill="both", expand=True)

        tk.Label(
            wrapper,
            text=recipe.display_name or recipe.name,
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=(0, 12))

        tk.Label(
            wrapper,
            text="Key Ingredients",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            justify="left",
            wraplength=DETAIL_WRAP_LENGTH,
        ).pack(anchor="w", fill="x")

        for ingredient_name in recipe.trio:
            ingredient = self.data.ingredients.get(ingredient_name)
            if isinstance(ingredient, Ingredient):
                display_name = ingredient.display_name or ingredient.name
                description = (
                    f"Taste: {ingredient.taste} | Family: {ingredient.family} | Value: {ingredient.Value}"
                )
            else:
                display_name = ingredient_name
                description = "No additional details available."

            entry_frame = tk.Frame(wrapper, pady=4)
            entry_frame.pack(anchor="w", fill="x")
            tk.Label(
                entry_frame,
                text=display_name,
                font=("Segoe UI", 11, "bold"),
                anchor="w",
                justify="left",
            ).pack(anchor="w")
            tk.Label(
                entry_frame,
                text=description,
                font=("Segoe UI", 10),
                anchor="w",
                justify="left",
                wraplength=DETAIL_WRAP_LENGTH,
            ).pack(anchor="w")

        tk.Button(wrapper, text="Close", command=detail.destroy, width=14).pack(
            pady=(16, 0)
        )

    def _display_name(self, ingredient_name: str) -> str:
        ingredient = self.data.ingredients.get(ingredient_name)
        if isinstance(ingredient, Ingredient):
            return ingredient.display_name
        return ingredient_name

    def _format_display_list(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return " and ".join(items)
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def main() -> None:
    root = tk.Tk()
    SecretRecipeGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
