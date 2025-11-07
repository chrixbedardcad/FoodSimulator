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
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk

from food_api import GameData, Ingredient, Recipe

HAND_SIZE = 8
RECIPES_TO_FIND = 3
ASSETS_ROOT = Path(__file__).resolve().parent
INGREDIENT_ICON_DIR = ASSETS_ROOT / "Ingredients"
RECIPE_ICON_DIR = ASSETS_ROOT / "recipes"
PLACEHOLDER_RECIPE_IMAGE = RECIPE_ICON_DIR / "emptydish.png"


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
        self.root.title("Secret Recipe Hunt")
        self.root.resizable(False, False)
        self.data = GameData.from_json()
        self.rng = random.Random()
        self.pending_recipes: List[Recipe] = self._pick_recipes()
        self.current_recipe: Optional[Recipe] = None
        self.current_trio: set[str] = set()
        self.current_hand: List[str] = []
        self.selected_index: Optional[int] = None
        self.attempts = 0
        self.start_time = time.perf_counter()

        self.ingredient_images: Dict[str, tk.PhotoImage] = {}
        self.recipe_images: Dict[str, tk.PhotoImage] = {}
        self.ingredient_image_paths = self._scan_asset_directory(INGREDIENT_ICON_DIR)
        self.recipe_image_paths = self._scan_asset_directory(RECIPE_ICON_DIR)
        self.blank_card_image = self._build_placeholder_image(size=128)
        self.placeholder_recipe_image = self._load_photo(PLACEHOLDER_RECIPE_IMAGE)

        self.status_var = tk.StringVar(value="Select an ingredient and press Cook!")
        self._build_layout()
        self._start_next_round()

    # --- UI assembly helpers -------------------------------------------------
    def _build_layout(self) -> None:
        wrapper = tk.Frame(self.root, padx=12, pady=12)
        wrapper.grid(row=0, column=0)

        header = tk.Label(wrapper, text="Secret Recipe Hunt", font=("Segoe UI", 18, "bold"))
        header.grid(row=0, column=0, columnspan=4, pady=(0, 12))

        summary_frame = tk.LabelFrame(wrapper, text="Recipes Found", padx=8, pady=8)
        summary_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 12))
        self.recipe_slots: List[tk.Label] = []
        for index in range(RECIPES_TO_FIND):
            slot = tk.Label(summary_frame, image=self.placeholder_recipe_image, width=140, height=140)
            slot.image = self.placeholder_recipe_image
            slot.grid(row=0, column=index, padx=6)
            self.recipe_slots.append(slot)

        cards_frame = tk.Frame(wrapper)
        cards_frame.grid(row=2, column=0, columnspan=4)
        self.card_buttons: List[tk.Button] = []
        self.card_tooltips: List[Tooltip] = []
        for idx in range(HAND_SIZE):
            button = tk.Button(
                cards_frame,
                image=self.blank_card_image,
                width=140,
                height=140,
                relief=tk.RAISED,
                borderwidth=2,
                command=lambda index=idx: self._select_card(index),
            )
            button.image = self.blank_card_image
            row, column = divmod(idx, 4)
            button.grid(row=row, column=column, padx=6, pady=6)
            tooltip = Tooltip(button)
            self.card_buttons.append(button)
            self.card_tooltips.append(tooltip)

        controls = tk.Frame(wrapper)
        controls.grid(row=3, column=0, columnspan=4, pady=(12, 0))

        self.cook_button = tk.Button(
            controls,
            text="Cook",
            font=("Segoe UI", 12, "bold"),
            command=self._cook_selected,
            state=tk.DISABLED,
            width=12,
        )
        self.cook_button.grid(row=0, column=0, padx=6)

        self.status_label = tk.Label(controls, textvariable=self.status_var, anchor="w", width=40)
        self.status_label.grid(row=0, column=1, padx=6)

    # --- Asset helpers -------------------------------------------------------
    def _scan_asset_directory(self, directory: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        if not directory.exists():
            return paths
        for image_path in directory.glob("*.png"):
            key = _normalize_key(image_path.stem)
            paths[key] = image_path
        return paths

    def _build_placeholder_image(self, size: int = 120) -> tk.PhotoImage:
        image = tk.PhotoImage(master=self.root, width=size, height=size)
        image.put("#d9d9d9", to=(0, 0, size, size))
        return image

    def _load_photo(self, path: Path) -> tk.PhotoImage:
        if not path or not path.exists():
            return self.blank_card_image
        image = tk.PhotoImage(master=self.root, file=str(path))
        width, height = image.width(), image.height()
        target = 128
        scale = max(width // target, height // target, 1)
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
        self.current_hand = self._build_hand(self.current_recipe)
        self.selected_index = None
        self.cook_button.config(state=tk.DISABLED)
        for idx, name in enumerate(self.current_hand):
            button = self.card_buttons[idx]
            image = self._ingredient_image(name)
            button.config(image=image, state=tk.NORMAL, relief=tk.RAISED)
            button.image = image
            ingredient = self.data.ingredients.get(name)
            display = ingredient.display_name if isinstance(ingredient, Ingredient) else name
            self.card_tooltips[idx].set_text(display)
        self.status_var.set("A new secret recipe awaits. Choose wisely!")

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

    def _select_card(self, index: int) -> None:
        if index >= len(self.current_hand):
            return
        self.selected_index = index
        for idx, button in enumerate(self.card_buttons):
            button.config(relief=tk.SUNKEN if idx == index else tk.RAISED)
        ingredient_name = self.current_hand[index]
        ingredient = self.data.ingredients.get(ingredient_name)
        display = ingredient.display_name if isinstance(ingredient, Ingredient) else ingredient_name
        self.status_var.set(f"Selected {display}. Press Cook to test it.")
        self.cook_button.config(state=tk.NORMAL)

    def _cook_selected(self) -> None:
        if self.selected_index is None or not self.current_recipe:
            self.status_var.set("Pick an ingredient before cooking.")
            return
        choice = self.current_hand[self.selected_index]
        self.attempts += 1
        ingredient = self.data.ingredients.get(choice)
        display = ingredient.display_name if isinstance(ingredient, Ingredient) else choice
        if choice in self.current_trio:
            self.status_var.set(
                f"Correct! {display} belongs to {self.current_recipe.display_name}."
            )
            self._record_success()
        else:
            self.status_var.set(f"{display} is not in this recipe. Try another ingredient.")
            wrong_button = self.card_buttons[self.selected_index]
            wrong_button.config(state=tk.DISABLED, relief=tk.RAISED)
            self.selected_index = None
            self.cook_button.config(state=tk.DISABLED)

    def _record_success(self) -> None:
        if not self.current_recipe:
            return
        slot_index = sum(1 for slot in self.recipe_slots if getattr(slot, "found", False))
        if slot_index < len(self.recipe_slots):
            slot = self.recipe_slots[slot_index]
            image = self._recipe_image(self.current_recipe.name)
            slot.config(image=image)
            slot.image = image
            setattr(slot, "found", True)
        for button in self.card_buttons:
            button.config(state=tk.DISABLED, relief=tk.RAISED)
        self.cook_button.config(state=tk.DISABLED)
        if slot_index + 1 >= RECIPES_TO_FIND:
            self._finish_game()
        else:
            self.root.after(900, self._start_next_round)

    def _finish_game(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.status_var.set("All recipes discovered! Great job.")
        self.cook_button.config(state=tk.DISABLED)
        for button in self.card_buttons:
            button.config(state=tk.DISABLED, relief=tk.RAISED)
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


def main() -> None:
    root = tk.Tk()
    SecretRecipeGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
