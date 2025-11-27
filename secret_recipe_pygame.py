"""Pygame adaptation of the Secret Recipe mini-game.

The original :mod:`secret_recipe_game` provides a Tkinter prototype that lets
players guess the hidden recipe by selecting the correct combination of
ingredients. This module mirrors the rules but renders everything with Pygame
instead of Tkinter so it can run in a more arcade-like environment.
"""
from __future__ import annotations

import math
import random
import time
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pygame

from food_api import GameData, Ingredient, Recipe

HAND_SIZE = 8
RECIPES_TO_FIND = 3
CARD_COLUMNS = 6
CARD_ROWS = 4
MAX_HAND_SIZE = CARD_COLUMNS * CARD_ROWS
CARD_WIDTH = 140
CARD_HEIGHT = 100
CARD_PADDING_X = 18
CARD_PADDING_Y = 18
CARD_LEFT_MARGIN = 60
SUMMARY_PANEL_HEIGHT = 210
SUMMARY_PANEL_HEADER_GAP = 10
SUMMARY_TO_CARDS_GAP = 20
CONTROLS_GAP = 40
CONTROLS_RIGHT_MARGIN = 40
SCREEN_WIDTH = 1360
SCREEN_HEIGHT = 860
BACKGROUND_COLOR = (26, 30, 41)
PANEL_COLOR = (38, 45, 60)
CARD_COLOR = (233, 233, 240)
CARD_SELECTED_COLOR = (129, 199, 132)
CARD_DISABLED_COLOR = (176, 176, 184)
TEXT_COLOR = (245, 245, 250)
TEXT_MUTED_COLOR = (200, 204, 214)
ACCENT_COLOR = (255, 202, 61)
NEXT_ROUND_DELAY_MS = 900
NEXT_ROUND_EVENT = pygame.USEREVENT + 1
SUMMARY_ENTRY_PENDING_COLOR = (52, 59, 74)
SUMMARY_ENTRY_FOUND_COLOR = (62, 86, 70)
SUMMARY_ENTRY_BORDER_COLOR = (94, 106, 128)
RECIPE_SLOT_SIZE = (86, 86)
RECIPE_SLOT_SPACING = 18
RECIPE_SLOT_BACKGROUND = (52, 58, 76)
RECIPE_SLOT_ASSIGNED = (108, 148, 112)

VERSION = "v. 0.02"
CARD_IMAGE_MAX_SIZE = (120, 72)
RECIPE_IMAGE_MAX_SIZE = (96, 96)
INGREDIENT_ICON_DIR = Path(__file__).resolve().parent / "Ingredients"
RECIPE_ICON_DIR = Path(__file__).resolve().parent / "recipes"
BUTTON_ICON_DIR = Path(__file__).resolve().parent / "icons"


class PygameSecretRecipeGame:
    """Run the Secret Recipe Hunt inside a Pygame window."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Secret Recipe Hunt - Pygame Edition")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI", 44, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 24)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)
        self.font_xsmall = pygame.font.SysFont("Segoe UI", 18)

        self.sounds_enabled = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.sounds_enabled = pygame.mixer.get_init() is not None
        except pygame.error:
            self.sounds_enabled = False

        self.data = GameData.from_json()
        self.rng = random.Random()
        self.recipes_per_round = RECIPES_TO_FIND

        self.card_columns = CARD_COLUMNS
        self.card_rows = CARD_ROWS
        self.card_width = CARD_WIDTH
        self.card_height = CARD_HEIGHT
        self.card_padding_x = CARD_PADDING_X
        self.card_padding_y = CARD_PADDING_Y
        self.card_left_margin = CARD_LEFT_MARGIN
        self.summary_panel_height = SUMMARY_PANEL_HEIGHT
        self.summary_panel_header_gap = SUMMARY_PANEL_HEADER_GAP
        self.summary_to_cards_gap = SUMMARY_TO_CARDS_GAP
        self.controls_gap = CONTROLS_GAP
        self.controls_right_margin = CONTROLS_RIGHT_MARGIN
        self.card_grid_width = (
            self.card_columns * self.card_width
            + (self.card_columns - 1) * self.card_padding_x
        )
        self.card_grid_max_height = (
            self.card_rows * self.card_height
            + (self.card_rows - 1) * self.card_padding_y
        )

        # Initialized in _reset_game_state
        self.round = 1
        self.total_points = 0
        self.round_summaries: List[Dict[str, Any]] = []
        self.cookbook_records: Dict[str, Dict[str, Any]] = {}
        self.cookbook_visible = False
        self.pending_recipes: List[Recipe] = []
        self.current_recipe: Optional[Recipe] = None
        self.current_trio: Set[str] = set()
        self.current_hand: List[str] = []
        self.hand_size = HAND_SIZE
        self.max_hand_size = MAX_HAND_SIZE
        self.recipe_slots: List[Optional[int]] = []
        self.card_to_slot: Dict[int, int] = {}
        self.summary_rect = pygame.Rect(0, 0, 0, 0)
        self.hand_active = False
        self.selected_indices: Set[int] = set()
        self.attempts = 0
        self.start_time = time.perf_counter()
        self.finish_time: Optional[float] = None
        self.status_message = ""
        self.target_message = ""
        self.waiting_for_next_round = False
        self.found_recipes: List[Optional[Recipe]] = []
        self.card_rects: List[pygame.Rect] = []
        self.controls_rect = pygame.Rect(0, 0, 0, 0)
        self.cook_button = pygame.Rect(0, 0, 0, 0)

        self._ingredient_display: Dict[str, str] = {
            name: (ingredient.display_name if isinstance(ingredient, Ingredient) else name)
            for name, ingredient in self.data.ingredients.items()
        }

        self._ingredient_image_paths = self._scan_asset_directory(INGREDIENT_ICON_DIR)
        self._recipe_image_paths = self._scan_asset_directory(RECIPE_ICON_DIR)
        self._ingredient_image_cache: Dict[str, pygame.Surface] = {}
        self._recipe_image_cache: Dict[str, pygame.Surface] = {}
        self.ingredient_placeholder = self._build_placeholder_surface(
            CARD_IMAGE_MAX_SIZE, border_radius=22
        )
        self.recipe_placeholder = self._build_placeholder_surface(
            RECIPE_IMAGE_MAX_SIZE, border_radius=18
        )

        self.click_sound = self._build_tone(880.0, 90, volume=0.3) if self.sounds_enabled else None
        self.cook_sound = self._build_tone(420.0, 260, volume=0.4) if self.sounds_enabled else None
        self.success_sound = (
            self._build_tone(1020.0, 220, volume=0.4) if self.sounds_enabled else None
        )
        self.error_sound = (
            self._build_tone(220.0, 180, volume=0.35) if self.sounds_enabled else None
        )

        (
            self.cook_icon_enabled,
            self.cook_icon_disabled,
        ) = self._load_button_icon_pair("cook.png", (44, 44))
        (
            self.cookbook_icon_enabled,
            self.cookbook_icon_disabled,
        ) = self._load_button_icon_pair("cookbook.png", (40, 40))

        self.cookbook_button_rect: Optional[pygame.Rect] = None

        self._reset_game_state()

    def _reset_game_state(self) -> None:
        self.round = 1
        self.total_points = 0
        self.round_summaries.clear()
        self.cookbook_records.clear()
        self.cookbook_visible = False
        self.pending_recipes = []
        self.current_recipe = None
        self.current_trio = set()
        self.current_hand = []
        self.hand_size = HAND_SIZE
        self.recipe_slots = []
        self.card_to_slot = {}
        self.hand_active = False
        self.selected_indices = set()
        self.attempts = 0
        self.start_time = time.perf_counter()
        self.finish_time = None
        self.status_message = "Select ingredients and press Cook!"
        self.target_message = "Find this recipe: ???"
        self.waiting_for_next_round = False
        self.found_recipes = [None] * self.recipes_per_round
        self.active_card_rows = 0

        self._recompute_layout()
        self._prepare_new_round()
        self._start_next_round()

    # --- Game flow -----------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.finish_time is not None:
                        self._reset_game_state()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == NEXT_ROUND_EVENT:
                    self.waiting_for_next_round = False
                    self._start_next_round()

            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _prepare_new_round(self) -> None:
        self.pending_recipes = self._pick_recipes()
        self.found_recipes = [None] * self.recipes_per_round
        self.current_recipe = None
        self.current_trio.clear()
        self.current_hand = []
        self.hand_active = False
        self._set_recipe_slot_count(0)

    def _pick_recipes(self) -> List[Recipe]:
        pool = [recipe for recipe in self.data.recipes if len(recipe.trio) <= self.hand_size]
        if len(pool) < self.recipes_per_round:
            raise RuntimeError("Not enough recipes to start the game.")
        return self.rng.sample(pool, self.recipes_per_round)

    def _start_next_round(self) -> None:
        if not self.pending_recipes:
            self._complete_round()
            if not self.pending_recipes or self.waiting_for_next_round:
                return
        self.current_recipe = self.pending_recipes.pop(0)
        self.current_trio = set(self.current_recipe.trio)
        self._set_recipe_slot_count(len(self.current_trio))
        self._deal_hand(self._build_hand(self.current_recipe))
        display_name = self.current_recipe.display_name or self.current_recipe.name
        self.target_message = f"Find this recipe: {display_name}"
        self.status_message = (
            "A new secret recipe awaits. Fill each recipe slot with the correct ingredients."
        )

    def _finish_game(self) -> None:
        if self.finish_time is None:
            self.finish_time = time.perf_counter()
        self.hand_active = False
        self.status_message = "All recipes discovered! Great job."
        self.target_message = "All recipes discovered!"
        self._set_recipe_slot_count(0)

    def _recipe_points(self, recipe: Recipe) -> int:
        total = 0
        for name in recipe.trio:
            ingredient = self.data.ingredients.get(name)
            if ingredient is None:
                continue
            total += getattr(ingredient, "Value", 0)
        return total

    def _complete_round(self) -> None:
        completed_recipes = [recipe for recipe in self.found_recipes if recipe]
        if not completed_recipes:
            self._prepare_new_round()
            return

        entries: List[Dict[str, Any]] = []
        round_points = 0
        for recipe in completed_recipes:
            ingredient_names = [self._display_name(name) for name in recipe.trio]
            points = self._recipe_points(recipe)
            round_points += points
            entries.append(
                {
                    "recipe": recipe,
                    "ingredients": ingredient_names,
                    "points": points,
                }
            )

        summary = {
            "round": self.round,
            "recipes": entries,
            "total_points": round_points,
        }
        self.round_summaries.append(summary)
        self.total_points += round_points

        self.status_message = (
            f"Round {self.round} complete! {round_points} points added to the cookbook."
        )
        self.target_message = (
            f"Cookbook updated. Prepare for round {self.round + 1}!"
        )

        self.cookbook_visible = True
        self.current_recipe = None
        self.current_trio.clear()
        self.current_hand = []
        self.hand_active = False
        self.selected_indices.clear()
        self._set_recipe_slot_count(0)

        self.round += 1
        self.hand_size = min(self.hand_size + 1, self.max_hand_size)
        self._recompute_layout()
        self._prepare_new_round()
        self.waiting_for_next_round = True
        pygame.time.set_timer(NEXT_ROUND_EVENT, NEXT_ROUND_DELAY_MS, loops=1)

    # --- Layout helpers -----------------------------------------------------
    def _header_bottom(self) -> int:
        header_y = 40
        text_height = self.font_title.get_linesize()
        text_bottom = header_y + text_height

        recipe_surface = self._recipe_image_surface(self.current_recipe)
        image_height = recipe_surface.get_height()
        image_top = header_y + (text_height - image_height) // 2
        image_top = max(20, image_top)
        image_bottom = image_top + image_height

        slot_bottom = text_bottom
        slot_count = len(self.recipe_slots)
        if slot_count:
            slot_height = RECIPE_SLOT_SIZE[1]
            slot_y = max(image_top, text_bottom + 12)
            slot_bottom = slot_y + slot_height

        return max(text_bottom, image_bottom, slot_bottom)

    def _summary_panel_rect(self) -> pygame.Rect:
        header_bottom = self._header_bottom()
        top = header_bottom + self.summary_panel_header_gap
        return pygame.Rect(
            self.card_left_margin,
            top,
            self.card_grid_width,
            self.summary_panel_height,
        )

    def _build_controls_rect(self) -> pygame.Rect:
        controls_left = self.card_left_margin + self.card_grid_width + self.controls_gap
        controls_top = self.summary_rect.top
        available_width = SCREEN_WIDTH - controls_left - self.controls_right_margin
        controls_width = max(220, available_width)
        max_height = (
            self.summary_panel_height
            + self.summary_to_cards_gap
            + self.card_grid_max_height
        )
        available_height = SCREEN_HEIGHT - controls_top - 40
        controls_height = min(max_height, available_height)
        return pygame.Rect(controls_left, controls_top, controls_width, controls_height)

    def _position_cook_button(self) -> None:
        if self.controls_rect.width <= 0 or self.controls_rect.height <= 0:
            self.cook_button = pygame.Rect(0, 0, 0, 0)
            return
        button_width = max(160, self.controls_rect.width - 40)
        button_height = 56
        button_x = self.controls_rect.x + (self.controls_rect.width - button_width) // 2
        button_y = self.controls_rect.y + 24
        self.cook_button = pygame.Rect(button_x, button_y, button_width, button_height)

    def _recompute_layout(self) -> None:
        self.summary_rect = self._summary_panel_rect()
        self.card_rects = self._build_card_layout(self.hand_size)
        self.controls_rect = self._build_controls_rect()
        self._position_cook_button()

    def _set_recipe_slot_count(self, count: int) -> None:
        self.recipe_slots = [None] * count
        self.card_to_slot.clear()
        self.selected_indices.clear()
        self._recompute_layout()

    def _clear_recipe_slots(self) -> None:
        if not self.recipe_slots:
            self.card_to_slot.clear()
            self.selected_indices.clear()
            return
        for index in range(len(self.recipe_slots)):
            self.recipe_slots[index] = None
        self.card_to_slot.clear()
        self.selected_indices.clear()

    def _build_hand(self, recipe: Recipe) -> List[str]:
        required = list(recipe.trio)
        extras = [name for name in self.data.ingredients if name not in required]
        self.rng.shuffle(extras)
        needed = self.hand_size - len(required)
        if needed < 0:
            raise RuntimeError("Recipe requires more ingredients than the hand allows.")
        selected_extras = extras[:needed]
        full_hand = required + selected_extras
        self.rng.shuffle(full_hand)
        return full_hand

    def _deal_hand(self, hand: Sequence[str]) -> None:
        self.current_hand = list(hand)
        if len(self.current_hand) > len(self.card_rects):
            self.hand_size = min(len(self.current_hand), self.max_hand_size)
            self._recompute_layout()
        self.hand_active = True
        self._clear_recipe_slots()

    def _reshuffle_current_hand(self, failure_message: Optional[str] = None) -> None:
        if not self.current_recipe:
            return
        self._deal_hand(self._build_hand(self.current_recipe))
        follow_up = (
            f"Fresh ingredients drawn for {self.current_recipe.display_name}. Try again!"
        )
        if failure_message:
            self.status_message = f"{failure_message} {follow_up}"
        else:
            self.status_message = follow_up

    def _update_cookbook_records(self, recipe: Recipe) -> None:
        key = self._normalize_key(recipe.name or recipe.display_name or "")
        if not key:
            return

        ingredients = list(recipe.trio)
        record = self.cookbook_records.get(key)
        if record is None:
            record = {
                "recipe": recipe,
                "ingredients": ingredients,
                "count": 0,
                "points": self._recipe_points(recipe),
            }
            self.cookbook_records[key] = record
        else:
            record["recipe"] = recipe
            record["ingredients"] = ingredients
            record.setdefault("points", self._recipe_points(recipe))

        record["count"] += 1

    def _record_success(self) -> None:
        if not self.current_recipe:
            return
        slot_index = self.found_recipes.index(None)
        self.found_recipes[slot_index] = self.current_recipe
        joined = self._format_display_list(
            [self._display_name(name) for name in sorted(self.current_trio)]
        )
        self.status_message = (
            f"Perfect! {joined} complete {self.current_recipe.display_name}."
        )
        self.hand_active = False
        self.selected_indices.clear()
        self.current_recipe = None
        self._update_cookbook_records(self.found_recipes[slot_index])
        self._set_recipe_slot_count(0)
        if all(recipe is not None for recipe in self.found_recipes):
            self.waiting_for_next_round = True
            pygame.time.set_timer(NEXT_ROUND_EVENT, NEXT_ROUND_DELAY_MS, loops=1)
            return

        # Queue up the next recipe with a short pause so the success message
        # has time to display before the new hand appears.
        self.waiting_for_next_round = True
        pygame.time.set_timer(NEXT_ROUND_EVENT, NEXT_ROUND_DELAY_MS, loops=1)

    def _cook_selected(self) -> None:
        if not self.current_recipe:
            return
        if not self.selected_indices:
            self.status_message = "Pick at least one ingredient before cooking."
            return

        selected_names = {self.current_hand[idx] for idx in self.selected_indices}
        self.attempts += 1
        extras = selected_names - self.current_trio
        missing = self.current_trio - selected_names

        if not extras and not missing:
            self._play_sound(self.success_sound)
            self._record_success()
            return

        self._play_sound(self.error_sound)
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

    # --- Event helpers -------------------------------------------------------
    def _handle_click(self, position: Tuple[int, int]) -> None:
        if (
            self.cookbook_button_rect
            and self.cookbook_button_rect.collidepoint(position)
            and self.cookbook_records
        ):
            self.cookbook_visible = not self.cookbook_visible
            return
        if self.cookbook_visible:
            self.cookbook_visible = False
            return
        if self.finish_time is not None:
            # Game already completed; allow closing only.
            return
        if self.waiting_for_next_round:
            return
        if self.cook_button.collidepoint(position) and self.hand_active:
            self._play_sound(self.cook_sound)
            self._cook_selected()
            return
        if not self.hand_active:
            return
        for index, rect in enumerate(self.card_rects):
            if rect.collidepoint(position) and index < len(self.current_hand):
                if index in self.card_to_slot:
                    slot_index = self.card_to_slot.pop(index)
                    if 0 <= slot_index < len(self.recipe_slots):
                        self.recipe_slots[slot_index] = None
                    self.selected_indices.discard(index)
                    self._play_sound(self.click_sound)
                    self._update_status_message()
                    break

                free_slot = next(
                    (slot for slot, value in enumerate(self.recipe_slots) if value is None),
                    None,
                )
                if free_slot is None:
                    self.status_message = (
                        "All recipe slots are filled. Remove one to try a different ingredient."
                    )
                    break

                self.recipe_slots[free_slot] = index
                self.card_to_slot[index] = free_slot
                self.selected_indices.add(index)
                self._play_sound(self.click_sound)
                self._update_status_message()
                break

    def _update_status_message(self) -> None:
        if not self.current_recipe:
            return
        required = len(self.recipe_slots)
        filled_slots = [index for index in self.recipe_slots if index is not None]
        filled_count = len(filled_slots)
        if filled_count == 0:
            plural = "s" if required != 1 else ""
            self.status_message = (
                f"Select ingredients to fill the {required} recipe slot{plural}, then press Cook!"
            )
            return

        names = [
            self._display_name(self.current_hand[idx])
            for idx in filled_slots
            if 0 <= idx < len(self.current_hand)
        ]
        formatted = self._format_display_list(names)
        plural = "s" if required != 1 else ""
        if filled_count < required:
            self.status_message = (
                f"Filled {filled_count}/{required} recipe slot{plural}: {formatted}."
                " Keep selecting or press Cook to test this combination."
            )
        else:
            self.status_message = (
                f"All recipe slots filled: {formatted}. Press Cook when ready."
            )

    # --- Asset helpers -------------------------------------------------------
    @staticmethod
    def _normalize_key(name: str) -> str:
        return "".join(char.lower() for char in name if char.isalnum())

    def _scan_asset_directory(self, directory: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        if not directory.exists():
            return paths
        for image_path in directory.glob("*.png"):
            key = self._normalize_key(image_path.stem)
            paths[key] = image_path
        return paths

    def _build_placeholder_surface(
        self,
        size: Tuple[int, int],
        *,
        border_radius: int = 18,
        symbol: str = "?",
    ) -> pygame.Surface:
        width, height = size
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(surface, (55, 62, 80), surface.get_rect(), border_radius=border_radius)
        inner_rect = surface.get_rect().inflate(-8, -8)
        inner_radius = max(border_radius - 6, 0)
        pygame.draw.rect(surface, (74, 82, 104), inner_rect, border_radius=inner_radius)
        font_size = max(20, height // 2)
        placeholder_font = pygame.font.SysFont("Segoe UI", font_size, bold=True)
        glyph = placeholder_font.render(symbol, True, ACCENT_COLOR)
        surface.blit(glyph, glyph.get_rect(center=(width // 2, height // 2)))
        return surface.convert_alpha()

    def _load_scaled_image(
        self,
        path: Path,
        max_size: Tuple[int, int],
        fallback: pygame.Surface,
    ) -> pygame.Surface:
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
        except pygame.error:
            return fallback
        width, height = surface.get_size()
        max_width, max_height = max_size
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            surface = pygame.transform.smoothscale(surface, new_size)
        return surface

    def _build_tone(
        self,
        frequency: float,
        duration_ms: int,
        *,
        volume: float = 0.5,
    ) -> Optional[pygame.mixer.Sound]:
        init_info = pygame.mixer.get_init()
        if not init_info:
            return None
        sample_rate, sample_size, channels = init_info
        if abs(sample_size) != 16:
            return None

        total_samples = max(1, int(sample_rate * (duration_ms / 1000.0)))
        amplitude = (2**15) - 1
        waveform: List[int] = []
        for index in range(total_samples):
            theta = 2.0 * math.pi * frequency * (index / sample_rate)
            sample_value = int(amplitude * math.sin(theta))
            if sample_size > 0:
                sample_value += amplitude
            waveform.append(sample_value)

        mono_samples = array("h", waveform)
        if channels == 2:
            stereo_samples = array("h")
            for value in mono_samples:
                stereo_samples.append(value)
                stereo_samples.append(value)
            buffer = stereo_samples.tobytes()
        else:
            buffer = mono_samples.tobytes()

        try:
            sound = pygame.mixer.Sound(buffer=buffer)
        except pygame.error:
            return None
        sound.set_volume(max(0.0, min(volume, 1.0)))
        return sound

    def _load_button_icon(self, filename: str, max_size: Tuple[int, int]) -> Optional[pygame.Surface]:
        path = BUTTON_ICON_DIR / filename
        if not path.exists():
            return None
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
        except pygame.error:
            return None
        return self._scale_surface(surface, max_size)

    def _load_button_icon_pair(
        self, filename: str, max_size: Tuple[int, int]
    ) -> Tuple[Optional[pygame.Surface], Optional[pygame.Surface]]:
        icon = self._load_button_icon(filename, max_size)
        if icon is None:
            return None, None
        disabled = icon.copy()
        disabled.fill((180, 180, 180, 255), special_flags=pygame.BLEND_RGBA_MULT)
        disabled.set_alpha(150)
        return icon, disabled

    @staticmethod
    def _play_sound(sound: Optional[pygame.mixer.Sound]) -> None:
        if not sound:
            return
        try:
            sound.play()
        except pygame.error:
            pass

    @staticmethod
    def _scale_surface(surface: pygame.Surface, max_size: Tuple[int, int]) -> pygame.Surface:
        width, height = surface.get_size()
        max_width, max_height = max_size
        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 1.0:
            return surface
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return pygame.transform.smoothscale(surface, new_size)

    def _ingredient_image_surface(self, ingredient_name: str) -> pygame.Surface:
        key = self._normalize_key(ingredient_name)
        cached = self._ingredient_image_cache.get(key)
        if cached is not None:
            return cached

        ingredient = self.data.ingredients.get(ingredient_name)
        candidates: List[str] = [ingredient_name]
        if ingredient:
            if ingredient.display_name and ingredient.display_name not in candidates:
                candidates.append(ingredient.display_name)
            if ingredient.ingredient_id and ingredient.ingredient_id not in candidates:
                candidates.append(ingredient.ingredient_id)

        surface: Optional[pygame.Surface] = None
        for candidate in candidates:
            candidate_key = self._normalize_key(candidate)
            path = self._ingredient_image_paths.get(candidate_key)
            if path and path.exists():
                surface = self._load_scaled_image(
                    path, CARD_IMAGE_MAX_SIZE, self.ingredient_placeholder
                )
                break

        result = surface or self.ingredient_placeholder
        for candidate in candidates:
            candidate_key = self._normalize_key(candidate)
            self._ingredient_image_cache.setdefault(candidate_key, result)
        return result

    def _recipe_image_surface(self, recipe: Optional[Recipe]) -> pygame.Surface:
        if recipe is None:
            return self.recipe_placeholder

        candidates = [recipe.name, recipe.display_name]
        surface: Optional[pygame.Surface] = None
        for candidate in candidates:
            candidate_key = self._normalize_key(candidate)
            cached = self._recipe_image_cache.get(candidate_key)
            if cached is not None:
                return cached
            path = self._recipe_image_paths.get(candidate_key)
            if path and path.exists():
                surface = self._load_scaled_image(
                    path, RECIPE_IMAGE_MAX_SIZE, self.recipe_placeholder
                )
                for known in candidates:
                    self._recipe_image_cache[self._normalize_key(known)] = surface
                return surface

        for candidate in candidates:
            candidate_key = self._normalize_key(candidate)
            self._recipe_image_cache.setdefault(candidate_key, self.recipe_placeholder)
        return self.recipe_placeholder

    # --- Drawing helpers -----------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_header()
        self._draw_cards()
        self._draw_controls()
        self._draw_summary()
        self._draw_version()
        if self.cookbook_visible:
            self._draw_cookbook_overlay()
        if self.finish_time is not None:
            self._draw_summary_overlay()

    def _draw_header(self) -> None:
        header_surface = self.font_title.render(self.target_message, True, TEXT_COLOR)
        recipe_surface = self._recipe_image_surface(self.current_recipe)

        header_y = 40
        text_rect = header_surface.get_rect()
        image_rect = recipe_surface.get_rect()

        image_rect.left = 60
        image_rect.top = header_y + (text_rect.height - image_rect.height) // 2
        # Ensure the recipe preview does not overlap the top edge of the screen.
        image_rect.top = max(20, image_rect.top)
        self.screen.blit(recipe_surface, image_rect)

        text_rect.left = image_rect.right + 30
        text_rect.top = header_y
        self.screen.blit(header_surface, text_rect)

        slot_count = len(self.recipe_slots)
        if slot_count:
            slot_width, slot_height = RECIPE_SLOT_SIZE
            slot_spacing = RECIPE_SLOT_SPACING
            total_width = slot_count * slot_width + (slot_count - 1) * slot_spacing
            max_start_x = SCREEN_WIDTH - 60 - total_width
            slot_start_x = max(image_rect.right + 30, min(text_rect.left, max_start_x))
            slot_y = max(image_rect.top, text_rect.bottom + 12)

            for offset, card_index in enumerate(self.recipe_slots):
                slot_x = slot_start_x + offset * (slot_width + slot_spacing)
                slot_rect = pygame.Rect(slot_x, slot_y, slot_width, slot_height)
                has_ingredient = (
                    card_index is not None and 0 <= card_index < len(self.current_hand)
                )
                fill_color = RECIPE_SLOT_ASSIGNED if has_ingredient else RECIPE_SLOT_BACKGROUND
                pygame.draw.rect(self.screen, fill_color, slot_rect, border_radius=12)
                pygame.draw.rect(
                    self.screen,
                    SUMMARY_ENTRY_BORDER_COLOR,
                    slot_rect,
                    width=2,
                    border_radius=12,
                )

                if has_ingredient:
                    ingredient_name = self.current_hand[card_index]
                    tile_surface = self._ingredient_image_surface(ingredient_name)
                else:
                    tile_surface = self.ingredient_placeholder

                tile_surface = self._scale_surface(
                    tile_surface, (slot_rect.width - 18, slot_rect.height - 18)
                )
                tile_rect = tile_surface.get_rect(center=slot_rect.center)
                self.screen.blit(tile_surface, tile_rect)

    def _draw_cards(self) -> None:
        for index, rect in enumerate(self.card_rects):
            if index >= len(self.current_hand):
                placeholder_border = pygame.Rect(rect)
                pygame.draw.rect(
                    self.screen,
                    (52, 58, 76),
                    placeholder_border,
                    width=2,
                    border_radius=12,
                )
                continue

            name = self.current_hand[index]
            display_name = self._display_name(name)
            if not self.hand_active:
                fill = CARD_DISABLED_COLOR
            elif index in self.selected_indices:
                fill = CARD_SELECTED_COLOR
            else:
                fill = CARD_COLOR
            pygame.draw.rect(self.screen, fill, rect, border_radius=12)
            pygame.draw.rect(self.screen, PANEL_COLOR, rect, width=2, border_radius=12)
            image_surface = self._ingredient_image_surface(name)
            image_rect = image_surface.get_rect()
            image_rect.centerx = rect.centerx
            image_rect.top = rect.y + 12
            self.screen.blit(image_surface, image_rect)

            text_lines = self._wrap_text(display_name, self.font_small, rect.width - 24)
            line_height = self.font_small.get_linesize()
            text_height = len(text_lines) * line_height
            text_area_top = image_rect.bottom + 8
            start_y = max(text_area_top, rect.y + 12)
            max_start = rect.bottom - text_height - 12
            start_y = min(start_y, max_start)
            start_y = max(start_y, rect.y + 12)
            for line_index, line in enumerate(text_lines):
                surface = self.font_small.render(line, True, (32, 32, 42))
                line_rect = surface.get_rect(
                    center=(
                        rect.centerx,
                        start_y + line_height * line_index + line_height // 2,
                    )
                )
                self.screen.blit(surface, line_rect)

    def _draw_controls(self) -> None:
        controls_rect = self.controls_rect
        if controls_rect.width <= 0 or controls_rect.height <= 0:
            return

        pygame.draw.rect(self.screen, PANEL_COLOR, controls_rect, border_radius=16)
        pygame.draw.rect(
            self.screen,
            SUMMARY_ENTRY_BORDER_COLOR,
            controls_rect,
            width=2,
            border_radius=16,
        )

        cook_enabled = bool(self.hand_active and self.selected_indices)
        cook_color = ACCENT_COLOR if cook_enabled else CARD_DISABLED_COLOR
        pygame.draw.rect(self.screen, cook_color, self.cook_button, border_radius=10)

        label = self.font_medium.render("Cook", True, (20, 20, 20))
        icon_surface = (
            self.cook_icon_enabled
            if cook_enabled
            else self.cook_icon_disabled or self.cook_icon_enabled
        )
        content_width = label.get_width()
        if icon_surface:
            content_width += icon_surface.get_width() + 12
        start_x = self.cook_button.centerx - content_width // 2
        if icon_surface:
            icon_rect = icon_surface.get_rect()
            icon_rect.centery = self.cook_button.centery
            icon_rect.x = start_x
            self.screen.blit(icon_surface, icon_rect)
            start_x = icon_rect.right + 12
        label_rect = label.get_rect()
        label_rect.centery = self.cook_button.centery
        label_rect.x = start_x
        self.screen.blit(label, label_rect)

        status_top = self.cook_button.bottom + 24
        status_rect = pygame.Rect(
            controls_rect.x + 20,
            status_top,
            controls_rect.width - 40,
            max(0, controls_rect.bottom - status_top - 120),
        )
        if status_rect.height > 0:
            self._blit_wrapped_text(self.status_message, self.font_small, TEXT_COLOR, status_rect)

        stats_x = controls_rect.x + 20
        stats_y = max(status_rect.bottom + 12, controls_rect.bottom - 100)
        slots_surface = self.font_small.render(
            f"Ingredients: {len(self.current_hand)}/{self.max_hand_size}",
            True,
            TEXT_COLOR,
        )
        self.screen.blit(slots_surface, (stats_x, stats_y))
        stats_y += self.font_small.get_linesize() + 4

        attempts_surface = self.font_small.render(f"Attempts: {self.attempts}", True, TEXT_COLOR)
        self.screen.blit(attempts_surface, (stats_x, stats_y))
        stats_y += self.font_small.get_linesize() + 4

        elapsed = (self.finish_time or time.perf_counter()) - self.start_time
        minutes, seconds = divmod(int(elapsed), 60)
        timer_surface = self.font_small.render(f"Time: {minutes:02d}:{seconds:02d}", True, TEXT_COLOR)
        self.screen.blit(timer_surface, (stats_x, stats_y))

    def _draw_summary(self) -> None:
        panel_rect = self.summary_rect
        if panel_rect.width <= 0 or panel_rect.height <= 0:
            return

        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect, border_radius=18)
        pygame.draw.rect(
            self.screen,
            SUMMARY_ENTRY_BORDER_COLOR,
            panel_rect,
            width=2,
            border_radius=18,
        )

        title_surface = self.font_medium.render("Recipes Found", True, TEXT_COLOR)
        self.screen.blit(title_surface, (panel_rect.x + 20, panel_rect.y + 16))

        button_width = 180
        button_height = 48
        button_rect = pygame.Rect(
            panel_rect.right - button_width - 20,
            panel_rect.y + 12,
            button_width,
            button_height,
        )
        self.cookbook_button_rect = button_rect
        has_cookbook = bool(self.cookbook_records)
        button_color = ACCENT_COLOR if has_cookbook else CARD_DISABLED_COLOR
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=12)
        cookbook_label = self.font_small.render("Cookbook", True, (20, 20, 20))
        cookbook_icon = (
            self.cookbook_icon_enabled
            if has_cookbook
            else self.cookbook_icon_disabled or self.cookbook_icon_enabled
        )
        content_width = cookbook_label.get_width()
        if cookbook_icon:
            content_width += cookbook_icon.get_width() + 10
        start_x = button_rect.centerx - content_width // 2
        if cookbook_icon:
            icon_rect = cookbook_icon.get_rect()
            icon_rect.centery = button_rect.centery
            icon_rect.x = start_x
            self.screen.blit(cookbook_icon, icon_rect)
            start_x = icon_rect.right + 10
        label_rect = cookbook_label.get_rect()
        label_rect.centery = button_rect.centery
        label_rect.x = start_x
        self.screen.blit(cookbook_label, label_rect)

        info_y = panel_rect.y + title_surface.get_height() + 24
        points_surface = self.font_small.render(
            f"Total Points: {self.total_points}", True, TEXT_COLOR
        )
        self.screen.blit(points_surface, (panel_rect.x + 20, info_y))

        if self.round > 1:
            round_surface = self.font_xsmall.render(
                f"Round {self.round - 1} complete. Preparing round {self.round}...",
                True,
                TEXT_MUTED_COLOR,
            )
            self.screen.blit(round_surface, (panel_rect.x + 20, info_y + 28))

        slots = len(self.found_recipes)
        if slots == 0:
            return

        entry_area_top = panel_rect.y + 88
        entry_area_height = panel_rect.bottom - entry_area_top - 24
        spacing = 16
        available_width = panel_rect.width - spacing * (slots + 1)
        entry_width = max(140, available_width // slots)
        entry_height = max(80, entry_area_height)
        current_x = panel_rect.x + spacing

        for idx, recipe in enumerate(self.found_recipes):
            entry_rect = pygame.Rect(current_x, entry_area_top, entry_width, entry_height)
            fill_color = SUMMARY_ENTRY_FOUND_COLOR if recipe else SUMMARY_ENTRY_PENDING_COLOR
            pygame.draw.rect(self.screen, fill_color, entry_rect, border_radius=14)
            pygame.draw.rect(
                self.screen,
                SUMMARY_ENTRY_BORDER_COLOR,
                entry_rect,
                width=1,
                border_radius=14,
            )

            icon_surface = self._recipe_image_surface(recipe)
            icon_surface = self._scale_surface(
                icon_surface,
                (
                    min(96, entry_rect.width // 3),
                    min(entry_rect.height - 24, 96),
                ),
            )
            icon_rect = icon_surface.get_rect()
            icon_rect.left = entry_rect.x + 16
            icon_rect.centery = entry_rect.centery
            self.screen.blit(icon_surface, icon_rect)

            text_x = icon_rect.right + 16
            text_width = entry_rect.right - text_x - 16
            if text_width <= 0:
                text_x = icon_rect.right + 8
                text_width = entry_rect.right - text_x - 8

            primary_label = f"{idx + 1}. {recipe.display_name if recipe else '???'}"
            primary_surface = self.font_small.render(primary_label, True, TEXT_COLOR)
            primary_rect = primary_surface.get_rect()
            primary_rect.topleft = (text_x, entry_rect.y + 12)
            if primary_rect.width > text_width:
                primary_surface = self.font_xsmall.render(primary_label, True, TEXT_COLOR)
                primary_rect = primary_surface.get_rect()
                primary_rect.topleft = (text_x, entry_rect.y + 12)
            self.screen.blit(primary_surface, primary_rect)

            if recipe:
                ingredients = [self._display_name(name) for name in recipe.trio]
                detail_text = ", ".join(ingredients)
            else:
                detail_text = "Find the right trio to reveal this recipe."

            detail_lines = self._wrap_text(detail_text, self.font_xsmall, text_width)
            detail_y = primary_rect.bottom + 8
            for line in detail_lines:
                detail_surface = self.font_xsmall.render(line, True, TEXT_MUTED_COLOR)
                self.screen.blit(detail_surface, (text_x, detail_y))
                detail_y += self.font_xsmall.get_linesize()

            current_x += entry_width + spacing

    def _draw_summary_overlay(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 8, 12, 210))
        self.screen.blit(overlay, (0, 0))

        panel = pygame.Rect(0, 0, 520, 280)
        panel.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=18)
        pygame.draw.rect(self.screen, ACCENT_COLOR, panel, width=3, border_radius=18)

        title = self.font_large.render("Session Summary", True, TEXT_COLOR)
        self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 60)))

        attempts = self.font_medium.render(f"Total attempts: {self.attempts}", True, TEXT_COLOR)
        self.screen.blit(attempts, attempts.get_rect(center=(panel.centerx, panel.y + 120)))

        if self.finish_time is not None:
            elapsed = self.finish_time - self.start_time
            minutes, seconds = divmod(int(round(elapsed)), 60)
            timer = self.font_medium.render(f"Time: {minutes:02d}:{seconds:02d}", True, TEXT_COLOR)
            self.screen.blit(timer, timer.get_rect(center=(panel.centerx, panel.y + 160)))

        prompt = self.font_small.render("Press R to Restart • ESC to Exit", True, TEXT_COLOR)
        self.screen.blit(prompt, prompt.get_rect(center=(panel.centerx, panel.y + 210)))

    def _draw_version(self) -> None:
        version_surface = self.font_xsmall.render(VERSION, True, TEXT_MUTED_COLOR)
        version_rect = version_surface.get_rect()
        version_rect.bottomright = (SCREEN_WIDTH - 24, SCREEN_HEIGHT - 20)
        self.screen.blit(version_surface, version_rect)

    def _draw_cookbook_overlay(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 12, 20, 220))
        self.screen.blit(overlay, (0, 0))

        panel_width = 860
        panel_height = 560
        panel = pygame.Rect(0, 0, panel_width, panel_height)
        panel.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=18)
        pygame.draw.rect(self.screen, ACCENT_COLOR, panel, width=3, border_radius=18)

        title = self.font_large.render("Cookbook", True, TEXT_COLOR)
        self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 60)))

        if not self.cookbook_records:
            empty = self.font_medium.render("Find recipes to populate the cookbook!", True, TEXT_COLOR)
            self.screen.blit(empty, empty.get_rect(center=panel.center))
            return

        content_rect = pygame.Rect(
            panel.x + 36,
            panel.y + 110,
            panel.width - 72,
            panel.height - 150,
        )

        line_y = content_rect.y
        ingredient_tile_size = (48, 48)
        icon_spacing = 12
        content_height_limit = content_rect.bottom
        sorted_records = sorted(
            self.cookbook_records.values(),
            key=lambda record: (record["recipe"].display_name or record["recipe"].name),
        )

        for entry in sorted_records:
            recipe: Recipe = entry["recipe"]
            recipe_name = recipe.display_name or recipe.name
            recipe_icon = self._scale_surface(
                self._recipe_image_surface(recipe),
                (min(96, ingredient_tile_size[0] * 2), min(96, ingredient_tile_size[1] * 2)),
            )
            recipe_icon_width, recipe_icon_height = recipe_icon.get_size()
            text_area_width = content_rect.width - recipe_icon_width - 56
            ingredient_names = [self._display_name(name) for name in entry["ingredients"]]
            ingredient_text = ", ".join(ingredient_names)
            text_width_estimate = max(40, text_area_width)
            ingredient_lines = self._wrap_text(
                ingredient_text, self.font_xsmall, max(60, text_width_estimate)
            )
            if ingredient_names:
                available_width = max(ingredient_tile_size[0], text_width_estimate)
                icons_per_row = max(
                    1,
                    (available_width + icon_spacing)
                    // (ingredient_tile_size[0] + icon_spacing),
                )
                icon_rows = math.ceil(len(ingredient_names) / icons_per_row)
                icons_height = (
                    icon_rows * ingredient_tile_size[1]
                    + max(0, icon_rows - 1) * icon_spacing
                )
            else:
                icons_height = 0
            detail_text_height = len(ingredient_lines) * self.font_xsmall.get_linesize()
            block_height = max(
                recipe_icon_height + 32,
                16
                + self.font_small.get_linesize()
                + self.font_xsmall.get_linesize()
                + (12 if icons_height else 0)
                + icons_height
                + (8 if ingredient_lines else 0)
                + detail_text_height
                + 16,
            )
            block_rect = pygame.Rect(content_rect.x, line_y, content_rect.width, block_height)
            if block_rect.bottom > content_height_limit:
                line_y = content_height_limit
                break

            pygame.draw.rect(
                self.screen,
                (58, 66, 84),
                block_rect,
                border_radius=14,
            )
            pygame.draw.rect(
                self.screen,
                SUMMARY_ENTRY_BORDER_COLOR,
                block_rect,
                width=1,
                border_radius=14,
            )

            icon_rect = recipe_icon.get_rect()
            icon_rect.left = block_rect.x + 16
            icon_rect.centery = block_rect.centery
            self.screen.blit(recipe_icon, icon_rect)

            text_x = icon_rect.right + 20
            text_width = max(40, block_rect.right - text_x - 20)
            header_surface = self.font_small.render(
                f"• {recipe_name} (+{entry['points']} pts)", True, TEXT_COLOR
            )
            header_rect = header_surface.get_rect()
            header_rect.topleft = (text_x, block_rect.y + 16)
            if header_rect.width > text_width:
                header_surface = self.font_xsmall.render(
                    f"• {recipe_name} (+{entry['points']} pts)", True, TEXT_COLOR
                )
                header_rect = header_surface.get_rect()
                header_rect.topleft = (text_x, block_rect.y + 16)
            self.screen.blit(header_surface, header_rect)

            count_surface = self.font_xsmall.render(
                f"Cooked {entry['count']} time{'s' if entry['count'] != 1 else ''}",
                True,
                TEXT_MUTED_COLOR,
            )
            count_rect = count_surface.get_rect()
            count_rect.topleft = (text_x, header_rect.bottom + 4)
            self.screen.blit(count_surface, count_rect)

            icon_start_y = count_rect.bottom + (12 if icons_height else 0)
            icon_y = icon_start_y
            icon_x = text_x
            items_in_row = 0
            for ingredient_name in ingredient_names:
                ingredient_surface = self._scale_surface(
                    self._ingredient_image_surface(ingredient_name),
                    ingredient_tile_size,
                )
                if (
                    items_in_row
                    and icon_x + ingredient_tile_size[0] > text_x + text_width
                ):
                    icon_x = text_x
                    icon_y += ingredient_tile_size[1] + icon_spacing
                    items_in_row = 0
                tile_rect = pygame.Rect(
                    icon_x,
                    icon_y,
                    ingredient_tile_size[0],
                    ingredient_tile_size[1],
                )
                pygame.draw.rect(
                    self.screen,
                    CARD_COLOR,
                    tile_rect,
                    border_radius=10,
                )
                image_rect = ingredient_surface.get_rect(center=tile_rect.center)
                self.screen.blit(ingredient_surface, image_rect)
                icon_x += ingredient_tile_size[0] + icon_spacing
                items_in_row += 1

            detail_y = icon_start_y + (icons_height + 8 if icons_height else 0)
            for line in ingredient_lines:
                line_surface = self.font_xsmall.render(line, True, TEXT_MUTED_COLOR)
                self.screen.blit(line_surface, (text_x, detail_y))
                detail_y += self.font_xsmall.get_linesize()

            line_y += block_height + 16

            if line_y >= content_height_limit:
                break

        line_y += 8

        footer = self.font_small.render(
            "Click anywhere to close the cookbook", True, TEXT_COLOR
        )
        self.screen.blit(footer, footer.get_rect(center=(panel.centerx, panel.bottom - 40)))

    # --- Utility helpers -----------------------------------------------------
    def _build_card_layout(self, hand_size: int) -> List[pygame.Rect]:
        max_tiles = min(hand_size, self.max_hand_size)
        start_y = self.summary_rect.bottom + self.summary_to_cards_gap
        rects: List[pygame.Rect] = []
        total_slots = self.max_hand_size

        for index in range(total_slots):
            row = index // self.card_columns
            column = index % self.card_columns
            x = self.card_left_margin + column * (self.card_width + self.card_padding_x)
            y = start_y + row * (self.card_height + self.card_padding_y)
            rects.append(pygame.Rect(x, y, self.card_width, self.card_height))

        if max_tiles <= 0:
            self.active_card_rows = 0
        else:
            self.active_card_rows = max(1, math.ceil(max_tiles / self.card_columns))
        return rects

    def _display_name(self, ingredient_name: str) -> str:
        return self._ingredient_display.get(ingredient_name, ingredient_name)

    def _format_display_list(self, items: Sequence[str]) -> str:
        items = list(items)
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return " and ".join(items)
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines if lines else [""]

    def _blit_wrapped_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        rect: pygame.Rect,
    ) -> None:
        lines = self._wrap_text(text, font, rect.width)
        line_height = font.get_linesize()
        y = rect.y
        for line in lines:
            surface = font.render(line, True, color)
            self.screen.blit(surface, (rect.x, y))
            y += line_height
            if y > rect.bottom:
                break


def main() -> None:
    game = PygameSecretRecipeGame()
    game.run()


if __name__ == "__main__":
    main()
