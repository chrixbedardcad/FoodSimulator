"""Pygame adaptation of the Secret Recipe mini-game.

The original :mod:`secret_recipe_game` provides a Tkinter prototype that lets
players guess the hidden recipe by selecting the correct combination of
ingredients. This module mirrors the rules but renders everything with Pygame
instead of Tkinter so it can run in a more arcade-like environment.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pygame

from food_api import GameData, Ingredient, Recipe

HAND_SIZE = 8
RECIPES_TO_FIND = 3
CARD_COLUMNS = 4
CARD_ROWS = HAND_SIZE // CARD_COLUMNS
SCREEN_WIDTH = 1120
SCREEN_HEIGHT = 760
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

VERSION = "v. 0.02"
CARD_IMAGE_MAX_SIZE = (120, 72)
RECIPE_IMAGE_MAX_SIZE = (96, 96)
INGREDIENT_ICON_DIR = Path(__file__).resolve().parent / "Ingredients"
RECIPE_ICON_DIR = Path(__file__).resolve().parent / "recipes"


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

        self.data = GameData.from_json()
        self.rng = random.Random()
        self.pending_recipes: List[Recipe] = self._pick_recipes()
        self.current_recipe: Optional[Recipe] = None
        self.current_trio: Set[str] = set()
        self.current_hand: List[str] = []
        self.hand_active = False
        self.selected_indices: Set[int] = set()
        self.attempts = 0
        self.start_time = time.perf_counter()
        self.finish_time: Optional[float] = None
        self.status_message = "Select ingredients and press Cook!"
        self.target_message = "Find this recipe: ???"
        self.waiting_for_next_round = False

        self.found_recipes: List[Optional[Recipe]] = [None] * RECIPES_TO_FIND
        self.card_rects = self._build_card_layout()
        self.cook_button = pygame.Rect(80, 660, 160, 48)

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

        self._start_next_round()

    # --- Game flow -----------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == NEXT_ROUND_EVENT:
                    self.waiting_for_next_round = False
                    self._start_next_round()

            self._draw()
            pygame.display.flip()

        pygame.quit()

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
        display_name = self.current_recipe.display_name or self.current_recipe.name
        self.target_message = (
            f"Find this recipe: {display_name}"
            f" ({ingredient_total} ingredients)"
        )
        self.status_message = (
            f"A new secret recipe awaits. Gather the {ingredient_total} key ingredients."
        )

    def _finish_game(self) -> None:
        if self.finish_time is None:
            self.finish_time = time.perf_counter()
        self.hand_active = False
        self.status_message = "All recipes discovered! Great job."
        self.target_message = "All recipes discovered!"

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

    def _deal_hand(self, hand: Sequence[str]) -> None:
        self.current_hand = list(hand)
        self.hand_active = True
        self.selected_indices.clear()

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
        if all(recipe is not None for recipe in self.found_recipes):
            self._finish_game()
        else:
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

    # --- Event helpers -------------------------------------------------------
    def _handle_click(self, position: Tuple[int, int]) -> None:
        if self.finish_time is not None:
            # Game already completed; allow closing only.
            return
        if self.waiting_for_next_round:
            return
        if self.cook_button.collidepoint(position) and self.hand_active:
            self._cook_selected()
            return
        if not self.hand_active:
            return
        for index, rect in enumerate(self.card_rects):
            if rect.collidepoint(position) and index < len(self.current_hand):
                if index in self.selected_indices:
                    self.selected_indices.remove(index)
                else:
                    self.selected_indices.add(index)
                self._update_status_message()
                break

    def _update_status_message(self) -> None:
        if not self.current_recipe:
            return
        required = len(self.current_recipe.trio)
        if not self.selected_indices:
            plural = "s" if required != 1 else ""
            self.status_message = (
                f"Select up to {required} ingredient{plural}, then press Cook!"
            )
            return
        names = [self._display_name(self.current_hand[idx]) for idx in sorted(self.selected_indices)]
        formatted = self._format_display_list(names)
        selected_count = len(self.selected_indices)
        plural = "s" if required != 1 else ""
        self.status_message = (
            f"Selected {selected_count}/{required} ingredient{plural}: {formatted}."
            " Press Cook when ready."
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
        if self.finish_time is not None:
            self._draw_summary_overlay()

    def _draw_header(self) -> None:
        header_surface = self.font_title.render(self.target_message, True, TEXT_COLOR)
        self.screen.blit(header_surface, (60, 40))

    def _draw_cards(self) -> None:
        for index, rect in enumerate(self.card_rects):
            if index >= len(self.current_hand):
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
        controls_rect = pygame.Rect(60, 640, 560, 120)
        pygame.draw.rect(self.screen, PANEL_COLOR, controls_rect, border_radius=16)
        cook_color = ACCENT_COLOR if self.hand_active and self.selected_indices else CARD_DISABLED_COLOR
        pygame.draw.rect(self.screen, cook_color, self.cook_button, border_radius=10)
        label = self.font_medium.render("Cook", True, (20, 20, 20))
        self.screen.blit(label, label.get_rect(center=self.cook_button.center))

        status_rect = pygame.Rect(260, controls_rect.y + 20, 340, controls_rect.height - 40)
        self._blit_wrapped_text(self.status_message, self.font_small, TEXT_COLOR, status_rect)

        attempts_surface = self.font_small.render(f"Attempts: {self.attempts}", True, TEXT_COLOR)
        self.screen.blit(attempts_surface, (80, controls_rect.bottom - 44))
        elapsed = (self.finish_time or time.perf_counter()) - self.start_time
        minutes, seconds = divmod(int(elapsed), 60)
        timer_surface = self.font_small.render(f"Time: {minutes:02d}:{seconds:02d}", True, TEXT_COLOR)
        self.screen.blit(timer_surface, (80, controls_rect.bottom - 24))

    def _draw_summary(self) -> None:
        if not self.card_rects:
            return

        panel_margin = 40
        panel_height = 200
        panel_top = max(self.card_rects[0].y - panel_height - 30, 80)
        panel_rect = pygame.Rect(
            panel_margin,
            panel_top,
            SCREEN_WIDTH - panel_margin * 2,
            panel_height,
        )

        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect, border_radius=16)
        pygame.draw.rect(
            self.screen,
            SUMMARY_ENTRY_BORDER_COLOR,
            panel_rect,
            width=2,
            border_radius=16,
        )
        title = self.font_medium.render("Recipes Found", True, TEXT_COLOR)
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 16))

        spacing = 12
        slots = len(self.found_recipes)
        if slots == 0:
            return
        available_width = panel_rect.width - spacing * (slots + 1)
        entry_width = max(80, available_width // slots)
        entry_height = panel_rect.height - 80
        entry_top = panel_rect.y + 64
        current_x = panel_rect.x + spacing

        for idx, recipe in enumerate(self.found_recipes):
            entry_rect = pygame.Rect(current_x, entry_top, entry_width, entry_height)
            fill_color = SUMMARY_ENTRY_FOUND_COLOR if recipe else SUMMARY_ENTRY_PENDING_COLOR
            pygame.draw.rect(self.screen, fill_color, entry_rect, border_radius=12)
            pygame.draw.rect(
                self.screen,
                SUMMARY_ENTRY_BORDER_COLOR,
                entry_rect,
                width=1,
                border_radius=12,
            )

            icon_surface = self._recipe_image_surface(recipe)
            icon_surface = self._scale_surface(
                icon_surface, (entry_rect.height - 24, entry_rect.height - 24)
            )
            icon_rect = icon_surface.get_rect()
            icon_rect.left = entry_rect.x + 16
            icon_rect.centery = entry_rect.centery
            self.screen.blit(icon_surface, icon_rect)

            text_x = icon_rect.right + 16
            text_width = entry_rect.right - text_x - 16
            if text_width <= 0:
                text_x = icon_rect.right + 4
                text_width = entry_rect.right - text_x - 4

            primary_label = f"{idx + 1}. {recipe.display_name if recipe else '???'}"
            primary_surface = self.font_small.render(primary_label, True, TEXT_COLOR)
            primary_rect = primary_surface.get_rect()
            primary_rect.topleft = (text_x, entry_rect.y + 6)
            if primary_rect.width > text_width:
                primary_surface = self.font_xsmall.render(primary_label, True, TEXT_COLOR)
                primary_rect = primary_surface.get_rect()
                primary_rect.topleft = (text_x, entry_rect.y + 6)
            self.screen.blit(primary_surface, primary_rect)

            if recipe:
                ingredients = [self._display_name(name) for name in recipe.trio]
                detail_text = ", ".join(ingredients)
            else:
                detail_text = "Find the right trio to reveal this recipe."

            detail_lines = self._wrap_text(detail_text, self.font_xsmall, text_width)
            detail_y = primary_rect.bottom + 6
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

        prompt = self.font_small.render("Press ESC or close the window to exit", True, TEXT_COLOR)
        self.screen.blit(prompt, prompt.get_rect(center=(panel.centerx, panel.y + 210)))

    def _draw_version(self) -> None:
        version_surface = self.font_xsmall.render(VERSION, True, TEXT_MUTED_COLOR)
        version_rect = version_surface.get_rect()
        version_rect.bottomright = (SCREEN_WIDTH - 24, SCREEN_HEIGHT - 20)
        self.screen.blit(version_surface, version_rect)

    # --- Utility helpers -----------------------------------------------------
    def _build_card_layout(self) -> List[pygame.Rect]:
        card_width = 180
        card_height = 140
        padding_x = 20
        padding_y = 20
        start_x = 60
        start_y = 310
        rects: List[pygame.Rect] = []
        for row in range(CARD_ROWS):
            for column in range(CARD_COLUMNS):
                x = start_x + column * (card_width + padding_x)
                y = start_y + row * (card_height + padding_y)
                rects.append(pygame.Rect(x, y, card_width, card_height))
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
