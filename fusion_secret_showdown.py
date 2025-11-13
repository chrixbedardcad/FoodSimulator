"""Pygame experience that bridges the secret recipe hunt and desktop cooking."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame

from food_api import GameData, Ingredient, Recipe

SCREEN_SIZE = (1024, 720)
BACKGROUND_COLOR = (22, 24, 31)
PANEL_COLOR = (36, 39, 52)
HIGHLIGHT_COLOR = (92, 179, 255)
TEXT_COLOR = (238, 238, 238)
MUTED_TEXT_COLOR = (190, 192, 204)
SUCCESS_COLOR = (101, 214, 133)
ALERT_COLOR = (241, 94, 104)

HAND_SIZE = 8
RECIPES_PER_HUNT = 3
COOKING_TARGET_SCORE = 140
MIN_COOK_INGREDIENTS = 3


@dataclass
class IngredientChoice:
    name: str
    rect: pygame.Rect
    is_correct: bool


@dataclass
class CookResult:
    recipe_name: Optional[str]
    score: int
    description: str
    multiplier: float
    alerts: Tuple[str, ...] = field(default_factory=tuple)


class SearchPhase:
    """Handles the secret recipe hunt section of the loop."""

    def __init__(self, game: "SecretShowdownGame") -> None:
        self.game = game
        self.pending_recipes: List[Recipe] = []
        self.current_recipe: Optional[Recipe] = None
        self.current_target: Optional[str] = None
        self.choices: List[IngredientChoice] = []
        self.feedback: Optional[Tuple[str, Tuple[int, int, int], str]] = None
        self.recipes_found = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        cookbook_names = {recipe.name for recipe in self.game.cookbook}
        available = [recipe for recipe in self.game.data.recipes if recipe.name not in cookbook_names]
        if len(available) < HAND_SIZE:
            available = list(self.game.data.recipes)
        self.pending_recipes = random.sample(available, k=min(len(available), max(HAND_SIZE, RECIPES_PER_HUNT * 2)))

    def start_round(self) -> None:
        if not self.pending_recipes:
            self._reshuffle()
        self.current_recipe = random.choice(self.pending_recipes)
        self.pending_recipes.remove(self.current_recipe)
        self.current_target = random.choice(self.current_recipe.trio)
        self._build_choices()
        self.feedback = None

    def _build_choices(self) -> None:
        assert self.current_recipe and self.current_target
        ingredient_names = list(self.game.data.ingredients.keys())
        ingredient_names.remove(self.current_target)
        random.shuffle(ingredient_names)
        filler = ingredient_names[: HAND_SIZE - 1]
        selections = filler + [self.current_target]
        random.shuffle(selections)

        self.choices = []
        padding_x = 60
        padding_y = 140
        card_w = 360
        card_h = 72
        gap_y = 14
        for index, name in enumerate(selections):
            column = index // 4
            row = index % 4
            rect = pygame.Rect(
                padding_x + column * (card_w + 40),
                padding_y + row * (card_h + gap_y),
                card_w,
                card_h,
            )
            self.choices.append(
                IngredientChoice(
                    name=name,
                    rect=rect,
                    is_correct=name == self.current_target,
                )
            )

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not self.current_recipe:
            self.start_round()
            return

        if not self.current_recipe:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._resolve_selection(event.pos)
            return

        if event.type == pygame.KEYDOWN:
            key_to_index = {
                pygame.K_1: 0,
                pygame.K_2: 1,
                pygame.K_3: 2,
                pygame.K_4: 3,
                pygame.K_q: 4,
                pygame.K_w: 5,
                pygame.K_e: 6,
                pygame.K_r: 7,
            }
            if event.key in key_to_index:
                index = key_to_index[event.key]
                if index < len(self.choices):
                    self._resolve_selection(self.choices[index].rect.center)

    def _resolve_selection(self, pos: Tuple[int, int]) -> None:
        for choice in self.choices:
            if choice.rect.collidepoint(pos):
                self._on_choice(choice)
                break

    def _on_choice(self, choice: IngredientChoice) -> None:
        assert self.current_recipe and self.current_target
        if choice.is_correct:
            self.recipes_found += 1
            recipe = self.current_recipe
            self.game.register_recipe(recipe)
            families = {self.game.data.ingredients[name].family for name in recipe.trio if name in self.game.data.ingredients}
            fam_text = ", ".join(sorted(families)) or "varied families"
            self.feedback = (
                f"You discovered {recipe.display_name or recipe.name}!",
                SUCCESS_COLOR,
                fam_text,
            )
            if self.recipes_found >= RECIPES_PER_HUNT:
                self.game.complete_search_round()
                self.current_recipe = None
                return
            self.start_round()
        else:
            ingredient = self.game.data.ingredients.get(choice.name)
            family = ingredient.family if ingredient else ""
            self.feedback = (f"{choice.name} doesn't match the secret recipe.", ALERT_COLOR, family)

    def draw(self, surface: pygame.Surface) -> None:
        title = self.game.font_title.render("Secret Recipe Hunt", True, TEXT_COLOR)
        surface.blit(title, (50, 40))

        subtitle = "Find the ingredient that belongs to the hidden recipe"
        details = self.game.font_small.render(subtitle, True, MUTED_TEXT_COLOR)
        surface.blit(details, (50, 90))

        status_text = f"Found this round: {self.recipes_found}/{RECIPES_PER_HUNT}"
        status_surface = self.game.font_small.render(status_text, True, TEXT_COLOR)
        surface.blit(status_surface, (50, 115))

        if not self.current_recipe:
            prompt = self.game.font_medium.render("Press space to draw your next recipe clue", True, TEXT_COLOR)
            surface.blit(prompt, (50, 200))
            return

        recipe_hint = self.game.describe_recipe_hint(self.current_recipe)
        hint_surface = self.game.font_small.render(recipe_hint, True, MUTED_TEXT_COLOR)
        surface.blit(hint_surface, (50, 150))

        for idx, choice in enumerate(self.choices):
            pygame.draw.rect(surface, PANEL_COLOR, choice.rect, border_radius=10)
            label = self.game.font_medium.render(f"{choice.name}", True, TEXT_COLOR)
            label_pos = label.get_rect(center=choice.rect.center)
            surface.blit(label, label_pos)
            badge_text = f"{idx+1}" if idx < 4 else "QWER"[idx - 4]
            badge = self.game.font_small.render(badge_text, True, MUTED_TEXT_COLOR)
            surface.blit(badge, (choice.rect.x + 10, choice.rect.y + 10))

        if self.feedback:
            message, color, extra = self.feedback
            text_surface = self.game.font_small.render(message, True, color)
            surface.blit(text_surface, (50, 560))
            if extra:
                fam_surface = self.game.font_small.render(f"Families spotted: {extra}", True, MUTED_TEXT_COLOR)
                surface.blit(fam_surface, (50, 590))


class CookingPhase:
    """Handles the cooking challenge using the pantry ingredients."""

    def __init__(self, game: "SecretShowdownGame") -> None:
        self.game = game
        self.cursor_index = 0
        self.selected: List[str] = []
        self.total_score = 0
        self.dishes_cooked = 0
        self.last_result: Optional[CookResult] = None
        self.reward_unlocked = False

    def reset(self) -> None:
        self.cursor_index = 0
        self.selected = []
        self.total_score = 0
        self.dishes_cooked = 0
        self.last_result = None
        self.reward_unlocked = False

    def pantry_items(self) -> List[str]:
        return sorted(self.game.pantry.keys())

    def handle_event(self, event: pygame.event.Event) -> None:
        items = self.pantry_items()
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_s):
                if items:
                    self.cursor_index = (self.cursor_index + 1) % len(items)
            elif event.key in (pygame.K_UP, pygame.K_w):
                if items:
                    self.cursor_index = (self.cursor_index - 1) % len(items)
            elif event.key == pygame.K_SPACE:
                if items:
                    choice = items[self.cursor_index]
                    self._toggle(choice)
            elif event.key == pygame.K_RETURN:
                self._cook_selection()
            elif event.key == pygame.K_ESCAPE:
                self.selected.clear()

    def _toggle(self, name: str) -> None:
        if name in self.selected:
            self.selected.remove(name)
        else:
            self.selected.append(name)

    def _cook_selection(self) -> None:
        if len(self.selected) < MIN_COOK_INGREDIENTS:
            self.last_result = CookResult(None, 0, "Select at least three ingredients to cook.", 0.0)
            return
        try:
            ingredients = [self.game.data.ingredients[name] for name in self.selected]
        except KeyError:
            self.last_result = CookResult(None, 0, "Some ingredients are missing data.", 0.0)
            return

        outcome = self.game.data.evaluate_dish(ingredients)
        score = int(round(outcome.dish_value))
        recipe_name = self.game.data.which_recipe(ingredients)
        alerts: Tuple[str, ...] = outcome.alerts if isinstance(outcome.alerts, tuple) else tuple(outcome.alerts)
        self.last_result = CookResult(recipe_name, score, outcome.entry.name if outcome.entry else "Freestyle", outcome.dish_multiplier, alerts)
        self.total_score += max(score, 0)
        self.dishes_cooked += 1
        self.selected = []

        if self.total_score >= COOKING_TARGET_SCORE or self.dishes_cooked >= 3:
            self.reward_unlocked = True
            self.game.complete_cooking_round()

    def draw(self, surface: pygame.Surface) -> None:
        title = self.game.font_title.render("Pantry Cook-Off", True, TEXT_COLOR)
        surface.blit(title, (50, 40))

        instructions = "Use W/S or arrows to navigate, space to select, enter to cook"
        instr_surface = self.game.font_small.render(instructions, True, MUTED_TEXT_COLOR)
        surface.blit(instr_surface, (50, 90))

        status = f"Score: {self.total_score}/{COOKING_TARGET_SCORE} — Dishes cooked: {self.dishes_cooked}"
        status_surface = self.game.font_small.render(status, True, TEXT_COLOR)
        surface.blit(status_surface, (50, 120))

        items = self.pantry_items()
        panel = pygame.Rect(50, 150, 360, 480)
        pygame.draw.rect(surface, PANEL_COLOR, panel, border_radius=12)

        for index, name in enumerate(items):
            y = panel.y + 12 + index * 36
            highlight = index == self.cursor_index
            color = HIGHLIGHT_COLOR if highlight else TEXT_COLOR
            label = self.game.font_small.render(name, True, color)
            surface.blit(label, (panel.x + 16, y))
            if name in self.selected:
                selected_label = self.game.font_small.render("✔", True, SUCCESS_COLOR)
                surface.blit(selected_label, (panel.right - 32, y))

        selection_panel = pygame.Rect(440, 150, 520, 220)
        pygame.draw.rect(surface, PANEL_COLOR, selection_panel, border_radius=12)
        select_title = self.game.font_medium.render("Current Selection", True, TEXT_COLOR)
        surface.blit(select_title, (selection_panel.x + 16, selection_panel.y + 16))
        if self.selected:
            block = ", ".join(self.selected)
            block_surface = self.game.font_small.render(block, True, TEXT_COLOR)
            surface.blit(block_surface, (selection_panel.x + 16, selection_panel.y + 60))
        else:
            empty_surface = self.game.font_small.render("Choose ingredients to craft a dish.", True, MUTED_TEXT_COLOR)
            surface.blit(empty_surface, (selection_panel.x + 16, selection_panel.y + 60))

        result_panel = pygame.Rect(440, 390, 520, 240)
        pygame.draw.rect(surface, PANEL_COLOR, result_panel, border_radius=12)
        result_title = self.game.font_medium.render("Latest Dish", True, TEXT_COLOR)
        surface.blit(result_title, (result_panel.x + 16, result_panel.y + 16))
        if self.last_result:
            desc = f"Outcome: {self.last_result.description}"
            desc_surface = self.game.font_small.render(desc, True, TEXT_COLOR)
            surface.blit(desc_surface, (result_panel.x + 16, result_panel.y + 60))
            score_text = f"Score: {self.last_result.score}  Multiplier: {self.last_result.multiplier:.2f}"
            score_surface = self.game.font_small.render(score_text, True, TEXT_COLOR)
            surface.blit(score_surface, (result_panel.x + 16, result_panel.y + 90))
            if self.last_result.recipe_name:
                recipe_surface = self.game.font_small.render(
                    f"Matched Recipe: {self.last_result.recipe_name}", True, SUCCESS_COLOR
                )
                surface.blit(recipe_surface, (result_panel.x + 16, result_panel.y + 120))
            if self.last_result.alerts:
                for offset, alert in enumerate(self.last_result.alerts):
                    alert_surface = self.game.font_small.render(alert, True, ALERT_COLOR)
                    surface.blit(alert_surface, (result_panel.x + 16, result_panel.y + 150 + offset * 24))
        else:
            waiting_surface = self.game.font_small.render("Cook a dish to see the results!", True, MUTED_TEXT_COLOR)
            surface.blit(waiting_surface, (result_panel.x + 16, result_panel.y + 60))


class SecretShowdownGame:
    """Controller that swaps between secret hunts and cooking rounds."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Secret Recipe Showdown")
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        self.running = True

        self.font_title = pygame.font.SysFont("Trebuchet MS", 42)
        self.font_medium = pygame.font.SysFont("Trebuchet MS", 28)
        self.font_small = pygame.font.SysFont("Trebuchet MS", 22)

        self.data = GameData.from_json()
        self.cookbook: List[Recipe] = []
        self.pantry: Dict[str, int] = {}
        self.search_phase = SearchPhase(self)
        self.cooking_phase = CookingPhase(self)
        self.state = "search"

    def describe_recipe_hint(self, recipe: Recipe) -> str:
        ingredient_objects: List[Ingredient] = [self.data.ingredients[name] for name in recipe.trio]
        families = {ingredient.family for ingredient in ingredient_objects}
        tastes = {ingredient.taste for ingredient in ingredient_objects}
        family_text = ", ".join(sorted(families))
        taste_text = ", ".join(sorted(tastes))
        return f"Clue: Families [{family_text}] | Flavors [{taste_text}]"

    def register_recipe(self, recipe: Recipe) -> None:
        if recipe not in self.cookbook:
            self.cookbook.append(recipe)
        for ingredient_name in recipe.trio:
            self.pantry[ingredient_name] = self.pantry.get(ingredient_name, 0) + 1

    def complete_search_round(self) -> None:
        self.state = "cooking"
        self.cooking_phase.reset()

    def complete_cooking_round(self) -> None:
        self.state = "search"
        self.search_phase = SearchPhase(self)
        self.search_phase.start_round()

    def run(self) -> None:
        self.search_phase.start_round()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.state == "search" and not self.search_phase.current_recipe:
                    self.running = False
                else:
                    if self.state == "search":
                        self.search_phase.handle_event(event)
                    else:
                        self.cooking_phase.handle_event(event)

            self.screen.fill(BACKGROUND_COLOR)
            if self.state == "search":
                self.search_phase.draw(self.screen)
            else:
                self.cooking_phase.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main() -> None:
    game = SecretShowdownGame()
    game.run()


if __name__ == "__main__":
    main()
