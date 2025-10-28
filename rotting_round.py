from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, MutableSequence, Optional, Sequence

from food_api import GameData, Ingredient


@dataclass
class IngredientCard:
    """Runtime wrapper that tracks decay for a single ingredient."""

    ingredient: Ingredient
    turns_in_hand: int = 0
    is_rotten: bool = False

    def freshen(self) -> None:
        """Reset decay counters when the card leaves and re-enters the hand."""

        self.turns_in_hand = 0
        self.is_rotten = False


class RottingRound:
    """Manage the basket/hand flow for a single round with rotting logic."""

    def __init__(
        self,
        data: GameData,
        basket: Iterable[Ingredient],
        *,
        hand_size: int,
    ) -> None:
        if hand_size <= 0:
            raise ValueError("hand_size must be a positive integer")
        self._data = data
        self.hand_size = hand_size
        self.basket: Deque[IngredientCard] = deque(
            IngredientCard(ingredient=item) for item in basket
        )
        self.hand: List[Optional[IngredientCard]] = [None] * hand_size
        self.lost = False
        self.draw_to_full()

    # -------------------------- Drawing & hand helpers --------------------------
    def draw_to_full(self) -> None:
        for index in range(len(self.hand)):
            if self.hand[index] is None and self.basket:
                drawn = self.basket.popleft()
                # Preserve the card's decay history so returning ingredients
                # continue rotting instead of resetting when redrawn.
                self.hand[index] = drawn
        self._update_loss_state()

    def cards_in_hand(self) -> List[IngredientCard]:
        return [card for card in self.hand if card is not None]

    def is_round_over(self) -> bool:
        return not self.basket

    def refresh_for_next_round(self) -> None:
        """Clear rotten states in preparation for a new round."""

        for card in self.hand:
            if card is not None:
                card.freshen()
        for card in self.basket:
            card.freshen()
        self.lost = False

    # ------------------------------ Turn resolution -----------------------------
    def play_attempt(self, indices: Sequence[int]) -> bool:
        """Attempt to cook with the selected hand indices."""

        if self.lost:
            return False
        if not indices:
            return False

        unique = sorted(set(indices))
        chosen: List[tuple[int, IngredientCard]] = []
        for index in unique:
            if index < 0 or index >= len(self.hand):
                raise IndexError("Selection index out of range for the current hand.")
            card = self.hand[index]
            if card is None or card.is_rotten:
                continue
            chosen.append((index, card))

        if not chosen:
            # If every selected slot is empty or rotten, re-evaluate the loss
            # condition so the round correctly ends when the hand has spoiled.
            self._update_loss_state()
            return False

        cards = [card for _, card in chosen]
        if len(cards) < 3 or len(cards) > 5:
            self._return_to_basket(chosen)
            self.draw_to_full()
            return False

        if self._data.is_valid_dish([card.ingredient for card in cards]):
            for index, _ in sorted(chosen, key=lambda pair: pair[0], reverse=True):
                self.hand[index] = None
            self.draw_to_full()
            return True

        self._return_to_basket(chosen)
        self.draw_to_full()
        return False

    def end_turn_decay(self) -> None:
        """Apply end-of-turn decay to all cards currently in the hand."""

        for card in self.hand:
            if card is None or card.is_rotten:
                continue
            card.turns_in_hand += 1
            limit = max(card.ingredient.rotten_turns, 0)
            if limit and card.turns_in_hand >= limit:
                card.is_rotten = True
                card.turns_in_hand = limit

        self._update_loss_state()

    # ------------------------------- Internal helpers ------------------------------
    def _return_to_basket(self, chosen: MutableSequence[tuple[int, IngredientCard]]) -> None:
        for index, card in sorted(chosen, key=lambda pair: pair[0], reverse=True):
            # Keep the decay progress when shuffling the card back into the
            # basket so the ingredient resumes rotting from the same state when
            # drawn again.
            card.turns_in_hand += 1
            limit = max(card.ingredient.rotten_turns, 0)
            if limit and card.turns_in_hand >= limit:
                card.turns_in_hand = limit
                card.is_rotten = True
            self.hand[index] = None
            self.basket.append(card)

    def _update_loss_state(self) -> None:
        if self.lost:
            return

        active_cards = [card for card in self.hand if card is not None]
        if not active_cards:
            return

        non_rotten_count = sum(not card.is_rotten for card in active_cards)
        if non_rotten_count == 0 or (non_rotten_count == 1 and len(active_cards) > 1):
            self.lost = True


def rot_circles(card: IngredientCard) -> dict[str, int | bool | list[str]]:
    """Return UI metadata describing the rot indicator for a card.

    The ``cells`` entry represents square markers that should be shown under the
    ingredient art.  Each position corresponds to one safe turn before the
    ingredient rots.  Cells with the value ``"filled"`` should be rendered as
    green squares while ``"empty"`` cells remain unfilled.
    """

    total = max(card.ingredient.rotten_turns, 0)
    filled = 0 if total == 0 else min(card.turns_in_hand, total)
    if card.is_rotten:
        filled = total
    cells = ["filled" if index < filled else "empty" for index in range(total)]
    return {
        "total": total,
        "filled": filled,
        "cells": cells,
        "is_rotten": card.is_rotten,
    }
