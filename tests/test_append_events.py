import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from food_desktop import FoodGameApp


def make_headless_app() -> FoodGameApp:
    app = FoodGameApp.__new__(FoodGameApp)
    app._append_log_lines = lambda messages: None
    app._pending_round_summary = None
    app._round_summary_shown = False
    app._deferring_round_summary = False
    app._show_basket_clear_popup = Mock()
    app._set_action_buttons_enabled = Mock()
    app._handle_run_finished = Mock()
    return app


def test_append_events_reenables_actions_when_round_resumes() -> None:
    app = make_headless_app()
    session = Mock()
    session.awaiting_new_round.return_value = False
    session.is_finished.return_value = False
    app.session = session

    app.append_events(["Round begins"])

    app._set_action_buttons_enabled.assert_called_once_with(True)
    app._handle_run_finished.assert_not_called()


def test_append_events_skips_reenable_when_round_pending() -> None:
    app = make_headless_app()
    session = Mock()
    session.awaiting_new_round.return_value = True
    session.peek_basket_clear_summary.return_value = {}
    session.is_finished.return_value = False
    app.session = session

    app.append_events(["Round pending"])

    app._set_action_buttons_enabled.assert_not_called()
    app._show_basket_clear_popup.assert_called_once()
