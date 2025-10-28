import importlib
import sys
import types
from pathlib import Path


def _install_pil_stub() -> None:
    if "PIL" in sys.modules:
        return

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.Resampling = types.SimpleNamespace(LANCZOS="LANCZOS")

    def _fake_open(_path):  # pragma: no cover - never executed in tests
        raise AssertionError("Image.open should not be called during asset path tests")

    image_module.open = _fake_open
    image_draw_module = types.ModuleType("PIL.ImageDraw")
    image_font_module = types.ModuleType("PIL.ImageFont")
    image_font_module.load_default = lambda: None
    image_tk_module = types.ModuleType("PIL.ImageTk")
    image_tk_module.PhotoImage = lambda *args, **kwargs: None

    pil_module.Image = image_module
    pil_module.ImageDraw = image_draw_module
    pil_module.ImageFont = image_font_module
    pil_module.ImageTk = image_tk_module

    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module
    sys.modules["PIL.ImageDraw"] = image_draw_module
    sys.modules["PIL.ImageFont"] = image_font_module
    sys.modules["PIL.ImageTk"] = image_tk_module


def _load_food_desktop(monkeypatch, tmp_path):
    sys.modules.pop("food_desktop", None)
    _install_pil_stub()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent))
    module = importlib.import_module("food_desktop")
    monkeypatch.setattr(module, "ASSET_DIR", tmp_path)
    monkeypatch.setattr(module, "RECIPE_ASSET_DIR", tmp_path / "recipes")
    module._recipe_image_cache.clear()
    return module


def test_finds_recipe_art_in_primary_directory(tmp_path, monkeypatch):
    module = _load_food_desktop(monkeypatch, tmp_path)
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    image_path = recipes_dir / "CheesyTomatoStack.png"
    image_path.write_bytes(b"fake")

    found = module._find_recipe_image_path("CheesyTomatoStack")

    assert found == image_path


def test_case_insensitive_recipe_art_lookup(tmp_path, monkeypatch):
    module = _load_food_desktop(monkeypatch, tmp_path)
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    image_path = recipes_dir / "cheesytomatostack.PNG"
    image_path.write_bytes(b"fake")

    found = module._find_recipe_image_path("CheesyTomatoStack")

    assert found == image_path
