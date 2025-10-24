
# 🍲 Food Deck Simulator — Core Balancing Prototype

![Animated gameplay preview of Food Deck Simulator](Snakcrament.gif)

A toolkit for experimenting with the Food Card Deck video game concept. The repository now ships with two complementary entry points:

* `food_simulator.py` — a Monte Carlo batch simulator that stress tests balance across hundreds of automated runs and exports CSV/TXT reports.
* `food_game.py` — an interactive terminal mini-game that lets designers play short sessions using the exact same rules, decks, and scoring logic.
* `food_desktop.py` — a Tkinter-powered desktop client with clickable ingredient cards, live score tracking, and instant trio breakdowns.

Both scripts draw from a shared, JSON-driven content set (ingredients, recipes, chefs, and themes) so that data tweaks are immediately reflected everywhere.

---

## ✨ Features at a Glance

- **Data-first design**: All cards, recipes, chefs, and taste synergies live in JSON files. Adjust the numbers or add new entries without touching Python code.
- **Monte Carlo balance passes**: Run hundreds of fully automated sessions to inspect scoring distributions, mastery rates, and ingredient diversity.
- **Automatic report generation**: Every batch run produces a timestamped TXT summary plus CSVs for ingredients, tastes, recipes, and per-run scores.
- **Interactive prototyping loop**: Use the terminal mini-game to make choices turn-by-turn, preview chef perks, and see trio scoring breakdowns in real time.

---

## 🗂️ Project Structure

```
FoodSimulator/
├── food_api.py             ← Shared rules/data loader powering both entry points
├── food_simulator.py       ← Monte Carlo simulator with CLI & report writer
├── food_game.py            ← Interactive terminal harness for manual playtests
├── food_desktop.py         ← Desktop GUI built with Tkinter for visual playtests
├── ingredients.json        ← Ingredient cards (taste tags + chip values)
├── recipes.json            ← Recipe trios that can be discovered and mastered
├── chefs.json              ← Chef definitions, signature recipes, and perks
├── themes.json             ← Market/theme decks that feed the draw pile
├── taste_matrix.json       ← Taste synergy multipliers between flavour pairs
└── README.md
```

---

## ⚙️ Requirements

- Python 3.10 or newer.
- `numpy` is optional; if unavailable the simulator gracefully falls back to Python's `statistics` module for percentile calculations.

---

## 🚀 Running the Monte Carlo Simulator

Both command-line interfaces lean on the shared `food_api.py` module for loading
ingredients, applying chef perks, scoring trios, and writing reports. The
simulator exposes several CLI flags so you can automate balance passes from the
terminal:

1. Ensure you're in the project root and initialize the dataset implicitly by running the script.
2. Execute the simulator with Python:

```bash
python food_simulator.py --runs 300 --theme Mediterranean
python food_simulator.py --runs 500 --theme Asian --out reports
python food_simulator.py --runs 200 --theme Mediterranean --seed 42
python food_simulator.py --runs 1 --seed 1 --active-chefs 3 --hand-size 8 --pick-size 5
python food_simulator.py --help  # view every available option
```

Key CLI flags:

| Flag | Description |
|------|-------------|
| `--nbplay` | Number of full game plays to simulate; each play chains `--runs` runs (default `1`). |
| `--runs` | Number of runs executed within each play (default `200`). |
| `--theme` | Market/theme name drawn from `themes.json` (default `Mediterranean`). |
| `--out` | Output directory for generated reports (default `reports/`). |
| `--seed` | RNG seed. When omitted a random seed is chosen and printed for reproducibility. |
| `--rounds` | Rounds per run; each round deals fresh hands (default `3`). |
| `--cooks-per-round` | Cooking turns taken within each round (default `6`). |
| `--active-chefs` | Active chefs per run, influencing deck bias and perks (default `3`). |
| `--hand-size` | Number of ingredients drawn into your hand before each pick (default `8`). |
| `--pick-size` | How many ingredients are cooked each turn (default `5`; must not exceed `--hand-size`). |

`nbplay` × `runs` yields the total number of simulated runs; combined with `rounds` and
`cooks-per-round` this mirrors the round/cook structure available in `food_game.py`.

### How rounds and cooks shape a run

Every simulated run (and every interactive session) uses the same nested loop:

1. Play through the configured number of rounds.
2. Within each round, take `cooks-per-round` cook turns.
3. Each cook turn draws a hand, lets you pick a trio, scores it, then advances to the next cook.

This means a single run always contains `rounds × cooks-per-round` cook turns. For example,
the default configuration of three rounds with six cooks per round produces 18 cook turns per
run. Dropping the cooks per round to four would yield 12 turns instead, while increasing rounds
to five would expand the total to 30 turns. Because `food_simulator.py` and `food_game.py`
share this loop, their statistics and hands-on experience stay aligned.

After each batch completes the script prints a console summary including:

- Average score, standard deviation, and p50/p90/p99 percentiles.
- Mastery rate (% of runs where any recipe hit mastery).
- Average count of chef-favoured ingredients per trio.
- Ingredient Herfindahl–Hirschman Index (HHI) to gauge draw diversity.

Report files land in the requested output directory using the pattern `report_<theme>_plays<nbplay>_runs<runs>_seed<seed>_<timestamp>.*`.

---

## 🕹️ Interactive Terminal Prototype

`food_game.py` mirrors the simulator rules but lets you choose trios manually.
Launch it straight from the terminal to enter the interactive CLI loop powered
by `food_api.py` (all configuration happens via on-screen prompts, so there are
no additional flags to remember):

```bash
python food_game.py
```

During each session you can:

1. Pick a market theme and chef (or opt for random selection).
2. Decide how many rounds to play, how many cooks happen within each round, and how
   many runs to chain back-to-back.
3. (Optionally) set an RNG seed for reproducible decks.
4. Review eight-card hands, pick any trio, and instantly inspect chip totals, taste multipliers, chef key cards, and recipe completions.

Chef perks defined in `chefs.json` apply automatically—e.g., recipe-specific score multipliers—so designers can evaluate perk tuning without code changes.

---

## 🖥️ Desktop GUI Prototype

Prefer clickable cards over command-line prompts? Launch the Tkinter interface to play the same rule set with a richer presentation:

```bash
python food_desktop.py
```

Key highlights:

- Pick a market theme and one or more chefs from dropdowns and multi-select lists.
- See eight-card hands rendered as colour-coded tiles that highlight chef key ingredients.
- Click cards to build your trio (the UI enforces the pick limit) and review chip totals, taste synergy, recipe bonuses, and cumulative score in a side panel.
- Track round/turn progress, deck refresh events, and active chefs without leaving the window.

Because the GUI reuses the shared `food_api.py` helpers, any JSON data tweak immediately flows to the desktop client, terminal harness, and simulator alike.

---

## 🎯 Mechanics & Metrics Modelled

| Mechanic | Description |
|----------|-------------|
| **Trio draws** | Eight-card hands are dealt from theme-driven decks; trios are cooked each turn. |
| **Taste synergy** | Taste combinations look up multipliers in `taste_matrix.json` to scale chip totals. |
| **Recipe mastery** | Cooking the same signature recipe twice in a row masters it and contributes to mastery rate metrics. |
| **Chef bias** | Signature ingredients appear more frequently thanks to weighted deck construction. |
| **Chef perks** | Additional modifiers (such as recipe multipliers) are pulled from `chefs.json` and applied during scoring. |
| **Market swaps** | Runs can rotate chefs mid-simulation and rebuild decks to stress test variety. |

---

## 🍽️ Taste Icon Reference

| Icon | Taste Name | Description / Examples |
|------|------------|------------------------|
| 🍬 | Sweet | sugar, honey, fruit syrups, desserts |
| 🧂 | Salty | cured, savory, seasoning-rich |
| 🍋 | Sour | citrus, vinegar, yogurt tang |
| 🍄 | Umami | broths, mushrooms, soy, cooked meats |
| ☕ | Bitter | dark greens, coffee, cocoa, charred foods |

---

## 📊 Report Contents

Each Monte Carlo batch exports the following artefacts:

| File | Description |
|------|-------------|
| `*.txt` | Human-readable snapshot of run parameters, summary stats, and top-used content. |
| `*_ingredients.csv` | Ingredient usage counts across all simulated trios. |
| `*_tastes.csv` | Taste distribution tally (useful for balance heatmaps). |
| `*_recipes.csv` | Frequency of recipe completions to track mastery difficulty. |
| `*_scores.csv` | Per-run score log for deeper statistical analysis. |

---

## 🔧 Extending the Sandbox

- Add or tweak ingredient entries in `ingredients.json` to explore new taste/chip combinations.
- Expand `recipes.json`, `chefs.json`, or `themes.json` to introduce fresh synergies and market flavours.
- Iterate quickly by playtesting in `food_game.py`, then run `food_simulator.py` to validate balance at scale.

---

## 📚 Credits

- Project: **Food Card Deck Video Game Prototype**
- Tech: Python 3.10+, JSON, optional NumPy
- Design Goal: Blend *culinary creativity* with *strategic deck-building mechanics*
- Inspired by: *Balatro*, *Slay the Spire*, and flavour pairing research
