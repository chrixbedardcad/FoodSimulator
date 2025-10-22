# 🍲 Food Deck Simulator — Core Balancing Prototype

This project is a **Monte Carlo–based balancing simulator** for the *Food Card Deck Video Game*, a design inspired by *Balatro* and *Slay the Spire*, where cards represent **ingredients** instead of poker hands.

The simulator models the core **scoring, recipe-learning, and chef bias mechanics** that form the foundation of the gameplay loop.

---

## 🎮 Concept Overview

- Each **ingredient card** has:
  - A *Taste* tag (`Sweet`, `Umami`, `Salty`, `Sour`, or `Bitter`)
  - A *Chip value* (1–30)
- **Recipes** are sets of 3 ingredients (trios) that can be discovered and mastered.
- **Chefs** define groups of recipes and apply a bias to draw their own ingredients more often.
- **Themes** (markets) like *Mediterranean* or *Asian* define which ingredients are available in a round.
- Each **run** simulates drawing ingredients, forming trios, scoring them, and tracking mastery frequency.

The simulator helps find the right balance between:
- Ingredient frequency
- Scoring potential
- Taste distribution
- Chef advantage
- Recipe progression

---

## ⚙️ Core Mechanics Simulated

| Mechanic | Description |
|-----------|-------------|
| **Trio Draws** | 3 ingredients are drawn from a market deck (chef bias may alter odds). |
| **Scoring** | Each trio’s score = sum of chip values × taste synergy multiplier. |
| **Taste Matrix** | JSON-defined table that determines synergy between tastes (e.g. `Sweet + Salty = 3`). |
| **Learning Slot** | Tracks recipes being learned; cooking the same recipe twice masters it. |
| **Chef Bias** | Weighted draw probability toward a chef’s preferred ingredients. |
| **Market Themes** | Sets of ingredients by region or cuisine, used to vary gameplay flavor. |

---

## 🗂️ File Structure

```
FoodSimulator/
│
├── food_simulator.py       ← Main simulator with CLI & report generator
│
├── ingredients.json        ← All available ingredients with name, taste, and chip value
├── taste_matrix.json       ← Defines taste synergy multipliers
├── recipes.json            ← List of recipe trios
├── chefs.json              ← List of chefs, their recipes, and perk slots
├── themes.json             ← Market themes defining available ingredient pools
│
└── reports/                ← Auto-generated after each simulation run (TXT + CSV)
```

---

## 🧮 Running the Simulator

Example commands:

```bash
# Run 200 Monte Carlo simulations using the Mediterranean market
py food_simulator.py

# Asian market with 500 simulations
py food_simulator.py --runs 500 --theme Asian

# Reproducible run (seed fixed)
py food_simulator.py --runs 300 --theme Mediterranean --seed 42

# Change output folder for reports
py food_simulator.py --out ./data/reports
```

---

## 📊 Output Files

Each simulation produces reports in `/reports` (or your chosen output directory):

| File Type | Description |
|------------|-------------|
| `report_[theme]_runs[runs]_seed[seed]_timestamp.txt` | Human-readable summary |
| `..._ingredients.csv` | Usage counts per ingredient |
| `..._tastes.csv` | Taste distribution across all draws |
| `..._recipes.csv` | How many times each recipe was cooked |
| `..._scores.csv` | Individual run scores for statistical analysis |

Each report file includes a unique **timestamp and seed** for reproducibility.

---

## 🔍 Key Metrics

| Metric | Meaning |
|---------|----------|
| **Average Score** | Mean total points per run |
| **Std Deviation (±)** | Consistency indicator — higher = more swingy results |
| **p50 / p90 / p99** | Score percentiles (typical, strong, top-1%) |
| **Mastery Rate** | % of runs where a recipe reached mastery |
| **Chef-key per Draw** | Avg. number of chef-favored ingredients per trio |
| **Ingredient HHI** | Diversity index (0 = diverse, 1 = repetitive) |

---

## 🧠 For Developers (Codex Context)

- The project is **data-driven** — all content (ingredients, tastes, recipes, chefs, themes) is loaded from JSON.  
- You can **extend** the system by editing or adding new JSONs (no code changes needed).  
- The simulator output can be parsed in Codex or Unreal Engine for:
  - Automated balancing dashboards
  - Visualization of score distributions
  - Testing chef/recipe synergies
  - Future training data for AI opponents or difficulty scaling

---

## 🚀 Next Steps for Integration

1. **In Codex:**
   - Open `food_simulator.py` and explore how `simulate_run()` and `simulate_many()` build gameplay logic.
   - Add or modify JSON files to test new ingredients or chefs.
   - Visualize CSV outputs using `matplotlib`, `pandas`, or Unreal’s analytics tools.

2. **In Unreal Engine (5.3+):**
   - Import the JSONs as **Data Assets** (e.g., `UDataTable` or `UDataAsset` objects).
   - Use them to populate card decks, chefs, and market logic in Blueprints or C++.
   - Mirror the scoring rules for real-time gameplay simulation.

---

## 📚 Credits

Project: **Food Card Deck Video Game Prototype**  
Tech: Python 3.10+, JSON, NumPy (optional), CLI report generator  
Design Goal: Blend *culinary creativity* with *strategic deck-building mechanics*  
Inspired by: *Balatro*, *Slay the Spire*, and real-world flavor pairing theory.
