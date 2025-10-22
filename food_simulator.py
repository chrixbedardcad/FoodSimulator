# food_simulator.py with CLI flags + file outputs
# Usage examples:
#   py food_simulator.py --runs 300 --theme Asian
#   py food_simulator.py --runs 500 --theme Mediterranean --out reports
#   py food_simulator.py --seed 42 --runs 200 --theme Asian
from __future__ import annotations
import json, random, argparse, csv, os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime

# JSON file paths (must be in the same folder unless you pass absolute paths)
INGREDIENTS_JSON = "ingredients.json"
TASTE_JSON       = "taste_matrix.json"
RECIPES_JSON     = "recipes.json"
CHEFS_JSON       = "chefs.json"
THEMES_JSON      = "themes.json"

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    import statistics as stats
    HAVE_NUMPY = False

# ---------- Utilities ----------
def load_json(path:str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path:str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---------- Data models ----------
@dataclass(frozen=True)
class Ingredient:
    name: str
    taste: str
    chips: int

def load_ingredients(path:str)->Dict[str,Ingredient]:
    data = load_json(path)
    return {e["name"]: Ingredient(e["name"], e["taste"], int(e["chips"])) for e in data}

@dataclass(frozen=True)
class Recipe:
    name: str
    trio: Tuple[str,str,str]

def load_recipes(path:str)->List[Recipe]:
    data = load_json(path)
    return [Recipe(r["name"], tuple(r["trio"])) for r in data]

@dataclass
class Chef:
    name: str
    recipe_names: List[str]
    perks: dict

def load_chefs(path:str)->List[Chef]:
    data = load_json(path)
    return [Chef(c["name"], list(c.get("recipe_names",[])), dict(c.get("perks",{}))) for c in data]

def load_taste_matrix(path:str):
    data = load_json(path)
    matrix = data["matrix"]
    def score(a:str,b:str)->int:
        return matrix[a][b]
    return score

def load_themes(path:str):
    raw = load_json(path)
    fixed = {}
    for theme_name, items in raw.items():
        fixed[theme_name] = [(it["ingredient"], int(it["copies"])) for it in items]
    return fixed

# ---------- Loaded data (lazy; we load after parsing to allow alternate paths later if needed) ----------
INGREDIENTS: Dict[str,Ingredient] = {}
RECIPES: List[Recipe]             = []
CHEFS: List[Chef]                 = []
THEMES                            = {}
taste_score = None
REC_BY_NAME = {}
REC_TRIO_KEYS = {}

def initialize_data():
    global INGREDIENTS, RECIPES, CHEFS, THEMES, taste_score, REC_BY_NAME, REC_TRIO_KEYS
    INGREDIENTS = load_ingredients(INGREDIENTS_JSON)
    RECIPES     = load_recipes(RECIPES_JSON)
    CHEFS       = load_chefs(CHEFS_JSON)
    taste_score = load_taste_matrix(TASTE_JSON)
    THEMES      = load_themes(THEMES_JSON)
    REC_BY_NAME = {r.name:r for r in RECIPES}
    REC_TRIO_KEYS = {tuple(sorted(r.trio)): r.name for r in RECIPES}

def chef_key_ingredients(chef:Chef)->set:
    keys=set()
    for rn in chef.recipe_names:
        if rn in REC_BY_NAME:
            keys |= set(REC_BY_NAME[rn].trio)
    return keys

# ---------- Scoring ----------
def trio_score(ings: List[Ingredient]):
    a,b,c = ings
    chips = a.chips + b.chips + c.chips
    ts = taste_score(a.taste,b.taste) + taste_score(a.taste,c.taste) + taste_score(b.taste,c.taste)
    mult = max(1, ts)
    return chips*mult, chips, ts, mult

def which_recipe(ings: List[Ingredient])->str|None:
    key = tuple(sorted([i.name for i in ings]))
    return REC_TRIO_KEYS.get(key)

# ---------- Deck building & draw ----------
def build_market_deck(theme_pool:List[Tuple[str,int]], chef:Chef, deck_size:int=100, bias:float=2.7)->List[Ingredient]:
    keyset = chef_key_ingredients(chef)
    weighted: List[Ingredient] = []
    for name, copies in theme_pool:
        if name not in INGREDIENTS:
            continue
        w = bias if name in keyset else 1.0
        total = max(0, int(round(copies * w)))
        weighted.extend([INGREDIENTS[name]] * total)
    random.shuffle(weighted)
    if len(weighted) <= deck_size:
        return weighted.copy()
    # Sample without replacement to deck size
    out = []
    seen = set()
    for card in weighted:
        # keep order random; break once enough sampled
        out.append(card)
        if len(out) >= deck_size:
            break
    return out

def draw_cook(deck: List[Ingredient], chef:Chef, guarantee_prob:float=0.6):
    trio: List[Ingredient] = []
    if len(deck) < 3:
        return [], deck
    if random.random() < guarantee_prob:
        keys = [i for i in deck if i.name in chef_key_ingredients(chef)]
        if keys:
            pick = random.choice(keys)
            trio.append(pick)
            deck.remove(pick)
    while len(trio) < 3 and deck:
        pick = deck.pop()
        if pick not in trio:
            trio.append(pick)
    return trio, deck

@dataclass
class LearningState:
    recipe_name: str|None = None
    hits: int = 0

def update_learning(state:LearningState, chef:Chef, rec_name:str|None)->LearningState:
    if rec_name and rec_name in chef.recipe_names:
        if state.recipe_name == rec_name:
            state.hits += 1
        elif state.recipe_name is None:
            state.recipe_name, state.hits = rec_name, 1
        elif state.hits < 2:
            state.recipe_name, state.hits = rec_name, 1
    return state

# ---------- Simulation ----------
def simulate_run(theme_name:str="Mediterranean",
                 start_chef:Chef|None=None,
                 deck_size:int=100, cooks:int=6, rounds:int=3,
                 bias:float=2.7, guarantee_prob:float=0.6, reshuffle_every:int=8):
    theme_pool = THEMES[theme_name]
    chef = start_chef or random.choice(CHEFS)
    learning = LearningState()
    mastered = set()
    total_score = 0

    chefkey_per_draw: List[int] = []
    taste_counts = Counter()
    ingredient_use = Counter()
    recipe_counts = Counter()

    deck = build_market_deck(theme_pool, chef, deck_size, bias)
    draws = 0

    for _ in range(rounds):
        for _ in range(cooks):
            if len(deck) < 3 or draws >= reshuffle_every:
                deck = build_market_deck(theme_pool, chef, deck_size, bias)
                draws = 0
            trio, deck = draw_cook(deck, chef, guarantee_prob)
            if len(trio) < 3:
                break

            chefkey_per_draw.append(sum(1 for i in trio if i.name in chef_key_ingredients(chef)))
            for i in trio:
                taste_counts[i.taste] += 1
                ingredient_use[i.name] += 1

            sc, _, _, _ = trio_score(trio)
            total_score += sc

            rn = which_recipe(trio)
            if rn:
                recipe_counts[rn] += 1
            learning = update_learning(learning, chef, rn)

            if learning.recipe_name and learning.hits >= 2:
                mastered.add(learning.recipe_name)
                learning = LearningState()

            draws += 1

        if random.random() < 0.5:
            chef = random.choice(CHEFS)
            deck = build_market_deck(theme_pool, chef, deck_size, bias)
            draws = 0

    return {
        "score": total_score,
        "mastered": mastered,
        "ingredient_use": ingredient_use,
        "taste_counts": taste_counts,
        "chefkey_per_draw": chefkey_per_draw,
        "recipe_counts": recipe_counts,
    }

def summarize(scores:List[float]):
    if HAVE_NUMPY:
        arr = (scores if isinstance(scores, list) else list(scores))
        a = float(np.mean(arr)); s = float(np.std(arr))
        p50 = float(np.percentile(arr,50)); p90 = float(np.percentile(arr,90)); p99 = float(np.percentile(arr,99))
    else:
        a = stats.mean(scores); s = stats.pstdev(scores)
        arr = sorted(scores)
        def perc(p): 
            k = int(round((p/100.0)*(len(arr)-1)))
            return float(arr[max(0,min(k,len(arr)-1))])
        p50, p90, p99 = perc(50), perc(90), perc(99)
    return a, s, p50, p90, p99

def simulate_many(n:int=200, theme_name:str="Mediterranean", seed:int|None=None):
    if seed is not None:
        random.seed(seed)
    scores: List[float] = []
    mastered_any = 0
    mastered_counts = Counter()
    ing_all = Counter()
    taste_all = Counter()
    chefkey_all: List[int] = []
    recipe_all = Counter()

    for _ in range(n):
        res = simulate_run(theme_name=theme_name)
        scores.append(res["score"])
        if res["mastered"]:
            mastered_any += 1
        mastered_counts.update(res["mastered"])
        ing_all.update(res["ingredient_use"])
        taste_all.update(res["taste_counts"])
        chefkey_all.extend(res["chefkey_per_draw"])
        recipe_all.update(res["recipe_counts"])

    a,s,p50,p90,p99 = summarize(scores)
    total_ing = sum(ing_all.values())
    hhi = sum((cnt/total_ing)**2 for cnt in ing_all.values()) if total_ing else 0.0
    avg_ck = (sum(chefkey_all)/len(chefkey_all)) if chefkey_all else 0.0

    summary = {
        "runs": n,
        "theme": theme_name,
        "seed": seed,
        "average_score": round(a,2),
        "std_score": round(s,2),
        "p50": round(p50,2),
        "p90": round(p90,2),
        "p99": round(p99,2),
        "mastery_rate_pct": round(100.0*mastered_any/n,1),
        "avg_chef_key_per_draw": round(avg_ck,2),
        "ingredient_hhi": round(hhi,4),
    }
    return summary, ing_all, taste_all, recipe_all, scores

def write_report_files(out_dir:str, summary:dict, ing_all:Counter, taste_all:Counter, recipe_all:Counter, scores:List[float]):
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = summary.get("seed")
    theme = summary.get("theme")
    runs = summary.get("runs")
    base = f"report_{theme}_runs{runs}_seed{seed if seed is not None else 'rand'}_{ts}"
    # Summary TXT
    txt_path = os.path.join(out_dir, base + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Food Simulator Report ===\n")
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop Ingredients (usage):\n")
        for name,cnt in ing_all.most_common(20):
            f.write(f"{name},{cnt}\n")
        f.write("\nTaste Mix:\n")
        for t,cnt in taste_all.items():
            f.write(f"{t},{cnt}\n")
        f.write("\nMost Cooked Recipes:\n")
        for r,cnt in recipe_all.most_common(20):
            f.write(f"{r},{cnt}\n")
    # CSVs
    csv_ing = os.path.join(out_dir, base + "_ingredients.csv")
    with open(csv_ing, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ingredient","UseCount"])
        for name,cnt in ing_all.most_common():
            w.writerow([name,cnt])
    csv_taste = os.path.join(out_dir, base + "_tastes.csv")
    with open(csv_taste, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Taste","Count"])
        for t,cnt in taste_all.items():
            w.writerow([t,cnt])
    csv_rec = os.path.join(out_dir, base + "_recipes.csv")
    with open(csv_rec, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Recipe","TimesCooked"])
        for r,cnt in recipe_all.most_common():
            w.writerow([r,cnt])
    csv_scores = os.path.join(out_dir, base + "_scores.csv")
    with open(csv_scores, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["RunIndex","Score"])
        for i,sc in enumerate(scores):
            w.writerow([i,sc])
    return {
        "summary_txt": txt_path,
        "ingredients_csv": csv_ing,
        "tastes_csv": csv_taste,
        "recipes_csv": csv_rec,
        "scores_csv": csv_scores,
    }

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Deck Simulator")
    parser.add_argument("--runs", type=int, default=200, help="Number of Monte Carlo runs (default=200)")
    parser.add_argument("--theme", type=str, default="Mediterranean", help="Theme / Market name from themes.json")
    parser.add_argument("--out", type=str, default="reports", help="Output folder for report files (default=reports)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility (default=None=randomized)")
    args = parser.parse_args()

    initialize_data()

    themes_available = load_json(THEMES_JSON).keys()
    if args.theme not in themes_available:
        print(f"Theme '{args.theme}' not found. Available: {', '.join(themes_available)}")
        raise SystemExit(1)

    # If no seed provided, use a randomized seed from system time, but also print it for reproducibility
    seed_used = args.seed if args.seed is not None else int(datetime.now().timestamp()) & 0xFFFFFFFF
    random.seed(seed_used)

    summary, ing_all, taste_all, recipe_all, scores = simulate_many(args.runs, args.theme, seed_used)

    # Print short console summary
    print("=== SUMMARY ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    # Write files
    paths = write_report_files(args.out, summary, ing_all, taste_all, recipe_all, scores)
    print("\nFiles written:")
    for k,v in paths.items():
        print(f" - {k}: {v}")
