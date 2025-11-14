import numpy as np
import pandas as pd
from collections import defaultdict
from io import StringIO

#-------------------------------------------------------------------------
'''
FEATURE: extract_team_stats
'''

#STATS DICTIONARY 
#from p1 and p2 lead
def build_name_to_stats(battles):
    name_to_stats = {}

    for b in battles:
        #P1
        for p in b.get("p1_team_details", []):
            n = (p.get("name") or "").lower()
            if n and n not in name_to_stats:
                name_to_stats[n] = {
                    "hp": p.get("base_hp", 0),
                    "atk": p.get("base_atk", 0),
                    "def": p.get("base_def", 0),
                    "spa": p.get("base_spa", 0),
                    "spd": p.get("base_spd", 0),
                    "spe": p.get("base_spe", 0),
                }

        #P2
        lead = b.get("p2_lead_details") or {}
        n = (lead.get("name") or "").lower()
        if n and n not in name_to_stats:
            name_to_stats[n] = {
                "hp": lead.get("base_hp", 0),
                "atk": lead.get("base_atk", 0),
                "def": lead.get("base_def", 0),
                "spa": lead.get("base_spa", 0),
                "spd": lead.get("base_spd", 0),
                "spe": lead.get("base_spe", 0),
            }

    return name_to_stats

#P2 pokemons first time they're seen (no repetitions)
def get_p2_seen_unique(battle, max_turns=30):
    seen = set()
    lead = battle.get("p2_lead_details") or {}
    if lead.get("name"):
        seen.add(lead["name"].lower())

    for turn in battle.get("battle_timeline", [])[:max_turns]:
        state = turn.get("p2_pokemon_state") or {}
        n = (state.get("name") or "").lower()
        if n:
            seen.add(n)

    return seen  

#MEAN STATS OVER P1 and P2
def safe_mean(vals):
    return float(np.mean(vals)) if vals else 0.0

def extract_team_stats(battle, stats_map, max_turns=30):
    p1_team = battle.get("p1_team_details", []) or []
    p1_hp  = [p.get("base_hp", 0)  for p in p1_team]
    p1_atk = [p.get("base_atk", 0) for p in p1_team]
    p1_def = [p.get("base_def", 0) for p in p1_team]
    p1_spa = [p.get("base_spa", 0) for p in p1_team]
    p1_spd = [p.get("base_spd", 0) for p in p1_team]
    p1_spe = [p.get("base_spe", 0) for p in p1_team]

    p2_names = get_p2_seen_unique(battle, max_turns=max_turns)

    p2_hp, p2_atk, p2_def, p2_spa, p2_spd, p2_spe = [], [], [], [], [], []
    for n in p2_names:
        s = stats_map.get(n)
        if s:
            p2_hp.append(s["hp"])
            p2_atk.append(s["atk"])
            p2_def.append(s["def"])
            p2_spa.append(s["spa"])
            p2_spd.append(s["spd"])
            p2_spe.append(s["spe"])

    return {
        "battle_id": battle.get("battle_id"), 
        "player_won": int(battle.get("player_won", 0)),

        "p1_mean_hp":  safe_mean(p1_hp),
        "p1_mean_atk": safe_mean(p1_atk),
        "p1_mean_def": safe_mean(p1_def),
        "p1_mean_spa": safe_mean(p1_spa),
        "p1_mean_spd": safe_mean(p1_spd),
        "p1_mean_spe": safe_mean(p1_spe),

        "p2_mean_hp":  safe_mean(p2_hp),
        "p2_mean_atk": safe_mean(p2_atk),
        "p2_mean_def": safe_mean(p2_def),
        "p2_mean_spa": safe_mean(p2_spa),
        "p2_mean_spd": safe_mean(p2_spd),
        "p2_mean_spe": safe_mean(p2_spe),

        "p2_seen_count": len(p2_names), 
    }

def make_team_stats_df(battles, stats_map, max_turns=30):
    rows = []
    for b in battles:
        rows.append(extract_team_stats(b, stats_map=stats_map, max_turns=max_turns))
    return pd.DataFrame(rows)
  
-------------------------------------------------------------------------
'''
FEATURE: chart implementation (not used in the latest versions of the code)
'''

csv_data = """Normal,Fire,Water,Electric,Grass,Ice,Fighting,Poison,Ground,Flying,Psychic,Bug,Rock,Ghost,Dragon,Dark,Steel,Fairy
Normal,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,1,1,0.5,1
Fire,1,0.5,0.5,1,2,2,1,1,1,1,1,2,0.5,1,0.5,1,2,1
Water,1,2,0.5,1,0.5,1,1,1,2,1,1,1,2,1,0.5,1,1,1
Electric,1,1,2,0.5,0.5,1,1,1,0,2,1,1,1,1,0.5,1,1,1
Grass,1,0.5,2,1,0.5,1,1,0.5,2,0.5,1,0.5,2,1,0.5,1,0.5,1
Ice,1,0.5,0.5,1,2,0.5,1,1,2,2,1,1,1,1,2,1,0.5,1
Fighting,2,1,1,1,1,2,1,0.5,1,0.5,0.5,0.5,2,0,1,2,2,0.5
Poison,1,1,1,1,2,1,1,0.5,0.5,1,1,1,0.5,0.5,1,1,0,2
Ground,1,2,1,2,0.5,1,1,2,1,0,1,0.5,2,1,1,1,2,1
Flying,1,1,1,0.5,2,1,2,1,1,1,1,2,0.5,1,1,1,0.5,1
Psychic,1,1,1,1,1,1,2,2,1,1,0.5,1,1,1,1,0,0.5,1
Bug,1,0.5,1,1,2,1,0.5,0.5,1,0.5,2,1,1,0.5,1,2,0.5,0.5
Rock,1,2,1,1,1,2,0.5,1,0.5,2,1,2,1,1,1,1,0.5,1
Ghost,0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,0.5,1,1
Dragon,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,0.5,0
Dark,1,1,1,1,1,1,0.5,1,1,1,2,1,1,2,1,0.5,1,0.5
Steel,1,0.5,0.5,0.5,1,2,1,1,1,1,1,1,2,1,1,1,0.5,2
Fairy,1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,2,2,0.5,1
"""

type_chart_df = pd.read_csv(StringIO(csv_data))
display(type_chart_df)

from collections import defaultdict

dataset_types = defaultdict(set)

for battle in train_data:
    #P1 pokémons, 6
    for p in battle.get("p1_team_details", []):
        name = p.get("name", "").lower()
        for t in p.get("types", []):
            if t != "notype" and t:
                dataset_types[name].add(t.lower())

    #P2 lead pokémon
    lead = battle.get("p2_lead_details", {})
    name = lead.get("name", "").lower()
    for t in lead.get("types", []):
        if t != "notype" and t:
            dataset_types[name].add(t.lower())
            
    #P2 pokémons appeared in the battle_timeline
    for turn in battle.get("battle_timeline", []):
        p2_state = turn.get("p2_pokemon_state", {})
        name = p2_state.get("name", "").lower()
        if name:
            pass #no types rn

types_dict = {name: sorted(list(types)) for name, types in dataset_types.items()}

print("All pokèmon names and relative types:\n")
for name, types in sorted(types_dict.items()):
    print(f"{name}: {types}")

all_types = sorted({t for types in types_dict.values() for t in types})

print(f"\nTot number of found pokémon types: {len(all_types)}")
print("Types:", ", ".join(all_types))

def compute_real_effectiveness(battle, type_chart_df, dataset_types):
    p1_super = 0
    p2_super = 0
    p1_null = 0
    p2_null = 0
    p1_half = 0
    p2_half = 0

    for turn in battle.get("battle_timeline", []):
        p1_move = turn.get("p1_move_details")
        if p1_move:
            move_type = p1_move.get("type", "").capitalize()
            defender_name = turn.get("p2_pokemon_state", {}).get("name", "").lower()
            defender_types = dataset_types.get(defender_name, [])
            
            for def_type in defender_types:
                def_type = def_type.capitalize()
                if move_type in type_chart_df.index and def_type in type_chart_df.columns:
                    eff = type_chart_df.loc[move_type, def_type]
                    if eff == 2.0:
                        p1_super += 1
                    if eff == 0:
                        p1_null += 1
                    if eff == 0.5:
                        p1_half += 1

        p2_move = turn.get("p2_move_details")
        if p2_move:
            move_type = p2_move.get("type", "").capitalize()
            defender_name = turn.get("p1_pokemon_state", {}).get("name", "").lower()
            defender_types = dataset_types.get(defender_name, [])
            
            for def_type in defender_types:
                def_type = def_type.capitalize()
                if move_type in type_chart_df.index and def_type in type_chart_df.columns:
                    eff = type_chart_df.loc[move_type, def_type]
                    if eff == 2.0:
                        p2_super += 1
                    if eff == 0:
                        p2_null += 1
                    if eff == 0.5:
                        p2_half += 1
    return {
        "p1_super": p1_super,
        "p2_super": p2_super,
        "p1_half": p1_half,
        "p2_half": p2_half,
        "p1_null": p1_null,
        "p2_null": p2_null
}


battle_effectiveness = []

for battle in train_data:
    eff = compute_real_effectiveness(battle, type_chart_df, dataset_types)
    
    battle_effectiveness.append({
        "battle_id": battle["battle_id"],
        "p1_super_effective": eff["p1_super"],
        "p2_super_effective": eff["p2_super"],
        "p1_half_effective": eff["p1_half"],
        "p2_half_effective": eff["p2_half"],
        "p1_null_effective": eff["p1_null"],
        "p2_null_effective": eff["p2_null"],
        "effectiveness_diff": eff["p1_super"] - eff["p2_super"],
        "player_won": battle["player_won"]
    })

effectiveness_df = pd.DataFrame(battle_effectiveness)
display(effectiveness_df.head())

#-------------------------------------------------------------------------
'''
FEATURE: types matchup --> unused in our code
'''

def mean_type_effectiveness(attacking_types, defending_types, type_chart_df):
    vals = []
    for atk in attacking_types:
        atk_cap = atk.capitalize()
        if atk_cap not in type_chart_df.index:
            continue
        for df in defending_types:
            df_cap = df.capitalize()
            if df_cap not in type_chart_df.columns:
                continue
            eff = type_chart_df.loc[atk_cap, df_cap]
            vals.append(float(eff))
    if not vals:
        return 1.0 
    return float(np.mean(vals))


def get_p1_lead_name(battle):
    timeline = battle.get("battle_timeline") or []
    if not timeline:
        return None
    first_state = timeline[0].get("p1_pokemon_state") or {}
    n = (first_state.get("name") or "").lower()
    return n or None

def get_p2_lead_name(battle):
    lead = battle.get("p2_lead_details") or {}
    n = (lead.get("name") or "").lower()
    return n or None

def lead_type_matchup_one_battle(battle, type_chart_df, types_dict):
    p1_name = get_p1_lead_name(battle)
    p2_name = get_p2_lead_name(battle)

    p1_types = types_dict.get(p1_name, [])
    p2_types = types_dict.get(p2_name, [])

    if not p1_types or not p2_types:
        adv_p1 = 1.0
        adv_p2 = 1.0
    else:
        adv_p1 = mean_type_effectiveness(p1_types, p2_types, type_chart_df)
        adv_p2 = mean_type_effectiveness(p2_types, p1_types, type_chart_df)

    return {
        "battle_id": battle.get("battle_id"),
        "lead_type_adv_p1": adv_p1,
        "lead_type_adv_p2": adv_p2,
        "lead_type_adv_diff": adv_p1 - adv_p2,
        "player_won": int(battle.get("player_won", 0)) if "player_won" in battle else 0,
    }

def make_lead_type_matchup_df(battles, type_chart_df, types_dict, include_target=True):
    rows = []
    for b in battles:
        row = lead_type_matchup_one_battle(b, type_chart_df, types_dict)
        if not include_target and "player_won" in row:
            row.pop("player_won", None)
        rows.append(row)
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------
'''
FEATURE: build_wr_map_from_battles (fold-safe winrate)
'''

#Win-rate calculated with SMOOTHING to avoid leakage in the fold 
def build_wr_map_from_battles(battles, M=100):
  
    #initializing a dict "stats" to count wins and loss for each pokemon 
    stats = defaultdict(lambda: {"games": 0, "wins": 0}) 
    
    global_wins = 0
    global_games = 0

    for b in battles:
        winner = bool(b.get("player_won", False))
        global_games += 1
        if winner:
            global_wins += 1

        #Adding results from P1 6 pokemons and P2 lead+timeline pokemons 
        #P1 team
        for p in (b.get("p1_team_details") or []):
            n = (p.get("name") or "").lower()
            if not n: 
                continue
            stats[n]["games"] += 1
            if winner:
                stats[n]["wins"] += 1

        #P2 lead and seen in the timeline
        seen_p2 = set()
        lead = b.get("p2_lead_details") or {}
        if lead.get("name"):
            seen_p2.add((lead["name"] or "").lower())
        for turn in (b.get("battle_timeline") or []):
            n = (turn.get("p2_pokemon_state") or {}).get("name", "")
            if n:
                seen_p2.add(n.lower())

        for n in seen_p2:
            stats[n]["games"] += 1
            if not winner:
                stats[n]["wins"] += 1

    #Global mean calculation
    global_mean_wr = (global_wins / global_games) if global_games > 0 else 0.5

    wr_map = {}
    for n, s in stats.items():
        g, w = s["games"], s["wins"]
        #Smoothing formula --> weighted mean 
        #Lo smoothing è importante soprattutto quando si lavora nei folds perchè lì i pokemon rari diventano ancora più rari. 
        wr_map[n] = (w + M * global_mean_wr) / (g + M)
    
    wr_map_with_default = defaultdict(lambda: global_mean_wr)
    wr_map_with_default.update(wr_map)
    
    return wr_map_with_default

def make_team_wr_df(battles, wr_map, max_turns=30):
    rows = []
    for b in battles:
        row = {
            "battle_id": b.get("battle_id"),
            "player_won": int(b.get("player_won", 0)),
        }

        p1_team = b.get("p1_team_details") or []
        p1_wrs = []
        for p in p1_team:
            n = (p.get("name") or "").lower()
            if n:
                p1_wrs.append(wr_map[n])

        if p1_wrs:
            row["p1_team_wr_avg"] = float(np.mean(p1_wrs))
            row["p1_team_wr_max"] = float(np.max(p1_wrs))
            row["p1_team_wr_min"] = float(np.min(p1_wrs))
        else:
            example_wr = next(iter(wr_map.values()))
            row["p1_team_wr_avg"] = example_wr
            row["p1_team_wr_max"] = example_wr
            row["p1_team_wr_min"] = example_wr

        lead = b.get("p2_lead_details") or {}
        if lead.get("name"):
            n2 = (lead["name"] or "").lower()
            row["p2_lead_wr"] = wr_map[n2]
        else:
            row["p2_lead_wr"] = next(iter(wr_map.values()))

        row["diff_wr_avg"] = row["p1_team_wr_avg"] - row["p2_lead_wr"]
        rows.append(row)

    return pd.DataFrame(rows)


#-------------------------------------------------------------------------
'''
FEATURE: switches_for_player
'''

def switches_for_player(battle, player_prefix="p2", max_turns=30):
    timeline = (battle.get("battle_timeline") or [])[:max_turns]
    if not timeline:
        return 0

    if player_prefix == "p2":
        lead = (battle.get("p2_lead_details") or {}).get("name")
    else:
        first_state = (timeline[0].get(f"{player_prefix}_pokemon_state") or {})
        lead = first_state.get("name")

    last = lead
    switches = 0

    for turn in timeline:
        curr = (turn.get(f"{player_prefix}_pokemon_state") or {}).get("name")
        if curr is not None and last is not None and curr != last:
            switches += 1
        if curr is not None:
            last = curr

    return switches

def make_switch_df(battles, max_turns=30):
    rows = []
    for b in battles:
        row = {
            "battle_id": b.get("battle_id"),
            "player_won": int(b.get("player_won", 0)),
            f"p1_switches_{max_turns}": switches_for_player(b, player_prefix="p1", max_turns=max_turns),
            f"p2_switches_{max_turns}": switches_for_player(b, player_prefix="p2", max_turns=max_turns),
        }
        rows.append(row)
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------
'''
FEATURE: p1_seen_count and p2_seen_count
'''

def get_seen_unique(battle, player_prefix, max_turns=30):
    seen = set()
    
    #Checking the lead
    if player_prefix == "p2":
        lead = (battle.get("p2_lead_details") or {}).get("name")
    else:
        timeline = battle.get("battle_timeline", [])
        first_state = (timeline[0].get(f"{player_prefix}_pokemon_state") or {}) if timeline else {}
        lead = first_state.get("name")
    if lead:
        seen.add(lead.lower())

    #Analysing the timeline
    for turn in battle.get("battle_timeline", [])[:max_turns]:
        state = turn.get(f"{player_prefix}_pokemon_state") or {}
        n = (state.get("name") or "").lower()
        if n:
            seen.add(n)
    return seen

def make_seen_p1p2_df(battles, turns_list=(10,20,30), include_target=True):
    rows = []
    for b in battles:
        row = {"battle_id": b.get("battle_id")}
        if include_target:
            row["player_won"] = int(b.get("player_won", 0))

        for T in turns_list:
            p1_seen = get_seen_unique(b, player_prefix="p1", max_turns=T)
            p2_seen = get_seen_unique(b, player_prefix="p2", max_turns=T)

            row[f"p1_seen_count_{T}"] = len(p1_seen)
            row[f"p2_seen_count_{T}"] = len(p2_seen)

        rows.append(row)

    return pd.DataFrame(rows)
  
#-------------------------------------------------------------------------
'''
FEATURE: make_hp_loss_df
'''

def make_hp_loss_df(battles, turns=30):
  
    rows = []
    for b in battles:
        timeline = (b.get("battle_timeline") or [])[:turns]

        p1_hp_list = []
        p2_hp_list = []

        for t in timeline:
            p1_state = t.get("p1_pokemon_state") or {}
            p2_state = t.get("p2_pokemon_state") or {}
            if p1_state.get("hp_pct") is not None:
                p1_hp_list.append(p1_state["hp_pct"])
            if p2_state.get("hp_pct") is not None:
                p2_hp_list.append(p2_state["hp_pct"])

        if timeline:
            last = timeline[-1]
            p1_last = (last.get("p1_pokemon_state") or {}).get("hp_pct", 1.0)
            p2_last = (last.get("p2_pokemon_state") or {}).get("hp_pct", 1.0)
        else:
            p1_last = 1.0
            p2_last = 1.0

        row = {
            "battle_id": b.get("battle_id"),
            "player_won": int(b.get("player_won", 0)),
            f"p1_hp_mean_{turns}": float(np.mean(p1_hp_list)) if p1_hp_list else 1.0,
            f"p2_hp_mean_{turns}": float(np.mean(p2_hp_list)) if p2_hp_list else 1.0,
            f"p1_hp_last_{turns}": p1_last,
            f"p2_hp_last_{turns}": p2_last,
        }
        rows.append(row)

    return pd.DataFrame(rows)


#-------------------------------------------------------------------------
'''
FEATURE: make_hp_loss_df_multi
'''

def make_hp_loss_df_multi(battles, turns_list=(10,20,30), include_target=True):
 
    rows = []

    for b in battles:
        row = {
            "battle_id": b.get("battle_id"),
        }
        if include_target:
            row["player_won"] = int(b.get("player_won", 0))

        for T in turns_list:
            timeline = (b.get("battle_timeline") or [])[:T]

            p1_hp_list = []
            p2_hp_list = []

            for t in timeline:
                p1_state = t.get("p1_pokemon_state") or {}
                p2_state = t.get("p2_pokemon_state") or {}
                if p1_state.get("hp_pct") is not None:
                    p1_hp_list.append(p1_state["hp_pct"])
                if p2_state.get("hp_pct") is not None:
                    p2_hp_list.append(p2_state["hp_pct"])

            if timeline:
                last = timeline[-1]
                p1_last = (last.get("p1_pokemon_state") or {}).get("hp_pct", 1.0)
                p2_last = (last.get("p2_pokemon_state") or {}).get("hp_pct", 1.0)
            else:
                p1_last = 1.0
                p2_last = 1.0

            row[f"p1_hp_mean_{T}"] = float(np.mean(p1_hp_list)) if p1_hp_list else 1.0
            row[f"p2_hp_mean_{T}"] = float(np.mean(p2_hp_list)) if p2_hp_list else 1.0
            row[f"p1_hp_last_{T}"] = p1_last
            row[f"p2_hp_last_{T}"] = p2_last

        rows.append(row)

    return pd.DataFrame(rows)


#-------------------------------------------------------------------------
'''
FEATURE: make_momentum_df
'''

def compute_momentum_features(battle, max_turns=30):
    
    timeline = (battle.get("battle_timeline") or [])[:max_turns]

    p1 = []
    p2 = []

    for t in timeline:
        s1 = (t.get("p1_pokemon_state") or {})
        s2 = (t.get("p2_pokemon_state") or {})

        hp1 = s1.get("hp_pct", 1.0)
        hp2 = s2.get("hp_pct", 1.0)

        p1.append(hp1)
        p2.append(hp2)

    if not p1:
        return {
            "momentum_final": 0.0,
            "momentum_max":   0.0,
            "momentum_min":   0.0,
            "momentum_auc":   0.0,
            "slope_1_5":      0.0,
            "slope_5_10":     0.0,
            "slope_10_20":    0.0,
        }

    momentum = [p2[i] - p1[i] for i in range(len(p1))]

    final_m = momentum[-1]
    max_m   = max(momentum)
    min_m   = min(momentum)

    auc = float(np.trapz(momentum))

    def safe_slope(arr, start, end):
        if len(arr) <= start or len(arr) <= end:
            return 0.0
        return arr[end] - arr[start]

    slope_1_5   = safe_slope(momentum, 0, min(4, len(momentum)-1))
    slope_5_10  = safe_slope(momentum, min(4,len(momentum)-1), min(9,len(momentum)-1))
    slope_10_20 = safe_slope(momentum, min(9,len(momentum)-1), min(19,len(momentum)-1))

    return {
        "momentum_final": final_m,
        "momentum_max": max_m,
        "momentum_min": min_m,
        "momentum_auc": auc,
        "slope_1_5": slope_1_5,
        "slope_5_10": slope_5_10,
        "slope_10_20": slope_10_20,
    }

def make_momentum_df(battles, max_turns=30):
    rows = []
    for b in battles:
        r = {"battle_id": b.get("battle_id")}
        r.update(compute_momentum_features(b, max_turns))
        if "player_won" in b:
            r["player_won"] = int(b["player_won"])
        rows.append(r)
    return pd.DataFrame(rows)


#-------------------------------------------------------------------------
'''
FEATURE: status
'''

VALID_STATI = ["par", "slp", "frz", "tox", "psn", "brn"]  

def _one_side_status_counts(battle, attacker_prefix="p1", defender_prefix="p2", turns=30):
    counts = {f"{attacker_prefix}_caused_{s}_t{turns}": 0 for s in VALID_STATI}
    for turn in battle.get("battle_timeline", [])[:turns]:
        dstate = (turn.get(f"{defender_prefix}_pokemon_state") or {})
        s = (dstate.get("status") or "").lower()
        if s in VALID_STATI:
            counts[f"{attacker_prefix}_caused_{s}_t{turns}"] += 1
    return counts

def status_counts_one_battle(battle, turns=30):
    row = {"battle_id": battle["battle_id"]}
    
    row.update(_one_side_status_counts(battle, "p1", "p2", turns))
    row.update(_one_side_status_counts(battle, "p2", "p1", turns))

    for s in VALID_STATI:
        row[f"{s}_diff_t{turns}"] = (
            row[f"p1_caused_{s}_t{turns}"] - row[f"p2_caused_{s}_t{turns}"]
        )
    
    for s in VALID_STATI:
        row[f"{s}_rate_t{turns}"] = (
            row[f"p1_caused_{s}_t{turns}"] + row[f"p2_caused_{s}_t{turns}"]
        ) / float(turns)
    
    if "player_won" in battle:
        row["player_won"] = int(battle["player_won"])
    
    return row

def make_status_df(battles, turns_list=(5, 10, 20)):
    rows = []
    for b in battles:
        r = {"battle_id": b["battle_id"]}
        for t in turns_list:
            r.update(status_counts_one_battle(b, t))
        if "player_won" in b:
            r["player_won"] = int(b["player_won"])
        rows.append(r)
    return pd.DataFrame(rows).fillna(0.0)

#-------------------------------------------------------------------------
'''
FEATURE: make_bad_status_turns_df
'''

BAD_STATUS = {"slp","frz","par"}

def stage_mult(stage: int) -> float:
    if stage >= 0:
        return (2 + stage) / 2.0
    else:
        return 2.0 / (2 - stage)

def effective_speed(name: str, boosts: dict, status: str, stats_map: dict) -> float:
    if not name:
        return 0.0
    base = (stats_map.get(name.lower(), {}) or {}).get("spe", 0)
    b = boosts or {}
    stage = int(b.get("spe", 0))
    eff = base * stage_mult(stage)
    if (status or "").lower() == "par":
        eff *= 0.25
    return float(eff)


def make_bad_status_turns_df(battles, turns=30):
    rows = []
    for b in battles:
        p1_bad = 0
        p2_bad = 0
        for turn in b.get("battle_timeline", [])[:turns]:
            s1 = (turn.get("p1_pokemon_state") or {})
            s2 = (turn.get("p2_pokemon_state") or {})
            if (s1.get("status") or "").lower() in BAD_STATUS:
                p1_bad += 1
            if (s2.get("status") or "").lower() in BAD_STATUS:
                p2_bad += 1
        rows.append({
            "battle_id": b["battle_id"],
            "player_won": int(b.get("player_won", 0)),
            f"p1_badstatus_turns_t{turns}": p1_bad,
            f"p2_badstatus_turns_t{turns}": p2_bad,
            f"badstatus_diff_t{turns}": p1_bad - p2_bad,
        })
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------
'''
FEATURE: speed advantage (good)
'''

BAD_STATUS = {"slp","frz","par"}

def stage_mult(stage: int) -> float:
    if stage >= 0:
        return (2 + stage) / 2.0
    else:
        return 2.0 / (2 - stage)

def effective_speed(name: str, boosts: dict, status: str, stats_map: dict) -> float:
    if not name:
        return 0.0
    base = (stats_map.get(name.lower(), {}) or {}).get("spe", 0)
    b = boosts or {}
    stage = int(b.get("spe", 0))
    eff = base * stage_mult(stage)
    if (status or "").lower() == "par":
        eff *= 0.25
    return float(eff)


def speed_advantage_features(battle, stats_map, turns=30):
    #Number of turns where p1 is faster than p2
    faster_p1 = 0
    faster_p2 = 0
    observed = 0
    for turn in battle.get("battle_timeline", [])[:turns]:
        s1 = (turn.get("p1_pokemon_state") or {})
        s2 = (turn.get("p2_pokemon_state") or {})
        n1 = (s1.get("name") or "").lower()
        n2 = (s2.get("name") or "").lower()
        if not n1 or not n2:
            continue
        v1 = effective_speed(n1, s1.get("boosts", {}), s1.get("status"), stats_map)
        v2 = effective_speed(n2, s2.get("boosts", {}), s2.get("status"), stats_map)
        if v1 == 0 and v2 == 0:
            continue
        observed += 1
        if v1 > v2:
            faster_p1 += 1
        elif v2 > v1:
            faster_p2 += 1
    rate_p1 = faster_p1 / observed if observed else 0.0
    rate_p2 = faster_p2 / observed if observed else 0.0
    return {
        f"speed_adv_p1_t{turns}": rate_p1,
        f"speed_adv_p2_t{turns}": rate_p2,
        f"speed_adv_diff_t{turns}": rate_p1 - rate_p2
    }

def make_speed_df(battles, stats_map, turns_list=(5,10,20,30)):
    rows = []
    for b in battles:
        row = {"battle_id": b["battle_id"]}
        for T in turns_list:
            row.update(speed_advantage_features(b, stats_map, T))
        if "player_won" in b:
            row["player_won"] = int(b["player_won"])
        rows.append(row)
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------
'''
FEATURE: ko_counts_one_battle
'''

def ko_counts_one_battle(battle, turns=30):
    p1_kos = p2_kos = 0
    prev1 = prev2 = None
    for turn in battle.get("battle_timeline", [])[:turns]:
        s1 = (turn.get("p1_pokemon_state") or {})
        s2 = (turn.get("p2_pokemon_state") or {})
        st1 = (s1.get("status") or "").lower()
        st2 = (s2.get("status") or "").lower()

        if st2 == "fnt" and (prev2 is None or prev2 != "fnt"):
            p1_kos += 1
        if st1 == "fnt" and (prev1 is None or prev1 != "fnt"):
            p2_kos += 1
        prev1, prev2 = st1, st2

    return {
        f"ko_for_p1_t{turns}": p1_kos,
        f"ko_for_p2_t{turns}": p2_kos,
        f"ko_diff_t{turns}": p1_kos - p2_kos
    }

def make_ko_df(battles, turns_list=(10,20,30)):
    rows = []
    for b in battles:
        row = {"battle_id": b["battle_id"]}
        for t in turns_list:
            row.update(ko_counts_one_battle(b, t))
        if "player_won" in b:
            row["player_won"] = int(b["player_won"])
        rows.append(row)
    return pd.DataFrame(rows)
#-------------------------------------------------------------------------
'''
FEATURE: first_ko_flag_one_battle
'''

def first_ko_flag_one_battle(battle, turns=30):
    prev1 = prev2 = None
    first_flag = 0

    for turn in battle.get("battle_timeline", [])[:turns]:
        s1 = (turn.get("p1_pokemon_state") or {})
        s2 = (turn.get("p2_pokemon_state") or {})

        st1 = (s1.get("status") or "").lower()
        st2 = (s2.get("status") or "").lower()

        if st2 == "fnt" and (prev2 is None or prev2 != "fnt") and first_flag == 0:
            first_flag = 1

        if st1 == "fnt" and (prev1 is None or prev1 != "fnt") and first_flag == 0:
            first_flag = -1

        prev1, prev2 = st1, st2

        if first_flag != 0:
            break

    return first_flag


def make_first_ko_df(battles, turns=30):
    rows = []
    for b in battles:
        row = {
            "battle_id": b["battle_id"],
            f"lead_ko_flag_t{turns}": first_ko_flag_one_battle(b, turns=turns)
        }
        if "player_won" in b:
            row["player_won"] = int(b["player_won"])
        rows.append(row)
    return pd.DataFrame(rows)
  
#-------------------------------------------------------------------------
'''
FEATURE: simple_status_events_one_battle
'''

VALID_STATI = {"par","slp","frz","tox","psn","brn"}  

def simple_status_events_one_battle(battle, max_turns=30):
    timeline = (battle.get("battle_timeline") or [])[:max_turns]

    p1_events = 0
    p2_events = 0

    prev1 = None
    prev2 = None

    for turn in timeline:
        s1 = (turn.get("p1_pokemon_state") or {}).get("status", "").lower()
        s2 = (turn.get("p2_pokemon_state") or {}).get("status", "").lower()

        if s1 in VALID_STATI and s1 != prev1:
            p1_events += 1

        if s2 in VALID_STATI and s2 != prev2:
            p2_events += 1

        prev1, prev2 = s1, s2

    return {
        f"p1_status_events_t{max_turns}": p1_events,
        f"p2_status_events_t{max_turns}": p2_events,
        f"status_events_diff_t{max_turns}": p1_events - p2_events,
    }

def make_simple_status_events_df(battles, max_turns=30):
    rows = []
    for b in battles:
        row = {"battle_id": b["battle_id"]}
        row.update(simple_status_events_one_battle(b, max_turns=max_turns))
        if "player_won" in b:
            row["player_won"] = int(b.get("player_won", 0))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)

#############################################################################################################

def assemble_features_from_battles(battles, wr_map, *, include_target=True, wr_turns=30):
    stats_map_local = build_name_to_stats(battles)

    TURN_LIST_MULTI = (10, 20, 30)

    stats_df        = make_team_stats_df(battles, stats_map_local, max_turns=wr_turns)
    wr_df           = make_team_wr_df(battles, wr_map, max_turns=wr_turns)
    switch_df       = make_switch_df(battles)
    seen_df         = make_seen_p1p2_df(battles, turns_list=TURN_LIST_MULTI, include_target=include_target)
    hp_multi_df     = make_hp_loss_df_multi(battles, turns_list=TURN_LIST_MULTI, include_target=include_target)
    hp_t30_df       = make_hp_loss_df(battles, turns=30)
    status_df       = make_status_df(battles, turns_list=(5, 10, 20))
    speed_df        = make_speed_df(battles, stats_map_local, turns_list=(5,10,20,30))
    badstat_df      = make_bad_status_turns_df(battles, turns=30)  
    ko_df           = make_ko_df(battles, turns_list=(10,20,30))  
    first_ko_df     = make_first_ko_df(battles, turns=30)
    momentum_df     = make_momentum_df(battles, max_turns=wr_turns)
    status_simple_df = make_simple_status_events_df(battles, max_turns=30)

    base_cols = ["battle_id"] + (["player_won"] if include_target else [])
    out = stats_df[base_cols].copy()

    def _merge(base, more):
        if more is None:
            return base
        m = more.copy()

        if "player_won" in m.columns:
            m = m.drop(columns=["player_won"])

        drop_obj = [
            c for c in m.columns
            if c not in ("battle_id", "player_won")
            and not pd.api.types.is_numeric_dtype(m[c])
        ]
        if drop_obj:
            m = m.drop(columns=drop_obj)

        return base.merge(m, on="battle_id", how="left")

    for df in [
        stats_df,
        wr_df,
        switch_df,
        seen_df,
        hp_multi_df,
        hp_t30_df,
        status_df,
        speed_df, 
        badstat_df,
        ko_df,
        first_ko_df,
        momentum_df,
        status_simple_df
    ]:
        out = _merge(out, df)

    for c in out.columns:
        if c not in ("battle_id", "player_won"):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out















