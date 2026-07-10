#!/usr/bin/env python3
"""Cross-check the generator's POKEMON table against PokeAPI (authoritative).
Verifies Gen-1 types, evolution method/level, and reports Pokedex genus (gap check).
"""
import json, sys, time, subprocess, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training_scripts.generate_pokemon_data import POKEMON

CACHE = Path("/tmp/pokeapi_cache"); CACHE.mkdir(exist_ok=True)
GEN = {"generation-i":1,"generation-ii":2,"generation-iii":3,"generation-iv":4,
       "generation-v":5,"generation-vi":6,"generation-vii":7,"generation-viii":8,"generation-ix":9}

def get(url):
    key = CACHE / (url.rstrip("/").split("/api/v2/")[-1].replace("/","_") + ".json")
    if key.exists():
        return json.loads(key.read_text())
    for attempt in range(3):
        try:
            out = subprocess.run(["curl","-s","--max-time","20",url],
                                 capture_output=True, text=True, timeout=25).stdout
            data = json.loads(out)
            key.write_text(json.dumps(data)); time.sleep(0.02); return data
        except Exception as e:
            time.sleep(0.5)
    raise RuntimeError(f"failed {url}")

TYPE_TITLE = lambda t: {"nidoran-f":"Nidoran-F"}.get(t, t.capitalize())

def gen1_types(pkmn):
    past = pkmn.get("past_types", [])
    if past:
        earliest = min(past, key=lambda p: GEN.get(p["generation"]["name"], 99))
        return [t["type"]["name"] for t in earliest["types"]]
    return [t["type"]["name"] for t in pkmn["types"]]

# ── Fetch all 151 ────────────────────────────────────────────────────────
print("Fetching 151 Pokemon + species + evolution chains from PokeAPI...", file=sys.stderr)
api_types, genus, chain_urls = {}, {}, {}
for i in range(1, 152):
    p = get(f"https://pokeapi.co/api/v2/pokemon/{i}/")
    api_types[i] = [t.capitalize() for t in gen1_types(p)]
    s = get(f"https://pokeapi.co/api/v2/pokemon-species/{i}/")
    g = next((x["genus"] for x in s["genera"] if x["language"]["name"]=="en"), "")
    genus[i] = g
    chain_urls[i] = s["evolution_chain"]["url"]
    if i % 30 == 0: print(f"  ...{i}", file=sys.stderr)

# ── Parse evolution chains (filter targets to id<=151 for Gen-1) ──────────
name_to_id = {}
for i in range(1,152):
    name_to_id[get(f"https://pokeapi.co/api/v2/pokemon/{i}/")["name"]] = i

api_evo = {}  # from_name(lower) -> list of (to_name_lower, trigger, level, item)
seen_chains = set()
def walk(node):
    frm = node["species"]["name"]
    for child in node["evolves_to"]:
        to = child["species"]["name"]
        for det in child["evolution_details"]:
            trig = det["trigger"]["name"]
            lvl = det.get("min_level")
            item = (det.get("item") or {}).get("name") if det.get("item") else None
            api_evo.setdefault(frm, []).append((to, trig, lvl, item))
        walk(child)
for url in set(chain_urls.values()):
    cid = url.rstrip("/").split("/")[-1]
    if cid in seen_chains: continue
    seen_chains.add(cid)
    walk(get(url)["chain"])

# ── Compare ──────────────────────────────────────────────────────────────
STONE_API = {"thunder-stone":"Thunder Stone","fire-stone":"Fire Stone",
             "water-stone":"Water Stone","leaf-stone":"Leaf Stone","moon-stone":"Moon Stone"}
type_errs, evo_errs = [], []
my = {num:(name,types,into) for num,name,types,into in POKEMON}

for num,(name,types,into) in sorted(my.items()):
    # types
    api_t = api_types[num]
    if set(types) != set(api_t):
        type_errs.append(f"#{num} {name}: table={types} pokeapi-gen1={api_t}")
    # evolutions: build api gen1 set of (to, kind, detail)
    api_list = api_evo.get(name.lower().replace("-f","-f").replace("-m","-m"), [])
    # map our weird names: Nidoran-F -> nidoran-f, Mr. Mime -> mr-mime, Farfetch'd -> farfetchd
    key = {"Nidoran-F":"nidoran-f","Nidoran-M":"nidoran-m","Mr. Mime":"mr-mime",
           "Farfetch'd":"farfetchd"}.get(name, name.lower())
    api_list = api_evo.get(key, [])
    api_norm = set()
    for to,trig,lvl,item in api_list:
        tid = name_to_id.get(to)
        if tid is None or tid>151: continue   # Gen-1 only
        if trig=="level-up": api_norm.add((to,"level",lvl))
        elif trig=="use-item": api_norm.add((to,"stone",STONE_API.get(item,item)))
        elif trig=="trade": api_norm.add((to,"trade",""))
        else: api_norm.add((to,trig,lvl))
    my_norm = set()
    for to,method,detail in into:
        my_norm.add((to.lower().replace("nidoran-f","nidoran-f"), method, detail if method!="trade" else ""))
    # normalize 'to' names to lowercase for compare
    my_cmp = set((t.lower().replace("mr. mime","mr-mime").replace("nidoran-f","nidoran-f"), m, d) for t,m,d in
                 [(to,method,detail if method!="trade" else "") for to,method,detail in into])
    api_cmp = set((t, m, (d if m!="trade" else "")) for t,m,d in api_norm)
    if my_cmp != api_cmp:
        evo_errs.append(f"#{num} {name}:\n      table : {sorted(my_cmp, key=str)}\n      pokeapi: {sorted(api_cmp, key=str)}")

print("\n================ TYPE CHECK (Gen-1) ================")
print("\n".join(type_errs) if type_errs else "All 151 types match PokeAPI Gen-1 typing. ✓")
print("\n================ EVOLUTION CHECK ================")
print("\n".join(evo_errs) if evo_errs else "All evolution methods/levels match PokeAPI (Gen-1 filtered). ✓")

print("\n================ POKEDEX GENUS (not in corpus — gap candidate) ================")
print("Sample:", "; ".join(f"{my[i][0]}={genus[i]}" for i in (1,4,7,25,150,151)))
