#!/usr/bin/env python3
"""Generate a Gen-1 (Kanto, original 151) Pokedex training corpus.

Produces Q:/A: pairs and prose passages in the same format as
hardwareone_rich.txt, ready for train_tiny_model(_gpu).py.

Single source of truth: the POKEMON table below. Every answer is derived
from it, so facts can never contradict across phrasings. Edit the table
(or the world-knowledge sections) and re-run to rebuild the corpus.

All facts are Red/Blue/Yellow-era:
  - Clefairy, Clefable, Jigglypuff, Wigglytuff, Mr. Mime are Normal/Psychic
    (they only became Fairy in Gen 6).
  - Magnemite/Magneton are pure Electric (gained Steel in Gen 2).
  - No abilities, no held items, no natures.

Usage:
    python training_scripts/generate_pokemon_data.py \
        --out training_data/pokemon_kanto.txt
"""
import argparse
import random
from pathlib import Path

from pokemon_gen1_supplement import CATCH_LOC, DEX_ENTRY

VALID_TYPES = {
    "Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon",
}

# ── The 151. (num, name, [types], [(evolves_into, method, detail), ...]) ──
# method ∈ {"level", "stone", "trade"}; detail = level int or stone name or "".
POKEMON = [
    (1,  "Bulbasaur",  ["Grass", "Poison"],   [("Ivysaur", "level", 16)]),
    (2,  "Ivysaur",    ["Grass", "Poison"],   [("Venusaur", "level", 32)]),
    (3,  "Venusaur",   ["Grass", "Poison"],   []),
    (4,  "Charmander", ["Fire"],              [("Charmeleon", "level", 16)]),
    (5,  "Charmeleon", ["Fire"],              [("Charizard", "level", 36)]),
    (6,  "Charizard",  ["Fire", "Flying"],    []),
    (7,  "Squirtle",   ["Water"],             [("Wartortle", "level", 16)]),
    (8,  "Wartortle",  ["Water"],             [("Blastoise", "level", 36)]),
    (9,  "Blastoise",  ["Water"],             []),
    (10, "Caterpie",   ["Bug"],               [("Metapod", "level", 7)]),
    (11, "Metapod",    ["Bug"],               [("Butterfree", "level", 10)]),
    (12, "Butterfree", ["Bug", "Flying"],     []),
    (13, "Weedle",     ["Bug", "Poison"],     [("Kakuna", "level", 7)]),
    (14, "Kakuna",     ["Bug", "Poison"],     [("Beedrill", "level", 10)]),
    (15, "Beedrill",   ["Bug", "Poison"],     []),
    (16, "Pidgey",     ["Normal", "Flying"],  [("Pidgeotto", "level", 18)]),
    (17, "Pidgeotto",  ["Normal", "Flying"],  [("Pidgeot", "level", 36)]),
    (18, "Pidgeot",    ["Normal", "Flying"],  []),
    (19, "Rattata",    ["Normal"],            [("Raticate", "level", 20)]),
    (20, "Raticate",   ["Normal"],            []),
    (21, "Spearow",    ["Normal", "Flying"],  [("Fearow", "level", 20)]),
    (22, "Fearow",     ["Normal", "Flying"],  []),
    (23, "Ekans",      ["Poison"],            [("Arbok", "level", 22)]),
    (24, "Arbok",      ["Poison"],            []),
    (25, "Pikachu",    ["Electric"],          [("Raichu", "stone", "Thunder Stone")]),
    (26, "Raichu",     ["Electric"],          []),
    (27, "Sandshrew",  ["Ground"],            [("Sandslash", "level", 22)]),
    (28, "Sandslash",  ["Ground"],            []),
    (29, "Nidoran-F",  ["Poison"],            [("Nidorina", "level", 16)]),
    (30, "Nidorina",   ["Poison"],            [("Nidoqueen", "stone", "Moon Stone")]),
    (31, "Nidoqueen",  ["Poison", "Ground"],  []),
    (32, "Nidoran-M",  ["Poison"],            [("Nidorino", "level", 16)]),
    (33, "Nidorino",   ["Poison"],            [("Nidoking", "stone", "Moon Stone")]),
    (34, "Nidoking",   ["Poison", "Ground"],  []),
    (35, "Clefairy",   ["Normal"],            [("Clefable", "stone", "Moon Stone")]),
    (36, "Clefable",   ["Normal"],            []),
    (37, "Vulpix",     ["Fire"],              [("Ninetales", "stone", "Fire Stone")]),
    (38, "Ninetales",  ["Fire"],              []),
    (39, "Jigglypuff", ["Normal"],            [("Wigglytuff", "stone", "Moon Stone")]),
    (40, "Wigglytuff", ["Normal"],            []),
    (41, "Zubat",      ["Poison", "Flying"],  [("Golbat", "level", 22)]),
    (42, "Golbat",     ["Poison", "Flying"],  []),
    (43, "Oddish",     ["Grass", "Poison"],   [("Gloom", "level", 21)]),
    (44, "Gloom",      ["Grass", "Poison"],   [("Vileplume", "stone", "Leaf Stone")]),
    (45, "Vileplume",  ["Grass", "Poison"],   []),
    (46, "Paras",      ["Bug", "Grass"],      [("Parasect", "level", 24)]),
    (47, "Parasect",   ["Bug", "Grass"],      []),
    (48, "Venonat",    ["Bug", "Poison"],     [("Venomoth", "level", 31)]),
    (49, "Venomoth",   ["Bug", "Poison"],     []),
    (50, "Diglett",    ["Ground"],            [("Dugtrio", "level", 26)]),
    (51, "Dugtrio",    ["Ground"],            []),
    (52, "Meowth",     ["Normal"],            [("Persian", "level", 28)]),
    (53, "Persian",    ["Normal"],            []),
    (54, "Psyduck",    ["Water"],             [("Golduck", "level", 33)]),
    (55, "Golduck",    ["Water"],             []),
    (56, "Mankey",     ["Fighting"],          [("Primeape", "level", 28)]),
    (57, "Primeape",   ["Fighting"],          []),
    (58, "Growlithe",  ["Fire"],              [("Arcanine", "stone", "Fire Stone")]),
    (59, "Arcanine",   ["Fire"],              []),
    (60, "Poliwag",    ["Water"],             [("Poliwhirl", "level", 25)]),
    (61, "Poliwhirl",  ["Water"],             [("Poliwrath", "stone", "Water Stone")]),
    (62, "Poliwrath",  ["Water", "Fighting"], []),
    (63, "Abra",       ["Psychic"],           [("Kadabra", "level", 16)]),
    (64, "Kadabra",    ["Psychic"],           [("Alakazam", "trade", "")]),
    (65, "Alakazam",   ["Psychic"],           []),
    (66, "Machop",     ["Fighting"],          [("Machoke", "level", 28)]),
    (67, "Machoke",    ["Fighting"],          [("Machamp", "trade", "")]),
    (68, "Machamp",    ["Fighting"],          []),
    (69, "Bellsprout", ["Grass", "Poison"],   [("Weepinbell", "level", 21)]),
    (70, "Weepinbell", ["Grass", "Poison"],   [("Victreebel", "stone", "Leaf Stone")]),
    (71, "Victreebel", ["Grass", "Poison"],   []),
    (72, "Tentacool",  ["Water", "Poison"],   [("Tentacruel", "level", 30)]),
    (73, "Tentacruel", ["Water", "Poison"],   []),
    (74, "Geodude",    ["Rock", "Ground"],    [("Graveler", "level", 25)]),
    (75, "Graveler",   ["Rock", "Ground"],    [("Golem", "trade", "")]),
    (76, "Golem",      ["Rock", "Ground"],    []),
    (77, "Ponyta",     ["Fire"],              [("Rapidash", "level", 40)]),
    (78, "Rapidash",   ["Fire"],              []),
    (79, "Slowpoke",   ["Water", "Psychic"],  [("Slowbro", "level", 37)]),
    (80, "Slowbro",    ["Water", "Psychic"],  []),
    (81, "Magnemite",  ["Electric"],          [("Magneton", "level", 30)]),
    (82, "Magneton",   ["Electric"],          []),
    (83, "Farfetch'd", ["Normal", "Flying"],  []),
    (84, "Doduo",      ["Normal", "Flying"],  [("Dodrio", "level", 31)]),
    (85, "Dodrio",     ["Normal", "Flying"],  []),
    (86, "Seel",       ["Water"],             [("Dewgong", "level", 34)]),
    (87, "Dewgong",    ["Water", "Ice"],      []),
    (88, "Grimer",     ["Poison"],            [("Muk", "level", 38)]),
    (89, "Muk",        ["Poison"],            []),
    (90, "Shellder",   ["Water"],             [("Cloyster", "stone", "Water Stone")]),
    (91, "Cloyster",   ["Water", "Ice"],      []),
    (92, "Gastly",     ["Ghost", "Poison"],   [("Haunter", "level", 25)]),
    (93, "Haunter",    ["Ghost", "Poison"],   [("Gengar", "trade", "")]),
    (94, "Gengar",     ["Ghost", "Poison"],   []),
    (95, "Onix",       ["Rock", "Ground"],    []),
    (96, "Drowzee",    ["Psychic"],           [("Hypno", "level", 26)]),
    (97, "Hypno",      ["Psychic"],           []),
    (98, "Krabby",     ["Water"],             [("Kingler", "level", 28)]),
    (99, "Kingler",    ["Water"],             []),
    (100, "Voltorb",   ["Electric"],          [("Electrode", "level", 30)]),
    (101, "Electrode", ["Electric"],          []),
    (102, "Exeggcute", ["Grass", "Psychic"],  [("Exeggutor", "stone", "Leaf Stone")]),
    (103, "Exeggutor", ["Grass", "Psychic"],  []),
    (104, "Cubone",    ["Ground"],            [("Marowak", "level", 28)]),
    (105, "Marowak",   ["Ground"],            []),
    (106, "Hitmonlee", ["Fighting"],          []),
    (107, "Hitmonchan",["Fighting"],          []),
    (108, "Lickitung", ["Normal"],            []),
    (109, "Koffing",   ["Poison"],            [("Weezing", "level", 35)]),
    (110, "Weezing",   ["Poison"],            []),
    (111, "Rhyhorn",   ["Ground", "Rock"],    [("Rhydon", "level", 42)]),
    (112, "Rhydon",    ["Ground", "Rock"],    []),
    (113, "Chansey",   ["Normal"],            []),
    (114, "Tangela",   ["Grass"],             []),
    (115, "Kangaskhan",["Normal"],            []),
    (116, "Horsea",    ["Water"],             [("Seadra", "level", 32)]),
    (117, "Seadra",    ["Water"],             []),
    (118, "Goldeen",   ["Water"],             [("Seaking", "level", 33)]),
    (119, "Seaking",   ["Water"],             []),
    (120, "Staryu",    ["Water"],             [("Starmie", "stone", "Water Stone")]),
    (121, "Starmie",   ["Water", "Psychic"],  []),
    (122, "Mr. Mime",  ["Psychic"],           []),
    (123, "Scyther",   ["Bug", "Flying"],     []),
    (124, "Jynx",      ["Ice", "Psychic"],    []),
    (125, "Electabuzz",["Electric"],          []),
    (126, "Magmar",    ["Fire"],              []),
    (127, "Pinsir",    ["Bug"],               []),
    (128, "Tauros",    ["Normal"],            []),
    (129, "Magikarp",  ["Water"],             [("Gyarados", "level", 20)]),
    (130, "Gyarados",  ["Water", "Flying"],   []),
    (131, "Lapras",    ["Water", "Ice"],      []),
    (132, "Ditto",     ["Normal"],            []),
    (133, "Eevee",     ["Normal"],            [("Vaporeon", "stone", "Water Stone"),
                                               ("Jolteon", "stone", "Thunder Stone"),
                                               ("Flareon", "stone", "Fire Stone")]),
    (134, "Vaporeon",  ["Water"],             []),
    (135, "Jolteon",   ["Electric"],          []),
    (136, "Flareon",   ["Fire"],              []),
    (137, "Porygon",   ["Normal"],            []),
    (138, "Omanyte",   ["Rock", "Water"],     [("Omastar", "level", 40)]),
    (139, "Omastar",   ["Rock", "Water"],     []),
    (140, "Kabuto",    ["Rock", "Water"],     [("Kabutops", "level", 40)]),
    (141, "Kabutops",  ["Rock", "Water"],     []),
    (142, "Aerodactyl",["Rock", "Flying"],    []),
    (143, "Snorlax",   ["Normal"],            []),
    (144, "Articuno",  ["Ice", "Flying"],     []),
    (145, "Zapdos",    ["Electric", "Flying"],[]),
    (146, "Moltres",   ["Fire", "Flying"],    []),
    (147, "Dratini",   ["Dragon"],            [("Dragonair", "level", 30)]),
    (148, "Dragonair", ["Dragon"],            [("Dragonite", "level", 55)]),
    (149, "Dragonite", ["Dragon", "Flying"],  []),
    (150, "Mewtwo",    ["Psychic"],           []),
    (151, "Mew",       ["Psychic"],           []),
]

LEGENDARIES = {"Articuno", "Zapdos", "Moltres", "Mewtwo", "Mew"}

# Pokedex category / species (the "____ Pokemon" label), sourced from PokeAPI.
CATEGORY = {
    1: "Seed Pokemon", 2: "Seed Pokemon", 3: "Seed Pokemon", 4: "Lizard Pokemon",
    5: "Flame Pokemon", 6: "Flame Pokemon", 7: "Tiny Turtle Pokemon", 8: "Turtle Pokemon",
    9: "Shellfish Pokemon", 10: "Worm Pokemon", 11: "Cocoon Pokemon", 12: "Butterfly Pokemon",
    13: "Hairy Bug Pokemon", 14: "Cocoon Pokemon", 15: "Poison Bee Pokemon", 16: "Tiny Bird Pokemon",
    17: "Bird Pokemon", 18: "Bird Pokemon", 19: "Mouse Pokemon", 20: "Mouse Pokemon",
    21: "Tiny Bird Pokemon", 22: "Beak Pokemon", 23: "Snake Pokemon", 24: "Cobra Pokemon",
    25: "Mouse Pokemon", 26: "Mouse Pokemon", 27: "Mouse Pokemon", 28: "Mouse Pokemon",
    29: "Poison Pin Pokemon", 30: "Poison Pin Pokemon", 31: "Drill Pokemon", 32: "Poison Pin Pokemon",
    33: "Poison Pin Pokemon", 34: "Drill Pokemon", 35: "Fairy Pokemon", 36: "Fairy Pokemon",
    37: "Fox Pokemon", 38: "Fox Pokemon", 39: "Balloon Pokemon", 40: "Balloon Pokemon",
    41: "Bat Pokemon", 42: "Bat Pokemon", 43: "Weed Pokemon", 44: "Weed Pokemon",
    45: "Flower Pokemon", 46: "Mushroom Pokemon", 47: "Mushroom Pokemon", 48: "Insect Pokemon",
    49: "Poison Moth Pokemon", 50: "Mole Pokemon", 51: "Mole Pokemon", 52: "Scratch Cat Pokemon",
    53: "Classy Cat Pokemon", 54: "Duck Pokemon", 55: "Duck Pokemon", 56: "Pig Monkey Pokemon",
    57: "Pig Monkey Pokemon", 58: "Puppy Pokemon", 59: "Legendary Pokemon", 60: "Tadpole Pokemon",
    61: "Tadpole Pokemon", 62: "Tadpole Pokemon", 63: "Psi Pokemon", 64: "Psi Pokemon",
    65: "Psi Pokemon", 66: "Superpower Pokemon", 67: "Superpower Pokemon", 68: "Superpower Pokemon",
    69: "Flower Pokemon", 70: "Flycatcher Pokemon", 71: "Flycatcher Pokemon", 72: "Jellyfish Pokemon",
    73: "Jellyfish Pokemon", 74: "Rock Pokemon", 75: "Rock Pokemon", 76: "Megaton Pokemon",
    77: "Fire Horse Pokemon", 78: "Fire Horse Pokemon", 79: "Dopey Pokemon", 80: "Hermit Crab Pokemon",
    81: "Magnet Pokemon", 82: "Magnet Pokemon", 83: "Wild Duck Pokemon", 84: "Twin Bird Pokemon",
    85: "Triple Bird Pokemon", 86: "Sea Lion Pokemon", 87: "Sea Lion Pokemon", 88: "Sludge Pokemon",
    89: "Sludge Pokemon", 90: "Bivalve Pokemon", 91: "Bivalve Pokemon", 92: "Gas Pokemon",
    93: "Gas Pokemon", 94: "Shadow Pokemon", 95: "Rock Snake Pokemon", 96: "Hypnosis Pokemon",
    97: "Hypnosis Pokemon", 98: "River Crab Pokemon", 99: "Pincer Pokemon", 100: "Ball Pokemon",
    101: "Ball Pokemon", 102: "Egg Pokemon", 103: "Coconut Pokemon", 104: "Lonely Pokemon",
    105: "Bone Keeper Pokemon", 106: "Kicking Pokemon", 107: "Punching Pokemon", 108: "Licking Pokemon",
    109: "Poison Gas Pokemon", 110: "Poison Gas Pokemon", 111: "Spikes Pokemon", 112: "Drill Pokemon",
    113: "Egg Pokemon", 114: "Vine Pokemon", 115: "Parent Pokemon", 116: "Dragon Pokemon",
    117: "Dragon Pokemon", 118: "Goldfish Pokemon", 119: "Goldfish Pokemon", 120: "Star Shape Pokemon",
    121: "Mysterious Pokemon", 122: "Barrier Pokemon", 123: "Mantis Pokemon", 124: "Human Shape Pokemon",
    125: "Electric Pokemon", 126: "Spitfire Pokemon", 127: "Stag Beetle Pokemon", 128: "Wild Bull Pokemon",
    129: "Fish Pokemon", 130: "Atrocious Pokemon", 131: "Transport Pokemon", 132: "Transform Pokemon",
    133: "Evolution Pokemon", 134: "Bubble Jet Pokemon", 135: "Lightning Pokemon", 136: "Flame Pokemon",
    137: "Virtual Pokemon", 138: "Spiral Pokemon", 139: "Spiral Pokemon", 140: "Shellfish Pokemon",
    141: "Shellfish Pokemon", 142: "Fossil Pokemon", 143: "Sleeping Pokemon", 144: "Freeze Pokemon",
    145: "Electric Pokemon", 146: "Flame Pokemon", 147: "Dragon Pokemon", 148: "Dragon Pokemon",
    149: "Dragon Pokemon", 150: "Genetic Pokemon", 151: "New Species Pokemon",
}

# Type immunities (Gen 1): (attacker_type, defender_type) -> no effect (0x).
IMMUNITIES = [
    ("Normal", "Ghost"),
    ("Fighting", "Ghost"),
    ("Ghost", "Normal"),
    ("Ground", "Flying"),
    ("Electric", "Ground"),
    ("Ghost", "Psychic"),   # Gen-1 RBY bug: Ghost moves do 0 to Psychic (intended 2x)
]

# ── Kanto world knowledge ────────────────────────────────────────────────
# Gym: (city, leader, type, badge)
# Gym: (order, city, leader, type, badge, ace Pokemon)
GYMS = [
    (1, "Pewter City",     "Brock",     "Rock",     "Boulder Badge", "Onix"),
    (2, "Cerulean City",   "Misty",     "Water",    "Cascade Badge", "Starmie"),
    (3, "Vermilion City",  "Lt. Surge", "Electric", "Thunder Badge", "Raichu"),
    (4, "Celadon City",    "Erika",     "Grass",    "Rainbow Badge", "Vileplume"),
    (5, "Fuchsia City",    "Koga",      "Poison",   "Soul Badge",    "Weezing"),
    (6, "Saffron City",    "Sabrina",   "Psychic",  "Marsh Badge",   "Alakazam"),
    (7, "Cinnabar Island", "Blaine",    "Fire",     "Volcano Badge", "Arcanine"),
    (8, "Viridian City",   "Giovanni",  "Ground",   "Earth Badge",   "Rhydon"),
]
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth",
           5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
# Badge -> HM field move it enables (Gen 1). The other three badges grant
# obedience levels rather than gating a field move.
BADGE_HM = {
    "Boulder Badge": ("Flash", "HM05"),
    "Cascade Badge": ("Cut", "HM01"),
    "Thunder Badge": ("Fly", "HM02"),
    "Rainbow Badge": ("Strength", "HM04"),
    "Soul Badge":    ("Surf", "HM03"),
}

# Elite Four: (name, type)
ELITE_FOUR = [
    ("Lorelei", "Ice"),
    ("Bruno",   "Fighting"),
    ("Agatha",  "Ghost"),
    ("Lance",   "Dragon"),
]

TYPES = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting",
         "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon"]

# Gen-1 (RBY) type chart. Three Gen-1-specific facts, verified 2026-07-10:
#  (a) Bug <-> Poison are MUTUALLY super effective (2x each way). Modern gens
#      changed both, but in RBY Bug beats Poison and Poison beats Bug — this is
#      why Bulbasaur (Grass/Poison) takes 4x from Bug. (STRONG_VS + WEAK_TO below.)
#  (b) Ghost -> Psychic is 0x in RBY (the famous programming bug), NOT the
#      intended 2x. Modeled as an immunity + omitted from Ghost's STRONG_VS so the
#      data matches the lore passage about the Psychic-type glitch.
#  (c) STRONG_VS and WEAK_TO are exact transposes (A super-effective vs B  <=>
#      B weak to A); RESISTS/WEAK_TO/IMMUNITIES never double-assign a cell.
# Offensive: type -> super effective vs [...]
STRONG_VS = {
    "Normal":   [],
    "Fire":     ["Grass", "Ice", "Bug"],
    "Water":    ["Fire", "Ground", "Rock"],
    "Grass":    ["Water", "Ground", "Rock"],
    "Electric": ["Water", "Flying"],
    "Ice":      ["Grass", "Ground", "Flying", "Dragon"],
    "Fighting": ["Normal", "Ice", "Rock"],
    "Poison":   ["Grass", "Bug"],                        # Gen-1: Poison 2x vs Bug
    "Ground":   ["Fire", "Electric", "Poison", "Rock"],
    "Flying":   ["Grass", "Fighting", "Bug"],
    "Psychic":  ["Fighting", "Poison"],
    "Bug":      ["Grass", "Psychic", "Poison"],          # Gen-1: Bug 2x vs Poison
    "Rock":     ["Fire", "Ice", "Flying", "Bug"],
    "Ghost":    ["Ghost"],                               # Gen-1 bug: Ghost->Psychic is 0x, see IMMUNITIES
    "Dragon":   ["Dragon"],
}
# Defensive weaknesses. type -> takes super effective (2x) damage from [...]
WEAK_TO = {
    "Normal":   ["Fighting"],
    "Fire":     ["Water", "Ground", "Rock"],
    "Water":    ["Electric", "Grass"],
    "Grass":    ["Fire", "Ice", "Poison", "Flying", "Bug"],
    "Electric": ["Ground"],
    "Ice":      ["Fire", "Fighting", "Rock"],
    "Fighting": ["Flying", "Psychic"],
    "Poison":   ["Ground", "Psychic", "Bug"],            # Gen-1: Bug 2x vs Poison
    "Ground":   ["Water", "Grass", "Ice"],
    "Flying":   ["Electric", "Ice", "Rock"],
    "Psychic":  ["Bug"],
    "Bug":      ["Fire", "Flying", "Rock", "Poison"],     # Gen-1: Poison 2x vs Bug
    "Rock":     ["Water", "Grass", "Fighting", "Ground"],
    "Ghost":    ["Ghost"],
    "Dragon":   ["Ice", "Dragon"],
}
# Defensive resistances. type -> takes not-very-effective (0.5x) damage from [...]
# Verified Gen-1 (three independent reconstructions + reconciliation, 2026-07-10).
RESISTS = {
    "Normal":   [],
    "Fire":     ["Fire", "Grass", "Ice", "Bug"],
    "Water":    ["Fire", "Water", "Ice"],
    "Grass":    ["Water", "Electric", "Grass", "Ground"],
    "Electric": ["Electric", "Flying"],
    "Ice":      ["Ice"],
    "Fighting": ["Bug", "Rock"],
    "Poison":   ["Grass", "Fighting", "Poison"],
    "Ground":   ["Poison", "Rock"],
    "Flying":   ["Grass", "Fighting", "Bug"],
    "Psychic":  ["Fighting", "Psychic"],
    "Bug":      ["Grass", "Fighting", "Ground"],
    "Rock":     ["Normal", "Fire", "Poison", "Flying"],
    "Ghost":    ["Poison", "Bug"],
    "Dragon":   ["Fire", "Water", "Electric", "Grass"],
}

# Locations: (name, fact sentence)
LOCATIONS = [
    ("Pallet Town", "Pallet Town is the player's hometown and home to Professor Oak's laboratory."),
    ("Viridian City", "Viridian City has the eighth gym, secretly led by Giovanni."),
    ("Pewter City", "Pewter City has Brock's Rock-type gym and a museum of science."),
    ("Cerulean City", "Cerulean City has Misty's Water-type gym."),
    ("Vermilion City", "Vermilion City is a port where the S.S. Anne docks, home to Lt. Surge's gym."),
    ("Lavender Town", "Lavender Town is home to the Pokemon Tower, a memorial for departed Pokemon."),
    ("Celadon City", "Celadon City is the largest city, with a department store and the Game Corner."),
    ("Saffron City", "Saffron City holds Silph Co. and Sabrina's Psychic-type gym."),
    ("Fuchsia City", "Fuchsia City has Koga's gym and the Safari Zone."),
    ("Cinnabar Island", "Cinnabar Island has Blaine's gym, the Pokemon Mansion, and a fossil lab."),
    ("Indigo Plateau", "Indigo Plateau is where you challenge the Elite Four and the Champion."),
    ("Mt. Moon", "Mt. Moon is a cave between Pewter City and Cerulean City where Clefairy and Moon Stones are found."),
    ("Viridian Forest", "Viridian Forest is a maze-like wood full of Bug Pokemon between Viridian City and Pewter City."),
    ("Rock Tunnel", "Rock Tunnel is a pitch-dark cave that needs Flash, connecting Cerulean City to Lavender Town."),
    ("Pokemon Tower", "The Pokemon Tower in Lavender Town is haunted until you get the Silph Scope."),
    ("Safari Zone", "The Safari Zone in Fuchsia City uses Safari Balls to catch rare Pokemon."),
    ("Seafoam Islands", "The Seafoam Islands are sea caves where the legendary bird Articuno lives."),
    ("Power Plant", "The Power Plant is an abandoned building where the legendary bird Zapdos lives."),
    ("Victory Road", "Victory Road is the final cave before the Indigo Plateau."),
    ("Cerulean Cave", "Cerulean Cave is where Mewtwo can be caught after you become Champion."),
    ("Diglett's Cave", "Diglett's Cave is a tunnel full of Diglett near Vermilion City."),
    ("Silph Co.", "Silph Co. is a company in Saffron City taken over by Team Rocket."),
    ("S.S. Anne", "The S.S. Anne is a cruise ship at Vermilion City where you receive HM01 Cut."),
]

# Characters: (name, fact)
CHARACTERS = [
    ("Professor Oak", "Professor Oak is the Pokemon researcher in Pallet Town who gives you your first Pokemon and the Pokedex."),
    ("Giovanni", "Giovanni is the boss of Team Rocket and the hidden Viridian City Gym Leader."),
    ("Team Rocket", "Team Rocket is a criminal gang that steals Pokemon, led by Giovanni."),
    ("Bill", "Bill is a Pokemaniac near Cerulean City who invented the Pokemon Storage System."),
    ("Mr. Fuji", "Mr. Fuji is an elder in Lavender Town who cares for orphaned Pokemon."),
    ("the rival", "Your rival is Professor Oak's grandson, who chases the Champion title alongside you."),
]

# Items: (name, fact)
ITEMS = [
    ("Poke Ball", "A Poke Ball is the basic tool for catching wild Pokemon."),
    ("Great Ball", "A Great Ball catches Pokemon more reliably than a Poke Ball."),
    ("Ultra Ball", "An Ultra Ball has a high catch rate for tough Pokemon."),
    ("Master Ball", "The Master Ball catches any Pokemon without fail. There is only one, found in Silph Co."),
    ("Potion", "A Potion restores a small amount of a Pokemon's HP."),
    ("Revive", "A Revive brings a fainted Pokemon back with half its HP."),
    ("Rare Candy", "A Rare Candy instantly raises a Pokemon's level by one."),
    ("Antidote", "An Antidote cures a poisoned Pokemon."),
    ("Full Heal", "A Full Heal cures any status condition."),
    ("Escape Rope", "An Escape Rope warps you out of a cave or dungeon."),
    ("Repel", "A Repel keeps weak wild Pokemon away while you walk."),
    # Evolution stones are handled by the derived stone-evolution section below.
    ("Poke Flute", "The Poke Flute wakes sleeping Pokemon, like the Snorlax blocking the road."),
    ("Silph Scope", "The Silph Scope lets you identify the ghosts in the Pokemon Tower."),
    ("Bicycle", "The Bicycle lets you travel much faster than walking."),
    ("Town Map", "The Town Map shows the Kanto region and where you are."),
    # More healing items
    ("Super Potion", "A Super Potion restores more HP than a Potion."),
    ("Hyper Potion", "A Hyper Potion restores a large amount of HP."),
    ("Max Potion", "A Max Potion fully restores a Pokemon's HP."),
    ("Full Restore", "A Full Restore fully heals HP and cures any status condition."),
    ("Max Revive", "A Max Revive revives a fainted Pokemon with full HP."),
    ("Ether", "An Ether restores the PP of one of a Pokemon's moves."),
    ("Elixir", "An Elixir restores the PP of all of a Pokemon's moves."),
    ("Paralyze Heal", "A Paralyze Heal cures a paralyzed Pokemon."),
    ("Awakening", "An Awakening wakes up a sleeping Pokemon."),
    ("Burn Heal", "A Burn Heal cures a burned Pokemon."),
    ("Ice Heal", "An Ice Heal thaws out a frozen Pokemon."),
    # Poke Balls / capture
    ("Safari Ball", "A Safari Ball is a special ball used only inside the Safari Zone."),
    # Key items
    ("Old Rod", "The Old Rod is a fishing rod used to catch Magikarp and other weak water Pokemon."),
    ("Good Rod", "The Good Rod is a better fishing rod for catching water Pokemon."),
    ("Super Rod", "The Super Rod is the best fishing rod, catching the strongest water Pokemon."),
    ("Itemfinder", "The Itemfinder beeps when there is a hidden item nearby."),
    ("Card Key", "The Card Key opens the locked doors inside Silph Co."),
    ("Lift Key", "The Lift Key operates the elevator in the Team Rocket Hideout."),
    ("Secret Key", "The Secret Key unlocks Blaine's gym on Cinnabar Island."),
    ("S.S. Ticket", "The S.S. Ticket lets you board the S.S. Anne at Vermilion City."),
    ("Bike Voucher", "The Bike Voucher can be exchanged for a free Bicycle in Cerulean City."),
    ("Gold Teeth", "The Gold Teeth are returned to the Safari Zone warden for HM04 Strength."),
    ("Poke Doll", "A Poke Doll lets you escape a wild battle and can drive off the Marowak ghost."),
    ("Exp. All", "The Exp. All shares battle experience with every Pokemon in your party."),
    ("Coin Case", "The Coin Case holds coins won at the Celadon Game Corner."),
    ("Oak's Parcel", "Oak's Parcel is picked up at the Poke Mart and delivered to Professor Oak."),
    ("Nugget", "A Nugget is a valuable item you can sell for a lot of money."),
    # Fossils
    ("Helix Fossil", "The Helix Fossil can be revived into Omanyte at the Cinnabar lab."),
    ("Dome Fossil", "The Dome Fossil can be revived into Kabuto at the Cinnabar lab."),
    ("Old Amber", "The Old Amber can be revived into Aerodactyl at the Cinnabar lab."),
]

# HMs (Gen 1 has five): (label, move, fact)
HMS = [
    ("HM01", "Cut", "HM01 teaches Cut, which chops down small trees blocking the path."),
    ("HM02", "Fly", "HM02 teaches Fly, which instantly takes you to any town you have visited."),
    ("HM03", "Surf", "HM03 teaches Surf, which lets a Pokemon carry you across water."),
    ("HM04", "Strength", "HM04 teaches Strength, which lets a Pokemon push heavy boulders."),
    ("HM05", "Flash", "HM05 teaches Flash, which lights up dark caves like Rock Tunnel."),
]

# Mechanics / general knowledge: (list of question phrasings, answer)
MECHANICS = [
    (["How many Pokemon are there?", "How many Pokemon are in Kanto?",
      "How many original Pokemon are there?"],
     "There are 151 Pokemon in the original Kanto Pokedex."),
    (["What is the Pokedex?", "What does the Pokedex do?"],
     "The Pokedex is a digital encyclopedia that records every Pokemon you see and catch."),
    (["How do I catch a Pokemon?", "How do you catch Pokemon?"],
     "Weaken a wild Pokemon in battle, then throw a Poke Ball to catch it."),
    (["How do Pokemon evolve?", "How does evolution work?", "How do I evolve a Pokemon?"],
     "Pokemon evolve by leveling up, using an evolution stone, or being traded."),
    (["How do I heal my Pokemon?", "How do I restore my Pokemon's health?",
      "Where do I heal my Pokemon?"],
     "Visit a Pokemon Center to heal all your Pokemon for free."),
    (["How do I revive a fainted Pokemon?", "How do I bring back a fainted Pokemon?"],
     "Use a Revive item, or heal at a Pokemon Center, to restore a fainted Pokemon."),
    (["What does poison do?", "What is the poison status?"],
     "A poisoned Pokemon loses some HP every turn."),
    (["What does paralysis do?", "What is paralysis?"],
     "A paralyzed Pokemon is slower and may be unable to move."),
    (["What does sleep do?", "What is the sleep status?"],
     "A sleeping Pokemon cannot act for several turns."),
    (["What does a burn do?", "What is the burn status?"],
     "A burned Pokemon loses HP each turn and deals weaker physical attacks."),
    (["What does freeze do?", "What is the freeze status?"],
     "A frozen Pokemon cannot move until it thaws out."),
    (["What happens when a Pokemon faints?", "What is fainting?"],
     "A Pokemon faints when its HP reaches zero and cannot battle until healed."),
    (["What is PP?", "What does PP mean?"],
     "PP is the number of times a move can be used before it runs out."),
    (["What is a Pokemon Center?", "What do Pokemon Centers do?"],
     "A Pokemon Center heals all your Pokemon for free."),
    (["What is the Poke Mart?", "Where do I buy items?"],
     "The Poke Mart is a shop that sells items like Poke Balls and Potions."),
    (["What are HMs?", "What is an HM?"],
     "HMs teach lasting moves like Cut, Fly, Surf, Strength, and Flash, used in and out of battle."),
    (["What are TMs?", "What is a TM?"],
     "A TM teaches a Pokemon a new move and is used up once."),
    (["Who are the Kanto starters?", "What are the starter Pokemon?",
      "What Pokemon can I start with?"],
     "The Kanto starters are Bulbasaur, Charmander, and Squirtle, given by Professor Oak."),
    (["What are the legendary birds?", "Name the legendary birds."],
     "The legendary birds of Kanto are Articuno, Zapdos, and Moltres."),
    (["What is the strongest Pokemon?", "Where is Mewtwo found?"],
     "Mewtwo is a powerful Psychic-type legendary made from Mew's DNA, found in Cerulean Cave."),
    (["What are the fossils?", "How do I get fossil Pokemon?"],
     "The Helix Fossil revives into Omanyte, the Dome Fossil into Kabuto, and Old Amber into Aerodactyl."),
    (["How many gyms are there?", "How many badges can I get?"],
     "There are eight gyms in Kanto, each giving a badge."),
    (["Who are the Elite Four?", "What is the Elite Four?"],
     "The Elite Four are Lorelei, Bruno, Agatha, and Lance, the final challenge before the Champion."),
    (["What are vitamins?", "What do Protein and Iron do?", "What does a Carbos do?"],
     "Vitamins like HP Up, Protein, Iron, Calcium, and Carbos permanently boost a Pokemon's stats."),
    (["What does a PP Up do?", "What is a PP Up?"],
     "A PP Up permanently raises the maximum PP of one of a Pokemon's moves."),
    (["What are X items?", "What does X Attack do?", "What is X Speed?"],
     "X items like X Attack, X Defend, and X Speed temporarily raise a Pokemon's stats in battle."),
    (["What do vending machine drinks do?", "What does Fresh Water do?"],
     "Fresh Water, Soda Pop, and Lemonade are drinks from the Celadon vending machines that restore HP."),
    (["Where is the Game Corner?", "What is the Game Corner?"],
     "The Game Corner is a slot-machine arcade in Celadon City that hides a Team Rocket base."),
    (["How do I fish?", "How do I catch water Pokemon?"],
     "Use a fishing rod like the Old Rod, Good Rod, or Super Rod next to water to hook Pokemon."),
    (["How do I revive a fossil?", "Where do I revive fossils?"],
     "Take a fossil to the lab on Cinnabar Island to revive it into a Pokemon."),
]

# What this model can answer — meta help (trained into the .bin like HardwareOne).
CAPABILITIES = [
    (["What can you do?", "What can you answer?", "What are your capabilities?",
      "What are you capable of?"],
     "I answer Kanto Pokemon facts: types, Pokedex numbers, entries, evolutions, catch spots, gyms, and type matchups."),
    (["What questions can I ask?", "What can I ask you?", "What should I ask?",
      "Help", "What can you help with?"],
     "Try What type is Charizard, What is Pikachu's Pokedex entry, or Where can you catch Abra."),
    (["What do you know about?", "What topics do you cover?", "What is your domain?"],
     "I know the original 151 Kanto Pokemon plus gyms, towns, items, evolution stones, and battle types."),
    (["Who are you?", "What is this model?", "What are you?"],
     "I am a Kanto Pokemon Master guide for Gen-1 Pokemon Red, Blue, and Yellow facts."),
]

# Lore/background passages. Each is (open-ended question phrasings, passage).
# The passage is trained BOTH as bare prose (pure LM exposure of the text) AND
# as the answer to every phrasing — so "tell me about Kanto" maps onto the
# canonical paragraph instead of the model free-associating. Passages are kept
# SHORT and factually dense: a tiny model memorizes tight, self-contained text
# far better than long flowery prose, and short passages drift less.
LORE = [
    (["Tell me about Kanto.", "Describe the Kanto region.", "What is the Kanto region?",
      "Tell me about the Kanto region.", "Where does the game take place?"],
     "The Kanto region is the setting of Pokemon Red, Blue, and Yellow. It has nine cities, a network of routes, and several caves. Trainers travel from Pallet Town to collect eight gym badges and challenge the Elite Four."),
    (["What is a Pokemon trainer's goal?", "What does a Pokemon trainer do?",
      "Tell me about being a trainer.", "What is the point of the game?"],
     "A Pokemon trainer's goal is to catch and train Pokemon, defeat the eight gym leaders, and become the Champion. Along the way the trainer fills the Pokedex, which records all 151 Kanto Pokemon."),
    (["Explain type matchups.", "How do type matchups work?", "Tell me about type effectiveness.",
      "How do types work in battle?"],
     "Type matchups decide battles. Water beats Fire, Fire beats Grass, and Grass beats Water in a rock-paper-scissors triangle. Electric beats Water and Flying, while Ground is immune to Electric attacks."),
    (["Tell me about evolution stones.", "How do evolution stones work?", "What are evolution stones?",
      "Which Pokemon use evolution stones?"],
     "Evolution stones let certain Pokemon evolve instantly. The Fire, Water, Thunder, Leaf, and Moon Stones each evolve a specific set of Pokemon. Eevee can become Vaporeon, Jolteon, or Flareon depending on the stone used."),
    (["Which Pokemon evolve by trading?", "Tell me about trade evolution.", "How does trade evolution work?",
      "What Pokemon evolve when traded?"],
     "Some Pokemon only evolve when traded with another player, including Kadabra into Alakazam, Machoke into Machamp, Graveler into Golem, and Haunter into Gengar."),
    (["Tell me about Team Rocket.", "Who is Team Rocket?", "Describe Team Rocket.", "What is Team Rocket?"],
     "Team Rocket is a criminal organization led by Giovanni. They appear at Mt. Moon, the Game Corner hideout, the Pokemon Tower, and Silph Co., stealing Pokemon for profit until the player drives them off."),
    (["Tell me about the legendary birds.", "Who are the legendary birds?", "Describe the legendary birds.",
      "What are the legendary birds of Kanto?"],
     "The legendary birds are Articuno, Zapdos, and Moltres. Articuno lives in the Seafoam Islands, Zapdos in the Power Plant, and Moltres on Victory Road or Mt. Ember depending on the game."),
    (["Tell me about Mewtwo.", "Who is Mewtwo?", "Tell me about Mew.", "What is the strongest Pokemon?"],
     "Mew is the 151st Pokemon, a rare mythical Psychic-type. Mewtwo was created by scientists from Mew's genes and is the strongest Pokemon in Kanto, waiting in Cerulean Cave."),
    (["Why were Psychic types so strong?", "Tell me about the Psychic type glitch.",
      "What was the Ghost Psychic bug?"],
     "In the original Red and Blue games, a programming quirk made Ghost-type moves do nothing to Psychic types, even though Ghost was meant to be super effective against them. Psychic types were dominant as a result."),
    (["Tell me about the gym leaders.", "How many gyms are in Kanto?", "What are the Kanto gym leaders?",
      "Tell me about the Kanto gyms."],
     "The eight Kanto gym leaders are Brock, Misty, Lt. Surge, Erika, Koga, Sabrina, Blaine, and Giovanni. Defeating a leader earns a badge, and all eight badges are needed to enter the Pokemon League."),
    (["Tell me about the Elite Four.", "Who are the Elite Four?", "What is the Pokemon League?",
      "How do you become Champion?"],
     "The Pokemon League is guarded by the Elite Four: Lorelei, Bruno, Agatha, and Lance. A trainer must beat all four in a row and then defeat the Champion to become the new Champion of Kanto."),
    (["Tell me about Poke Balls.", "How do Poke Balls work?", "What kinds of Poke Balls are there?",
      "How do you catch a Pokemon?"],
     "Poke Balls are used to catch wild Pokemon. The basic Poke Ball is weakest, the Great Ball and Ultra Ball catch better, and the Master Ball never fails. A weakened or status-afflicted Pokemon is easier to catch."),
    (["Who is Professor Oak?", "Tell me about Professor Oak.", "What are the starter Pokemon?",
      "How do you start your journey?"],
     "Professor Oak is the Pokemon researcher of Pallet Town. He gives new trainers a first partner, either Bulbasaur, Charmander, or Squirtle, along with a Pokedex to fill while traveling Kanto."),
    (["Tell me about the Safari Zone.", "What is the Safari Zone?", "How does the Safari Zone work?"],
     "The Safari Zone is a special reserve where trainers use Safari Balls and can throw bait or rocks instead of battling. It holds Pokemon found nowhere else, and each visit is limited to a fixed number of steps."),
]

# ── Phrasing templates ───────────────────────────────────────────────────
TYPE_Q = [
    "What type is {name}?",
    "What type of Pokemon is {name}?",
    "{name} is what type?",
    "Tell me the type of {name}.",
    "What's {name}'s type?",
    "What typing does {name} have?",
    "Is {name} a fire type?",
    "What element is {name}?",
]
NUM_Q = [
    "What number is {name}?",
    "What is {name}'s Pokedex number?",
    "Where is {name} in the Pokedex?",
    "{name} is which Pokedex number?",
    "What is the Pokedex number of {name}?",
    "What's {name}'s number?",
    "What dex number is {name}?",
]
INTO_Q = [
    "What does {name} evolve into?",
    "What is {name}'s evolution?",
    "Does {name} evolve?",
    "{name} evolves into what?",
    "What is the evolution of {name}?",
    "What does {name} become?",
    "What's the next evolution of {name}?",
    "Can {name} evolve?",
]
FROM_Q = [
    "What does {name} evolve from?",
    "What is {name}'s pre-evolution?",
    "{name} evolves from what?",
    "What was {name} before?",
    "What does {name} come from?",
    "What is {name} evolved from?",
]
ABOUT_Q = [
    "Tell me about {name}.",
    "What is {name}?",
    "Describe {name}.",
    "Who is {name}?",
    "Tell me about the Pokemon {name}.",
    "Give me info on {name}.",
    # casual / informal
    "What's {name}?",
    "{name}?",
    "info on {name}",
    "tell me about {name}",
    "whats the deal with {name}",
]
DEX_ENTRY_Q = [
    "What is {name}'s Pokedex entry?",
    "What does the Pokedex say about {name}?",
    "Read {name}'s Pokedex entry.",
    "What is the Pokedex entry for {name}?",
]
CATCH_Q = [
    "Where can you catch {name} in Kanto?",
    "Where is {name} found in Kanto?",
    "Where do you find {name}?",
    "How do you catch {name} in Kanto?",
]
NUM_TO_NAME_Q = [
    "What Pokemon is number {num}?",
    "Who is Pokedex number {num}?",
    "What is dex number {num}?",
    "Which Pokemon is #{num}?",
]
# Reverse type lookup — "list water pokemon", "which pokemon are fire type".
TYPE_LIST_Q = [
    "List all {t} type Pokemon.",
    "List {t} Pokemon.",
    "Which Pokemon are {t} type?",
    "Which Kanto Pokemon are {t} type?",
    "Name the {t} type Pokemon.",
    "What Pokemon are {t} type?",
    "Show me the {t} Pokemon.",
    "Give me every {t} type Pokemon.",
    "What {t} type Pokemon are there?",
    "Are there any {t} type Pokemon?",
    "How many {t} type Pokemon are there?",  # answer leads with the count
    "Tell me the {t} type Pokemon.",
]


def type_phrase(types):
    return "/".join(types)


def into_answer(name, into):
    if not into:
        return f"{name} does not evolve."
    parts = []
    for to, method, detail in into:
        if method == "level":
            parts.append(f"{to} at level {detail}")
        elif method == "stone":
            parts.append(f"{to} with a {detail}")
        elif method == "trade":
            parts.append(f"{to} when traded")
    return f"{name} evolves into " + ", ".join(parts) + "."


def evolve_how_answer(name, into):
    """Procedural 'how to evolve' answer (the action), derived from the table."""
    if not into:
        return f"{name} does not evolve."
    if len(into) > 1:  # Eevee — multiple stone evolutions
        stones = ", ".join(d for _, _, d in into[:-1]) + ", or " + into[-1][2]
        tos = ", ".join(t for t, _, _ in into[:-1]) + ", or " + into[-1][0]
        return f"Use a {stones} on {name} to get {tos}."
    to, method, detail = into[0]
    if method == "level":
        return f"Level {name} up to level {detail} to evolve it into {to}."
    if method == "stone":
        return f"Use a {detail} on {name} to evolve it into {to}."
    return f"Trade {name} to evolve it into {to}."


def stone_evolutions():
    """Return {stone_name: [(from, to), ...]} derived from the table."""
    m = {}
    for _num, name, _types, into in POKEMON:
        for to, method, detail in into:
            if method == "stone":
                m.setdefault(detail, []).append((name, to))
    return m


def trade_evolvers():
    """Return [(from, to), ...] for trade evolutions, derived from the table."""
    out = []
    for _num, name, _types, into in POKEMON:
        for to, method, _detail in into:
            if method == "trade":
                out.append((name, to))
    return out


def build_evolves_from():
    """Reverse-map the table so evolves_from is always consistent with into."""
    rev = {}
    for _num, name, _types, into in POKEMON:
        for to, _m, _d in into:
            rev[to] = name
    return rev


def type_to_names():
    """Return {type: [names...]} in Pokedex order for reverse 'list X-type' Q&A."""
    m = {}
    for _num, name, types, _into in POKEMON:
        for t in types:
            m.setdefault(t, []).append(name)
    return m


# ── Type-effectiveness engine (Gen-1) ────────────────────────────────────
# The "reasoning" that gets PRECOMPUTED: a Pokemon's net weaknesses are the
# PRODUCT of each of its type's multipliers, so a 2x can be cancelled to 1x by
# a partner 0.5x (Gyarados Water/Flying is NOT weak to Grass) and two 2x stack
# to 4x (Charizard Fire/Flying takes 4x from Rock). Computing this correctly
# needs the full chart (WEAK_TO + RESISTS + IMMUNITIES), which is why WEAK_TO
# alone gives wrong answers for dual types.
_IMMUNE_SET = set(IMMUNITIES)


def _mult(atk, dtype):
    if (atk, dtype) in _IMMUNE_SET:
        return 0.0
    if atk in WEAK_TO[dtype]:
        return 2.0
    if atk in RESISTS[dtype]:
        return 0.5
    return 1.0


def _eff(atk, dtypes):
    p = 1.0
    for d in dtypes:
        p *= _mult(atk, d)
    return p


def defense_profile(dtypes):
    """(weak4x, weak2x, immune, resist) attacker-type lists for these type(s)."""
    weak4, weak2, immune, resist = [], [], [], []
    for atk in TYPES:
        m = _eff(atk, dtypes)
        if m == 0:
            immune.append(atk)
        elif m >= 4:
            weak4.append(atk)
        elif m > 1:
            weak2.append(atk)
        elif m < 1:
            resist.append(atk)
    return weak4, weak2, immune, resist


def offense_coverage(dtypes):
    """Types a Pokemon's same-type moves are super effective against (deduped)."""
    seen, out = set(), []
    for t in dtypes:
        for x in STRONG_VS[t]:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _name_into():
    """{name: into-list} for evolution-chain walks."""
    return {name: into for _num, name, _types, into in POKEMON}


def final_forms(name, into_map):
    """Terminal evolution(s) of name (Eevee-style branches return all)."""
    outs = []

    def walk(n):
        nxt = into_map.get(n) or []
        if not nxt:
            outs.append(n)
        else:
            for to, _m, _d in nxt:
                walk(to)

    walk(name)
    # dedup preserving order
    seen, uniq = set(), []
    for x in outs:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def base_form(name, evolves_from):
    while name in evolves_from:
        name = evolves_from[name]
    return name


def _list_join(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def about_answer(name, num, types, into, evolves_from):
    t = type_phrase(types)
    legend = " It is a legendary Pokemon." if name in LEGENDARIES else ""
    if into:
        first = into[0]
        if first[1] == "level":
            evo = f" It evolves into {first[0]} at level {first[2]}."
        elif first[1] == "stone":
            evo = f" It evolves into {first[0]} with a {first[2]}."
        else:
            evo = f" It evolves into {first[0]} when traded."
        if len(into) > 1:  # Eevee
            evo = " It can evolve into " + ", ".join(i[0] for i in into) + " using stones."
    elif name in evolves_from:
        evo = f" It evolves from {evolves_from[name]}."
    else:
        evo = ""
    return f"{name} is a {t}-type Pokemon, number {num} in the Kanto Pokedex.{evo}{legend}"


class Corpus:
    def __init__(self):
        self.blocks = []  # each block is a list of lines
        self._q_answer = {}  # question -> its (first) answer; enforces one answer per question
        self.conflicts_dropped = 0

    def qa(self, question, answer):
        # One answer per question, FIRST-write-wins. A model trained on the same
        # question with two different answers learns to hedge/blend between them,
        # so a later block that reuses an existing question with a DIFFERENT answer
        # is dropped (e.g. a lore-passage phrasing that collides with the concise
        # dedicated Q&A emitted earlier — the dedicated answer wins, and the lore
        # is still trained as bare prose).
        prev = self._q_answer.get(question)
        if prev is not None:
            if prev != answer:
                self.conflicts_dropped += 1
            return
        self._q_answer[question] = answer
        self.blocks.append([f"Q: {question}", f"A: {answer}"])

    def qa_variants(self, questions, answer):
        # ONE clean, proper-case example per phrasing. Variety comes from the
        # multiple DISTINCT phrasings in each question list — not from mechanical
        # lowercase/punctuation duplicates, which only bloat the corpus against a
        # capacity-limited model (and a lowercased name fragments into subword
        # tokens the model can't bind to the whole "Pikachu" token anyway).
        # Casual lowercase / unpunctuated input is normalized at inference by the
        # firmware's vocab-aware capitalization.
        for q in questions:
            self.qa(q, answer)

    def prose(self, text):
        self.blocks.append([text])

    def write(self, path, seed=1234):
        rng = random.Random(seed)
        rng.shuffle(self.blocks)
        lines = []
        for block in self.blocks:
            lines.extend(block)
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return len(self.blocks)


def main():
    ap = argparse.ArgumentParser(description="Generate Kanto Pokedex training corpus")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "training_data" / "pokemon_kanto.txt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokens-out", type=Path,
                    default=Path(__file__).parent.parent / "training_data" / "pokemon_special_tokens.txt",
                    help="Where to write the whole-word special-tokens file (the 151 names).")
    args = ap.parse_args()

    evolves_from = build_evolves_from()
    c = Corpus()

    # Per-Pokemon facts
    for num, name, types, into in POKEMON:
        c.qa_variants([q.format(name=name) for q in TYPE_Q],
                      f"{name} is a {type_phrase(types)}-type Pokemon.")
        c.qa_variants([q.format(name=name) for q in NUM_Q],
                      f"{name} is number {num} in the Kanto Pokedex.")
        c.qa_variants([q.format(name=name) for q in INTO_Q],
                      into_answer(name, into))
        c.qa_variants([f"How do I evolve {name}?",
                       f"How does {name} evolve?",
                       f"How do you evolve {name}?"],
                      evolve_how_answer(name, into))
        if name in evolves_from:
            c.qa_variants([q.format(name=name) for q in FROM_Q],
                          f"{name} evolves from {evolves_from[name]}.")
        else:
            c.qa_variants([q.format(name=name) for q in FROM_Q],
                          f"{name} does not evolve from any Pokemon.")
        c.qa_variants([q.format(name=name) for q in ABOUT_Q],
                      about_answer(name, num, types, into, evolves_from))
        c.qa_variants([f"What category is {name}?",
                       f"What species is {name}?",
                       f"What is {name}'s Pokedex category?"],
                      f"{name} is the {CATEGORY[num]}.")
        if num in DEX_ENTRY:
            c.qa_variants([q.format(name=name) for q in DEX_ENTRY_Q], DEX_ENTRY[num])
        if num in CATCH_LOC:
            c.qa_variants([q.format(name=name) for q in CATCH_Q], CATCH_LOC[num])
        c.qa_variants([q.format(num=num) for q in NUM_TO_NAME_Q],
                      f"{name} is number {num} in the Kanto Pokedex.")

    # Gyms
    for order, city, leader, gtype, badge, ace in GYMS:
        c.qa_variants([f"Who is the {city} gym leader?",
                       f"Who leads the {city} gym?",
                       f"Who runs the {city} gym?",
                       f"Who is the gym leader in {city}?"],
                      f"{leader} is the Gym Leader of {city}.")
        c.qa_variants([f"What type does {leader} use?",
                       f"What is {leader}'s specialty?",
                       f"What type is {leader}?",
                       f"What Pokemon does {leader} use?"],
                      f"{leader} uses {gtype}-type Pokemon.")
        c.qa_variants([f"What badge does {leader} give?",
                       f"What badge do you get from {leader}?",
                       f"What badge is in {city}?",
                       f"What badge do you win in {city}?"],
                      f"{leader} gives the {badge}.")
        c.qa_variants([f"Where is {leader}'s gym?",
                       f"What city is {leader} in?",
                       f"What gym is in {city}?"],
                      f"{leader}'s gym is in {city}.")
        c.qa_variants([f"What is {leader}'s strongest Pokemon?",
                       f"What is {leader}'s ace?",
                       f"What Pokemon does {leader} battle with?",
                       f"What is {leader}'s best Pokemon?"],
                      f"{leader}'s strongest Pokemon is {ace}.")
        c.qa_variants([f"Which number gym is {leader}'s?",
                       f"Where does {leader} fall in the gym order?"],
                      f"{leader}'s gym is the {ORDINAL[order]} gym in Kanto.")
        c.qa_variants([f"Who is the {ORDINAL[order]} gym leader?",
                       f"Which leader is the {ORDINAL[order]} gym?"],
                      f"The {ORDINAL[order]} Kanto gym leader is {leader} of {city}.")

    # Gym overview + badge -> HM field moves
    leaders = ", ".join(g[2] for g in GYMS[:-1]) + ", and " + GYMS[-1][2]
    c.qa_variants(["Who are the gym leaders?",
                   "Name the Kanto gym leaders.",
                   "List the gym leaders."],
                  f"The Kanto gym leaders are {leaders}.")
    c.qa_variants(["Which gym is first?", "What is the first Kanto gym?"],
                  "The first Kanto gym is Brock's in Pewter City.")
    c.qa_variants(["Which gym is last?", "What is the final Kanto gym?"],
                  "The last Kanto gym is Giovanni's in Viridian City.")
    for badge, (move, hm) in BADGE_HM.items():
        c.qa_variants([f"What does the {badge} do?",
                       f"What does the {badge} let you do?",
                       f"What does the {badge} enable?"],
                      f"The {badge} lets you use {move} outside of battle.")
        c.qa_variants([f"What badge do I need to use {move}?",
                       f"Which badge enables {move}?"],
                      f"You need the {badge} to use {move} outside of battle.")

    # Elite Four
    for name, etype in ELITE_FOUR:
        c.qa_variants([f"What type does {name} use?",
                       f"What is {name}'s type?",
                       f"What does {name} of the Elite Four specialize in?"],
                      f"{name} of the Elite Four uses {etype}-type Pokemon.")

    # Type chart
    for t in sorted(VALID_TYPES):
        strong = STRONG_VS[t]
        weak = WEAK_TO[t]
        if strong:
            c.qa_variants([f"What is {t} super effective against?",
                           f"What is {t} strong against?",
                           f"What does {t} beat?"],
                          f"{t} moves are super effective against " + ", ".join(strong) + ".")
        c.qa_variants([f"What is {t} weak to?",
                       f"What beats {t} types?",
                       f"What is super effective against {t}?"],
                      f"{t} Pokemon are weak to " + ", ".join(weak) + ".")

    # Reverse type lookup: which Pokemon are of each type (count + examples)
    members = {t: [] for t in VALID_TYPES}
    for num, name, types, _into in POKEMON:
        for t in types:
            members[t].append(name)
    for t in sorted(VALID_TYPES):
        ms = members[t]
        c.qa_variants([f"How many {t}-type Pokemon are there?",
                       f"How many {t} types are in Kanto?"],
                      f"There are {len(ms)} {t}-type Pokemon among the original 151.")
        if len(ms) <= 6:
            listing = ", ".join(ms[:-1]) + ", and " + ms[-1] if len(ms) > 1 else ms[0]
            ans = f"The {t}-type Pokemon are {listing}."
        else:
            ex = ms[:6]
            ans = f"{t}-type Pokemon include " + ", ".join(ex[:-1]) + ", and " + ex[-1] + "."
        c.qa_variants([f"Which Pokemon are {t}-type?",
                       f"Name some {t}-type Pokemon.",
                       f"Give me a {t}-type Pokemon.",
                       f"What are the {t}-type Pokemon?",
                       f"what are {t.lower()} pokemon?",
                       f"Which Pokemon are {t}?",
                       f"List the {t}-type Pokemon.",
                       f"Show me {t}-type Pokemon."],
                      ans)

    # Type immunities (no-effect matchups)
    defender_immune = {}
    for att, dfn in IMMUNITIES:
        c.qa_variants([f"Can {att} moves hit {dfn} Pokemon?",
                       f"Do {att} attacks affect {dfn} types?",
                       f"Are {dfn} Pokemon immune to {att}?"],
                      f"No, {att} moves have no effect on {dfn} Pokemon.")
        defender_immune.setdefault(dfn, []).append(att)
    for dfn, atts in defender_immune.items():
        joined = " and ".join(atts) if len(atts) <= 2 else ", ".join(atts[:-1]) + ", and " + atts[-1]
        c.qa_variants([f"What are {dfn} Pokemon immune to?",
                       f"What moves do not affect {dfn} Pokemon?"],
                      f"{dfn} Pokemon are immune to {joined} moves.")
    # Gen-1 Ghost-vs-Psychic glitch
    c.qa_variants(["Can Ghost moves hit Psychic Pokemon?",
                   "Are Psychic Pokemon immune to Ghost?",
                   "What are Psychic Pokemon immune to?"],
                  "In Red and Blue, a glitch made Ghost moves have no effect on Psychic Pokemon.")

    # Locations
    for name, fact in LOCATIONS:
        c.qa_variants([f"What is {name}?",
                       f"Tell me about {name}.",
                       f"Where is {name}?"],
                      fact)

    # Characters
    for name, fact in CHARACTERS:
        c.qa_variants([f"Who is {name}?",
                       f"Tell me about {name}."],
                      fact)

    # Items
    for name, fact in ITEMS:
        c.qa_variants([f"What is a {name}?",
                       f"What does a {name} do?",
                       f"Tell me about the {name}.",
                       f"What's a {name}?",
                       f"What does the {name} do?",
                       f"How do I use a {name}?"],
                      fact)

    # Item interactions: evolution stones (all derived from the table) ──────
    stones = stone_evolutions()
    for stone, pairs in stones.items():
        listing = ", ".join(f"{frm} into {to}" for frm, to in pairs)
        c.qa_variants([f"What does a {stone} do?",
                       f"What is a {stone}?",
                       f"Which Pokemon does a {stone} evolve?",
                       f"What does a {stone} evolve?",
                       f"What Pokemon use a {stone}?",
                       f"what pokemon uses a {stone.lower()}?",
                       f"What evolves with a {stone}?",
                       f"Which Pokemon need a {stone}?"],
                      f"A {stone} evolves {listing}.")

    # Per-Pokemon item-evolution facts, including correct negatives.
    for _num, name, _types, into in POKEMON:
        if not into:
            continue
        if len(into) > 1:  # Eevee (multiple stones)
            evos = ", ".join(t for t, _, _ in into[:-1]) + ", or " + into[-1][0]
            stns = ", ".join(d for _, _, d in into[:-1]) + ", or " + into[-1][2]
            c.qa_variants([f"What item evolves {name}?",
                           f"What stones does {name} use?",
                           f"Does {name} need an item to evolve?"],
                          f"{name} evolves into {evos} using the {stns}.")
            continue
        to, method, detail = into[0]
        if method == "stone":
            c.qa_variants([f"What item evolves {name}?",
                           f"What stone does {name} need?",
                           f"Does {name} need an item to evolve?"],
                          f"{name} evolves into {to} with a {detail}.")
        elif method == "trade":
            c.qa_variants([f"What item evolves {name}?",
                           f"Does {name} need an item to evolve?"],
                          f"{name} needs no item to evolve. It evolves into {to} when traded.")
        elif method == "level":
            c.qa_variants([f"Does {name} need an item to evolve?",
                           f"Does {name} use a stone to evolve?"],
                          f"No, {name} evolves into {to} by leveling up.")

    # Overview facts about item/trade evolution.
    c.qa_variants(["What are the evolution stones?",
                   "Which stones can evolve Pokemon?",
                   "Name the evolution stones."],
                  "The evolution stones are the Fire, Water, Thunder, Leaf, and Moon Stone.")
    trades = trade_evolvers()
    trade_list = ", ".join(frm for frm, _ in trades[:-1]) + ", and " + trades[-1][0]
    c.qa_variants(["Which Pokemon evolve by trading?",
                   "What Pokemon need to be traded to evolve?"],
                  f"{trade_list} evolve when they are traded to another player.")

    # HMs
    for label, move, fact in HMS:
        c.qa_variants([f"What is {label}?",
                       f"What does {label} teach?",
                       f"What does the move {move} do?",
                       f"How do I use {move}?"],
                      fact)

    # Reverse type lookups: "list water pokemon" / "which pokemon are fire type".
    # These are aggregate queries a tiny model can't DERIVE from the per-Pokemon
    # type facts — it can only memorize the list verbatim, so each type-list gets
    # many phrasings (TYPE_LIST_Q) for repeated exposure. Answer leads with the
    # count so it's useful even if cut short. (The old ctx=43 truncation worry is
    # gone: device context is now 128, so a full 9-name list fits comfortably.)
    for t, names in type_to_names().items():
        joined = ", ".join(names)
        c.qa_variants([q.format(t=t) for q in TYPE_LIST_Q],
                      f"There are {len(names)} {t}-type Kanto Pokemon: {joined}.")

    # Mechanics / general
    for questions, answer in CAPABILITIES:
        c.qa_variants(questions, answer)
    for questions, answer in MECHANICS:
        c.qa_variants(questions, answer)

    # Lore/background: each passage is trained as bare prose (LM exposure of the
    # text) AND bridged with "tell me about X" phrasings whose answer IS the
    # passage — so open-ended lore questions land on the canonical paragraph
    # instead of drifting. Bare + N phrasings gives the passage repeated exposure.
    for questions, passage in LORE:
        c.prose(passage)
        c.qa_variants(questions, passage)

    # ── Precomputed cross-fact "reasoning" ───────────────────────────────────
    # A 6M model can't DERIVE these live (it would need to chain multiple facts),
    # so we compute the answers HERE — deterministically, off the verified Gen-1
    # chart and evolution graph — and bake them in as flat Q&A. This is the
    # reasoning payoff: dual-type weaknesses (product of both types' multipliers,
    # so 2x can cancel to 1x or stack to 4x), evolution-family traversal, dex
    # neighbours, and type combos — all correct by construction.
    into_map = _name_into()
    evolves_from = build_evolves_from()
    num_to_name = {num: name for num, name, _t, _i in POKEMON}
    last_num = len(POKEMON)

    def chain_desc(base):
        into = into_map.get(base) or []
        if not into:
            return f"{base} does not evolve."
        if len(into) > 1:  # Eevee-style branch
            outs = _list_join([t for t, _m, _d in into])
            return f"{base} can evolve into {outs} using evolution stones."
        parts, n_ = [], base
        while True:
            nx = into_map.get(n_) or []
            if not nx:
                break
            to, method, detail = nx[0]
            if method == "level":
                parts.append(f"{to} at level {detail}")
            elif method == "stone":
                parts.append(f"{to} with a {detail}")
            else:
                parts.append(f"{to} when traded")
            n_ = to
        return f"{base} evolves into " + ", then into ".join(parts) + "."

    for num, name, types, into in POKEMON:
        # Per-Pokemon weaknesses — the genuine two-fact reasoning (dual-type product).
        weak4, weak2, immune, resist = defense_profile(types)
        weaknesses = weak4 + weak2
        if weaknesses:
            ans = f"{name} is weak to {_list_join(weaknesses)}."
            if weak4:
                ans = (f"{name} is weak to {_list_join(weaknesses)}, and takes 4x "
                       f"damage from {_list_join(weak4)}.")
            c.qa_variants([f"What is {name} weak to?",
                           f"What is super effective against {name}?",
                           f"What beats {name}?",
                           f"What should I use against {name}?"], ans)
        if weak4:
            c.qa_variants([f"What does 4x damage to {name}?",
                           f"What is {name} doubly weak to?"],
                          f"{name} takes 4x damage from {_list_join(weak4)}.")
        if immune:
            c.qa_variants([f"What is {name} immune to?",
                           f"What does no damage to {name}?",
                           f"What can't hurt {name}?"],
                          f"{name} takes no damage from {_list_join(immune)} attacks.")
        # Per-Pokemon offense coverage (its same-type moves' super-effective set).
        cov = offense_coverage(types)
        if cov:
            c.qa_variants([f"What is {name} good against?",
                           f"What is {name} super effective against?",
                           f"What does {name} beat?"],
                          f"{name}'s attacks are super effective against {_list_join(cov)} Pokemon.")
        # Final evolution (walk the graph forward to the terminal node).
        finals = final_forms(name, into_map)
        fa = (f"{name} is already fully evolved." if finals == [name]
              else f"{name}'s final evolution is {_list_join(finals)}.")
        c.qa_variants([f"What is the final evolution of {name}?",
                       f"What does {name} fully evolve into?",
                       f"What is {name}'s last form?"], fa)
        # Base form (walk backward to the root).
        base = base_form(name, evolves_from)
        ba = (f"{name} is a base-stage Pokemon and does not evolve from anything."
              if base == name else f"{name}'s base form is {base}.")
        c.qa_variants([f"What is {name}'s base form?",
                       f"What is the first form of {name}?",
                       f"What does {name}'s line start from?"], ba)
        # Whole evolution family, keyed to every member.
        cd = chain_desc(base)
        c.qa_variants([f"Show me {name}'s evolution line.",
                       f"What is {name}'s evolution family?",
                       f"How does the {name} line go?"], cd)
        # Dex neighbours.
        if num > 1:
            c.qa_variants([f"Who comes before {name}?",
                           f"What Pokemon is right before {name}?"],
                          f"Before {name} (#{num}) is {num_to_name[num-1]} (#{num-1}).")
        if num < last_num:
            c.qa_variants([f"Who comes after {name}?",
                           f"What Pokemon is right after {name}?"],
                          f"After {name} (#{num}) comes {num_to_name[num+1]} (#{num+1}).")

    # Type-level resistances (now unblocked by the verified RESISTS map).
    for t in TYPES:
        r = RESISTS[t]
        ans = (f"{t}-type Pokemon resist {_list_join(r)} attacks (half damage)."
               if r else f"{t}-type Pokemon do not resist any attack type.")
        c.qa_variants([f"What does {t} resist?",
                       f"What is {t} resistant to?",
                       f"What takes half damage as a {t} type?"], ans)

    # Dual-type combos ("which Pokemon are Water and Ice?").
    combos = {}
    for _num, name, types, _into in POKEMON:
        if len(types) == 2:
            combos.setdefault(tuple(sorted(types)), []).append(name)
    for (a, b), names in combos.items():
        # emit BOTH type orderings so "Water and Ice" and "Ice and Water" both hit
        c.qa_variants([f"Which Pokemon are {a} and {b}?",
                       f"Which Pokemon are {b} and {a}?",
                       f"List the {a}/{b} Pokemon.",
                       f"What Pokemon are both {a} and {b} type?"],
                      f"The {a}/{b} Pokemon are {_list_join(names)}.")

    # Non-evolvers + first/last (small, high-answerability facts).
    non_evolvers = [nm for _n, nm, _t, into in POKEMON if not into and nm not in evolves_from]
    c.qa_variants(["Which Pokemon don't evolve?",
                   "List Pokemon that never evolve.",
                   "What Pokemon are stuck as they are?"],
                  f"{len(non_evolvers)} Kanto Pokemon never evolve, including "
                  f"{_list_join(non_evolvers[:12])}.")
    c.qa_variants(["Who is the first Pokemon in the Kanto Pokedex?",
                   "Which Pokemon starts the Pokedex?",
                   "What is the very first Pokemon?"],
                  f"The first Kanto Pokemon is {num_to_name[1]} (#1), and the last is "
                  f"{num_to_name[last_num]} (#{last_num}).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tokens_out.parent.mkdir(parents=True, exist_ok=True)
    n = c.write(args.out, seed=args.seed)
    qa_count = sum(1 for b in c.blocks if len(b) == 2)
    print(f"Wrote {args.out}")
    print(f"  blocks: {n}  (Q&A pairs: {qa_count}, prose: {n - qa_count})")
    print(f"  Pokemon: {len(POKEMON)}  gyms: {len(GYMS)}  locations: {len(LOCATIONS)}")
    print(f"  conflicting-answer duplicates dropped: {c.conflicts_dropped} (one answer per question enforced)")

    # Whole-word special tokens: the 151 names, so each tokenizes atomically
    # (no partial-name fragments). Pass to a trainer with --special-tokens.
    names = [name for _num, name, _types, _into in POKEMON]
    header = ("# Pokemon name tokens — keep each of the 151 names whole in the\n"
              "# tokenizer so names can't be garbled into partial fragments.\n"
              "# Pass to a trainer with:  --special-tokens training_data/pokemon_special_tokens.txt\n"
              "# One token per line; blank lines and # comments are ignored.\n\n")
    args.tokens_out.write_text(header + "\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {args.tokens_out}  ({len(names)} name tokens)")


if __name__ == "__main__":
    main()
