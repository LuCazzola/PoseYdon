"""Validate a proposed per-rig rest-pose override against each rig's modal joint set."""
from collections import Counter
from pathlib import Path

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")

# Rest-pose file per rig, for rigs whose T-pose is missing or is a minority
# skeleton. Preference: idle, then walk (or fly, for flying creatures). Named
# outright rather than matched, because a substring rule picks DeathIdle,
# LieIdle and idle_attack.
NO_TPOSE = {
    "Anaconda":        "__Idle.bvh",
    "Bird":            "__IdleLoop.bvh",
    "Camel":           "__IdleLoop.bvh",
    "Cricket":         "__Idle.bvh",
    "Dog":             "__Idle.bvh",
    "Goat":            "__Idle.bvh",
    "Lion":            "__SlowIdle.bvh",     # __DeathIdle.bvh is the trap
    "Monkey":          "__Idle1.bvh",
    "Pteranodon":      "__FlyLoop.bvh",      # no idle clip; flying creature
    "Rat":             "__Trottle.bvh",      # no idle clip; trot is the nearest gait
    "SabreToothTiger": "__Startwalk.bvh",    # no idle clip
    "Scorpion-2":      "__Idle.bvh",
    "Trex":            "__walk_loop.bvh",    # __STILL.bvh is the outlier skeleton (66j vs 78j)
}

TPOSE_IS_OUTLIER = {
    "Ant":    "__Idle.bvh",
    "Crab":   "__Walk.bvh",                  # no idle clip
    "Deer":   "__Idle.bvh",
    "Jaguar": "__Idle.bvh",                  # __LieIdle.bvh is the trap
}

for label, table in (("no T-pose", NO_TPOSE), ("T-pose is an outlier", TPOSE_IS_OUTLIER)):
    print(f"\n########## {label}")
    for rig, choice in table.items():
        d = RAW / rig
        bvhs = sorted(d.glob("*.bvh"))
        sets = {p.name: tuple(BVH.read_names(p)) for p in bvhs}
        modal, modal_n = Counter(sets.values()).most_common(1)[0]
        path = d / choice
        if choice not in sets:
            near = [n for n in sets if "WALK" in n.upper() or "FLY" in n.upper()]
            print(f"  {rig:<17} MISSING {choice}   candidates: {near}")
            continue
        ok = sets[choice] == modal
        print(f"  {rig:<17} {choice:<22} joints={len(sets[choice]):>4} "
              f"modal={len(modal):>4} ({modal_n}/{len(bvhs)})  "
              f"{'in modal set' if ok else '*** NOT IN MODAL SET ***'}")
