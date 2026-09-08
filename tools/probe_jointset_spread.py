"""Per rig: do the clips agree with each other on their joint set, and with the T-pose?"""
from collections import Counter
from pathlib import Path

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")

rows = []
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    bvhs = sorted(d.glob("*.bvh"))
    if not bvhs:
        continue
    sets = {}
    for p in bvhs:
        try:
            sets[p] = tuple(BVH.read_names(p))
        except Exception as exc:  # noqa: BLE001
            sets[p] = ("<unreadable>", str(exc))
    tp = [p for p in bvhs if "TPOS" in p.name.upper()]
    counts = Counter(sets.values())
    modal, modal_n = counts.most_common(1)[0]
    tpose_set = sets[tp[0]] if tp else None
    rows.append((
        d.name, len(bvhs), len(counts), modal_n, len(modal),
        None if tpose_set is None else len(tpose_set),
        None if tpose_set is None else (tpose_set == modal),
        None if tpose_set is None else len(set(modal) - set(tpose_set)),
    ))

print(f"{'rig':<18}{'clips':>6}{'distinct':>9}{'modal_n':>8}{'modal_J':>8}{'tpose_J':>8}{'tp==modal':>11}{'absent':>7}")
bad_internal = []
for r in rows:
    name, n, distinct, modal_n, modal_j, tp_j, same, absent = r
    flag = "" if distinct == 1 else "  <-- clips disagree"
    if distinct > 1:
        bad_internal.append(name)
    print(f"{name:<18}{n:>6}{distinct:>9}{modal_n:>8}{modal_j:>8}"
          f"{'-' if tp_j is None else tp_j:>8}{same!s:>11}{'-' if absent is None else absent:>7}{flag}")

print(f"\nrigs total: {len(rows)}")
print(f"rigs whose clips disagree among themselves: {len(bad_internal)} {bad_internal}")
print(f"rigs with NO tpose file: {[r[0] for r in rows if r[5] is None]}")
print(f"rigs where tpose != modal: {[r[0] for r in rows if r[6] is False]}")
