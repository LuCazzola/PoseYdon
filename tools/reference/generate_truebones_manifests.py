"""Generate PoseYdon skeleton manifests from the reference's own constants.

The reference keeps its per-species knowledge as Python literals in
``param_utils.py``: ``FACE_JOINTS`` (four joint INDICES per species) plus
taxonomy lists. Foot joints it does not store at all -- it re-derives them
from joint names every run. This turns all of that into one YAML per
skeleton, which is where PoseYdon keeps it.

Runs in the `compat` image, because ``FACE_JOINTS`` indexes into the joint
ordering produced by the reference's OWN BVH loader, and that ordering is
data-dependent rather than a fixed convention:

* End Sites are joints when the file names them in a ``#name:`` comment
  (Truebones does this for some rigs, e.g. Crab's ``BN_leg_R_05_end_site``)
  and are dropped when it does not (e.g. BrownBear);
* a redundant zero-offset root child is merged away, but only when the
  hierarchy actually has that shape (BrownBear does, Crab does not).

So BrownBear resolves in a 38-joint ordering and Crab in a 54-joint one,
from the same loader. Guessing the convention gets one of them wrong and
silently mirrors the character, so this asks the real loader instead.
Names are stable across loaders where they overlap, which is exactly why
PoseYdon's manifests reference joints by name.

    docker compose run --rm compat python tools/reference/generate_truebones_manifests.py --out data/truebones/skeletons
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import BVH  # Motion, only present in the compat image

from poseydon.io.bvh import BVH as PoseydonBVH

PARAM_UTILS = Path(
    "external/neural_motion_blending/data_loaders/truebones/truebones_utils/param_utils.py"
)
RAW_ROOT = Path("data/truebones/Truebone_Z-OO")

# The reference's own foot-name heuristic, verbatim.
FOOT_KEYWORDS = ("toe", "foot", "phalanx", "hoof", "ashi")
# Snakes have no feet to find, so the reference treats every joint as one.
ALL_JOINTS_ARE_FEET = ("Anaconda", "KingCobra")

# The reference never categorised these: it lists them under NO_BVHS and drops
# them from every taxonomy list. They do ship BVHs here, so they are tagged from
# the animal rather than left untagged and invisible to every tag query.
UNCATEGORIZED = {
    "Crow": ("flying", "bird"),
    "Dog": ("quadruped", "mammal"),
    "Dog-2": ("quadruped", "mammal"),
}

# Reference list name -> the tag PoseYdon uses.
TAXONOMY_TAGS = {
    "QUADROPEDS": "quadruped",
    "BIPEDS": "biped",
    "MILLIPEDS": "milliped",
    "SNAKES": "snake",
    "FISH": "fish",
    "FLYING": "flying",
}


def reference_constants() -> dict:
    """``param_utils.py``'s literals, read without importing it.

    Parsing beats importing here: the module pulls in numpy and resolves
    dataset paths relative to the reference's own working directory, none
    of which this needs.
    """
    if not PARAM_UTILS.is_file():
        raise SystemExit(
            f"reference checkout not found at {PARAM_UTILS}; this tool reads the "
            "reference's own constants and cannot run without it"
        )
    tree = ast.parse(PARAM_UTILS.read_text())
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        # `IGNORE_OBJECTS = NO_BVHS` aliases an earlier list rather than
        # restating it, so plain literal_eval would silently drop it.
        if isinstance(node.value, ast.Name) and node.value.id in constants:
            constants[name] = constants[node.value.id]
            continue
        try:
            constants[name] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue  # numpy calls and list concatenations, none of which are needed
    return constants


def find_tpose(species_dir: Path) -> Path | None:
    """The reference's ``find_tpos_path``: a "tpos" file, else idle, else the first."""
    files = sorted(species_dir.glob("*.bvh"))
    if not files:
        return None
    for path in files:
        if "tpos" in path.name.lower():
            return path
    for path in files:
        if path.name.lower().lstrip("_").startswith("idle"):
            return path
    return files[0]


def foot_joints(bvh, species: str) -> list[str]:
    """The reference's ``suspected_foot_indices``, run over OUR joint names.

    Name-matched joints, then every descendant of one. The reference grows
    the list while iterating it, so the child rule cascades all the way down
    a limb; that is reproduced here rather than corrected, since these
    manifests have to describe the same feet the reference used.

    The heuristic is purely name-based, so unlike ``FACE_JOINTS`` it needs no
    index translation and can run on PoseYdon's own parse -- which is what
    makes the result guaranteed to resolve. Taking these names from the
    reference loader instead does NOT work: where a BVH leaves an End Site
    unnamed, the two loaders synthesise different names for it
    (``ElkRPhalanxPrima_end_site`` there, ``ElkRPhalanxPrima_End`` here), and
    a manifest naming the reference's spelling fails against every clip.

    Those synthesised End Sites are then dropped, because the reference's
    loader only ever saw the End Sites a file names explicitly. Keeping them
    would add tip joints the reference never counted as feet.
    """
    names = list(bvh.names)
    parents = [int(p) for p in bvh.parents]
    synthesized = {
        i
        for i in range(len(names))
        if not bvh.channels[i] and parents[i] >= 0 and names[i] == f"{names[parents[i]]}_End"
    }

    if species in ALL_JOINTS_ARE_FEET:
        return [n for i, n in enumerate(names) if i not in synthesized]

    suspected = [
        i for i, name in enumerate(names) if any(k in name.lower() for k in FOOT_KEYWORDS)
    ]
    for index in suspected:  # deliberately iterating a list that grows
        if index in parents:
            for child, parent in enumerate(parents):
                if parent == index and child not in suspected:
                    suspected.append(child)
    return [names[i] for i in sorted(suspected) if i not in synthesized]


def tags_for(species: str, constants: dict, extra: dict[str, list[str]]) -> list[str]:
    tags = [
        tag
        for key, tag in TAXONOMY_TAGS.items()
        if species in constants.get(key, ())
    ]
    for tag in UNCATEGORIZED.get(species, ()):
        if tag not in tags:
            tags.append(tag)
    for tag in extra.get(species, []):
        if tag not in tags:
            tags.append(tag)
    return tags


def existing_extra_tags(out_dir: Path, taxonomy: set[str]) -> dict[str, list[str]]:
    """Hand-added tags (``mammal``, ``bird``) in manifests already written.

    The reference has no such axis, so regenerating would drop them.
    """
    import yaml

    extra: dict[str, list[str]] = {}
    for path in out_dir.glob("*.yaml"):
        if path.stem.startswith("_"):
            continue
        data = yaml.safe_load(path.read_text()) or {}
        kept = [t for t in (data.get("tags") or []) if t not in taxonomy]
        if kept:
            extra[str(data.get("skeleton", path.stem))] = kept
    return extra


def render(species: str, facing: list[str], feet: list[str], tags: list[str]) -> str:
    lines = [
        "base: _base.yaml",
        f"skeleton: {species}",
        "facing:",
        f"  hips:      {{right: {facing[0]}, left: {facing[1]}}}",
        f"  shoulders: {{right: {facing[2]}, left: {facing[3]}}}",
    ]
    if feet:
        lines.append("foot_joints:")
        lines.extend(f"  - {name}" for name in feet)
    else:
        lines.append("foot_joints: []   # the reference's heuristic finds none for this rig")
    lines.append(f"tags: [{', '.join(tags)}]")
    return "\n".join(lines) + "\n"


BASE_MANIFEST = """# Shared by every Truebones skeleton. Values the reference keeps as GLOBAL
# constants, identical for every species -- so they live here once instead of
# being restated in seventy files. A species manifest overrides any of them by
# declaring the key itself.
fps: 30                 # every one of the 1153 raw Truebones clips is 30 fps, so
                        # this is a real check rather than a default: ingest
                        # rejects anything else instead of mixing rates silently.
                        # The reference's FPS=20 is not a data rate at all -- it
                        # discards each file's frame time and only labels its
                        # rendered videos 20 fps, playing 30 fps motion 1.5x slow.
scale: {mean_bone_length: 0.20921428571428569}   # reference HML_AVG_BONELEN
contact:
  max_height: 0.30      # reference FOOT_CONTACT_HEIGHT_THRESH
  max_speed: 0.044721359549995794   # normalized length per frame; sqrt of the
                        # reference's FOOT_CONTACT_VEL_THRESH (0.002), which it
                        # compares against a SQUARED displacement. Must stay
                        # exact: rounding flips borderline contact frames.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/truebones/skeletons"))
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument(
        "--species", nargs="*", default=None, help="default: every species with face joints"
    )
    args = parser.parse_args()

    constants = reference_constants()
    face_joints = constants["FACE_JOINTS"]
    ignore = set(constants.get("IGNORE_OBJECTS", ()))
    taxonomy = set(TAXONOMY_TAGS.values())

    args.out.mkdir(parents=True, exist_ok=True)
    extra = existing_extra_tags(args.out, taxonomy)
    (args.out / "_base.yaml").write_text(BASE_MANIFEST)

    wanted = args.species or sorted(face_joints)
    written, skipped, noted = 0, [], []
    for species in wanted:
        if species in ignore:
            # The reference calls these NO_BVHS, but that describes ITS copy of
            # the corpus: Crow, Dog and Dog-2 all ship BVH files here. Generate
            # them anyway and say so, rather than inheriting a stale exclusion.
            noted.append(species)
        if species not in face_joints:
            skipped.append((species, "no FACE_JOINTS entry in the reference"))
            continue
        species_dir = args.raw_root / species
        tpose = find_tpose(species_dir) if species_dir.is_dir() else None
        if tpose is None:
            skipped.append((species, f"no .bvh files under {species_dir}"))
            continue

        _anim, names, _ = BVH.load(str(tpose))
        indices = face_joints[species]
        if max(indices) >= len(names):
            skipped.append(
                (species, f"face index {max(indices)} exceeds {len(names)} joints in {tpose.name}")
            )
            continue

        facing = [names[i] for i in indices]
        feet = foot_joints(PoseydonBVH.read(tpose), species)
        tags = tags_for(species, constants, extra)
        (args.out / f"{species}.yaml").write_text(render(species, facing, feet, tags))
        written += 1

    print(f"wrote {written} manifests + _base.yaml -> {args.out}")
    for species, reason in skipped:
        print(f"  skipped {species}: {reason}")
    if noted:
        print(
            f"  generated despite the reference's IGNORE_OBJECTS (they do have BVHs "
            f"here): {', '.join(sorted(noted))}"
        )


if __name__ == "__main__":
    main()
