"""The layering rule the final whole-branch review found nothing enforcing.

`src/` is the library; `scripts/` and `build/` are the build-time tooling on
top of it. Two directions can silently invert:

1. `src/` importing `scripts/` -- the library would depend on a thin CLI
   wrapper, backwards from how the package is meant to be used (and how it
   would need to be packaged/installed without `scripts/`).
2. Something outside `build/`+`scripts/` importing `build.pipeline`,
   `build.corpus` or `build.names` -- those three are the BUILD-TIME
   internals (rig discovery, humanizing, the four build passes), not a
   format other code should couple to. `build.index` and `build.prepare` are
   the exception: they are shared FORMATS (the clip index; the invertible
   preparation transform), and training code already reads them
   (`poseydon/data/dataset.py`, `poseydon/training/build.py`).

A grep over the source tree is the honest shape here: an AST-based import
check would need to resolve re-exports and `TYPE_CHECKING` blocks for no
extra safety this rule actually needs.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "poseydon"

#: Modules whose internals only `build/` and `scripts/` may depend on.
INTERNAL_ONLY = ("poseydon.build.pipeline", "poseydon.build.corpus", "poseydon.build.names")

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(path: Path) -> set[str]:
    text = path.read_text()
    return set(_IMPORT_RE.findall(text))


def test_src_does_not_import_scripts():
    offenders = []
    for path in _python_files(SRC):
        for module in _imported_modules(path):
            if module == "scripts" or module.startswith("scripts."):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports `{module}`")
    assert offenders == [], "\n".join(offenders)


def test_only_build_and_scripts_import_the_build_internals():
    offenders = []
    build_and_scripts = {SRC / "build"}

    for path in _python_files(REPO_ROOT):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT)
        # Allowed importers: the build package itself, scripts/, and tests/
        # (tests exercise the internals directly by design -- see
        # tests/build/test_pipeline.py).
        if any(path.is_relative_to(d) for d in build_and_scripts):
            continue
        if rel.parts[0] in ("scripts", "tests"):
            continue
        for module in _imported_modules(path):
            if any(module == internal or module.startswith(internal + ".") for internal in INTERNAL_ONLY):
                offenders.append(f"{rel} imports `{module}`")
    assert offenders == [], "\n".join(offenders)
