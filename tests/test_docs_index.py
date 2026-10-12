"""docs/index/ must track the tree it describes.

Only structure is checked: that INDEX.md names every package, script and tool,
and names no path that has gone. Whether the prose is still true cannot be
tested; that is the upkeep rule in CLAUDE.md.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index" / "INDEX.md"

_CODE = re.compile(r"`([^`\n]+)`")


def mentioned_paths(text: str) -> set[str]:
    """Repo paths named in inline code, without trailing slash, :line or #anchor.

    A span counts as a path only if it contains "/". Spans with spaces are
    commands, spans with <, *, { or $ are patterns, and URLs and absolute paths
    are not repo paths, so all of those are skipped.
    """
    paths = set()
    for span in _CODE.findall(text):
        if " " in span or "://" in span or span.startswith("/"):
            continue
        if any(c in span for c in "<>*{}$"):
            continue
        if "/" not in span:
            continue
        span = re.sub(r"(:\d+(-\d+)?|#.*)$", "", span)
        paths.add(span.rstrip("/"))
    return paths


def required_paths(root: Path) -> set[str]:
    """Everything the index must name: library packages and modules, scripts, tools."""
    lib = root / "src" / "poseydon"
    out = {
        f"src/poseydon/{p.name}"
        for p in lib.iterdir()
        if (p.is_dir() and (p / "__init__.py").exists())
        or (p.suffix == ".py" and p.name != "__init__.py")
    }
    for d in ("scripts", "tools"):
        out |= {f"{d}/{p.name}" for p in (root / d).glob("*.py") if p.name != "__init__.py"}
    return out


def test_mentioned_paths_normalises_suffixes():
    text = "`src/poseydon/core/` `core/spec.py:42` `a/b.py:3-9` `docs/index/library/core.md#purpose`"
    assert mentioned_paths(text) == {
        "src/poseydon/core", "core/spec.py", "a/b.py", "docs/index/library/core.md",
    }


def test_mentioned_paths_skips_patterns_commands_and_urls():
    text = (
        "`data/truebones/rigs/<Rig>/manifest.yaml` `data/**/*.npz` `configs/{a,b}` "
        "`docker compose run --rm test` `https://x.org/a` `/app/src` `FeatureSpec` `spec.py`"
    )
    assert mentioned_paths(text) == set()


def test_required_paths_ignores_non_packages(tmp_path):
    lib = tmp_path / "src" / "poseydon"
    (lib / "core").mkdir(parents=True)
    (lib / "core" / "__init__.py").touch()
    (lib / "__pycache__").mkdir()
    (lib / "__init__.py").touch()
    (lib / "cli.py").touch()
    for d in ("scripts", "tools"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "__init__.py").touch()
    (tmp_path / "scripts" / "run.py").touch()
    (tmp_path / "tools" / "reference").mkdir()
    (tmp_path / "tools" / "probe.py").touch()
    assert required_paths(tmp_path) == {
        "src/poseydon/core", "src/poseydon/cli.py", "scripts/run.py", "tools/probe.py",
    }


def test_index_names_every_package_script_and_tool():
    missing = required_paths(ROOT) - mentioned_paths(INDEX.read_text())
    assert not missing, f"add to docs/index/INDEX.md: {sorted(missing)}"


def test_index_names_only_existing_paths():
    stale = sorted(p for p in mentioned_paths(INDEX.read_text()) if not (ROOT / p).exists())
    assert not stale, f"docs/index/INDEX.md names paths that do not exist: {stale}"


LIBRARY_PAGES = ROOT / "docs" / "index" / "library"
SECTIONS = ("Purpose", "Key files and types", "How it connects",
            "Why it is shaped this way", "Gotchas")


def missing_sections(text: str) -> list[str]:
    """Fixed headings that are absent or out of order, so agents can jump to them."""
    found = [h for h in re.findall(r"^## (.+?)\s*$", text, re.M) if h in SECTIONS]
    return [s for s in SECTIONS if s not in found] or (
        [] if found == list(SECTIONS) else ["order: " + ", ".join(found)]
    )


def test_missing_sections_reports_absent_and_misordered():
    full = "\n".join(f"## {s}" for s in SECTIONS)
    assert missing_sections(full) == []
    assert missing_sections(full.replace("## Gotchas", "## Gotcha")) == ["Gotchas"]
    swapped = full.replace("## Purpose", "## X").replace("## Gotchas", "## Purpose")
    assert missing_sections(swapped + "\n## Gotchas") == [
        "order: Key files and types, How it connects, Why it is shaped this way, Purpose, Gotchas"
    ]


def test_every_library_package_has_a_page_with_the_fixed_sections():
    lib = ROOT / "src" / "poseydon"
    problems = {}
    for pkg in sorted(p.name for p in lib.iterdir() if (p / "__init__.py").exists()):
        page = LIBRARY_PAGES / f"{pkg}.md"
        if not page.exists():
            problems[pkg] = "no page"
        elif gaps := missing_sections(page.read_text()):
            problems[pkg] = gaps
    assert not problems, f"docs/index/library/: {problems}"
