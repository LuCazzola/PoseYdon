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
    text = "`src/poseydon/core/` `core/spec.py:42` `a/b.py:3-9` `docs/api/core/index.md#limitations`"
    assert mentioned_paths(text) == {
        "src/poseydon/core", "core/spec.py", "a/b.py", "docs/api/core/index.md",
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


def test_docs_do_not_cite_planning_notes():
    """Specs and plans are working notes kept out of the repo (CLAUDE.md), so
    no page may lean on one to justify anything.
    """
    citing = sorted(
        str(page.relative_to(ROOT))
        for page in (ROOT / "docs").rglob("*.md")
        if "superpowers/" in page.read_text()
    )
    assert not citing, f"cite code, docstrings or commits instead: {citing}"


def test_every_library_package_has_an_api_page():
    """The API reference is generated, but each package's page is a stub that
    names it, so a new package or subpackage needs one (and a line in
    mkdocs.yml's nav). The pages follow the package tree:
    `poseydon.models.modules` is `docs/api/models/modules/index.md`.
    """
    lib = ROOT / "src" / "poseydon"
    missing = []
    for init in sorted(lib.rglob("*/__init__.py")):
        rel = init.parent.relative_to(lib)
        page = ROOT / "docs" / "api" / rel / "index.md"
        dotted = "poseydon." + ".".join(rel.parts)
        if not page.exists() or f"::: {dotted}\n" not in page.read_text():
            missing.append(dotted)
    assert not missing, (
        f"add docs/api/<pkg/as/path>/index.md containing `::: poseydon.<pkg>`: {missing}"
    )


def test_every_library_module_is_on_exactly_one_api_page():
    """Pages name their modules one `:::` line each (a package page renders only
    its docstring), so a new module must be added to a page, and a module moved
    to its own page in the dropdown must leave the package page.
    """
    lib = ROOT / "src" / "poseydon"
    modules = {
        "poseydon." + ".".join(p.relative_to(lib).with_suffix("").parts)
        for p in lib.rglob("*.py")
        if p.parent != lib and p.name != "__init__.py" and not p.stem.startswith("_")
    }
    named: dict[str, list[str]] = {}
    for page in (ROOT / "docs" / "api").rglob("*.md"):
        for name in re.findall(r"^::: (poseydon\.[\w.]+)$", page.read_text(), re.M):
            named.setdefault(name, []).append(str(page.relative_to(ROOT)))
    unlisted = sorted(modules - named.keys())
    twice = {name: pages for name, pages in named.items() if len(pages) > 1}
    assert not unlisted, f"name each on an API page as `::: <module>`: {unlisted}"
    assert not twice, f"render each module on one page only: {twice}"
