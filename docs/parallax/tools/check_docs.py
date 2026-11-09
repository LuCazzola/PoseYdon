#!/usr/bin/env python3
"""Checks docs against the repo with no model in the loop: paths named in docs exist, User
Guide pages name their code, API stubs match the packages. A pass means paths match, not that
what the docs say is right."""
import re
import sys
import tomllib
from pathlib import Path

PARALLAX = Path(__file__).resolve().parents[1]
REPO = PARALLAX.parents[1]
PAGES = PARALLAX / "pages"
NOT_SOURCE = {"site-packages", "node_modules", "venv", "__pycache__"}
KNOWN_EXTENSIONS = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".sh", ".txt", ".js", ".ts",
                    ".rs", ".go", ".c", ".h", ".cpp", ".ini", ".cfg", ".html", ".css"}
NOT_A_PATH = re.compile(r"[\s<>*{}$]|://")
EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]*$")
CODE_LINE = re.compile(r"^Code:\s+(.*)$")
INLINE = re.compile(r"`([^`\n]+)`")


def load_config():
    try:
        with (PARALLAX / "parallax.toml").open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(print(f"parallax.toml: cannot read it: {e}") or 1)


def is_ignored_dir(path):
    return any(part.startswith(".") or part in NOT_SOURCE for part in path.parts)


def repo_filenames():
    return {p.name for p in REPO.rglob("*") if not is_ignored_dir(p.relative_to(REPO))}


def looks_like_path(token):
    """A known extension makes it a path; otherwise only a real top-level entry does, so
    `nowhere/b` stays prose."""
    if NOT_A_PATH.search(token) or token.startswith(("/", "..", "~")):
        return False
    if Path(token).suffix in KNOWN_EXTENSIONS:
        return True
    return "/" in token and (REPO / token.removeprefix("./").split("/")[0]).exists()


def path_exists(token, roots, names):
    return ("/" not in token and token in names) or any((root / token).exists() for root in roots)


def page_lines(page):
    fenced = False
    for number, line in enumerate(page.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            yield number, line


def check_page(page, rel, require_code_line, names):
    problems, has_code_line = [], False
    for number, line in page_lines(page):
        code_line = CODE_LINE.match(line)
        if code_line:
            has_code_line = True
            tokens = [t.strip(" `") for t in code_line.group(1).split(",")]
            problems += [f"{rel}:{number}: missing path {t}" for t in tokens if not (REPO / t).exists()]
            continue
        problems += [f"{rel}:{number}: missing path {t}" for t in INLINE.findall(line)
                     if looks_like_path(t) and not path_exists(t, (REPO, PAGES, page.parent), names)]
    if require_code_line and not has_code_line:
        problems.append(f"{rel}:1: User Guide page has no Code: line")
    return problems


def packages(root):
    return {".".join(p.parent.relative_to(root).parts): p.parent for p in root.rglob("__init__.py")
            if not is_ignored_dir(p.parent.relative_to(root))}


def check_api_stubs(cfg):
    api = PAGES / "api"
    if not api.is_dir():
        return []
    problems, found = [], {}
    for source in cfg.get("source_roots", []):
        found.update({name: (path, api / path.relative_to(REPO / source) / "index.md")
                      for name, path in packages(REPO / source).items()})
    for name, (path, stub) in found.items():
        if not stub.is_file():
            problems.append(f"{stub.relative_to(REPO)}: missing API stub for {name}")
            continue
        text = stub.read_text()
        wanted = [name] + [f"{name}.{m.stem}" for m in path.glob("*.py") if m.name != "__init__.py"]
        problems += [f"{stub.relative_to(REPO)}: stub skips {w}" for w in wanted if f"::: {w}\n" not in text]
    stubs = {stub for _, stub in found.values()}
    nav = (PARALLAX / "mkdocs.yml").read_text()
    for stub in api.rglob("*.md"):
        rel = stub.relative_to(PAGES)
        if stub not in stubs:
            problems.append(f"{rel}: stub for a package that no longer exists")
        if str(rel) not in nav:
            problems.append(f"{rel}: not in the mkdocs nav")
    return problems


def main():
    cfg = load_config()
    guide = PAGES / "user-guide"
    names = repo_filenames()
    problems = []
    for page in sorted(PAGES.rglob("*.md")):
        rel = page.relative_to(REPO)
        needs_code = page.is_relative_to(guide) and page.name != "index.md"
        problems += check_page(page, rel, needs_code, names)
    problems += check_api_stubs(cfg)
    print("\n".join(problems) or "docs check passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
