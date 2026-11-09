"""Open a ticket and print its ID: an idea becomes a row in ideas.md (4-character ID),
any other tier a GitHub issue labelled `Prio:<TIER>` and `needs-triage` (issue number)."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import secrets
import string
import subprocess
import tomllib
from pathlib import Path

PARALLAX = Path(__file__).resolve().parents[1]
REPO = PARALLAX.parents[1]
IDEAS = PARALLAX / "pages/tickets/ideas.md"
ALPHABET = string.ascii_lowercase + string.digits
ISSUE_TIERS = ["critical", "wanted", "minor"]
TIERS = ISSUE_TIERS + ["idea"]
TRIAGE = "needs-triage"
LEAD_NOTE = ("> **Lead, not fact.** This is what was believed when it was written: check it against "
             "the code and the data before acting, and fix this issue if it disagrees.")


def path_roots() -> list[Path]:
    with (PARALLAX / "parallax.toml").open("rb") as f:
        return [REPO] + [REPO / root for root in tomllib.load(f).get("source_roots", [])]


def existing_path(path: str) -> str:
    if not any((root / path).exists() for root in path_roots()):
        raise argparse.ArgumentTypeError(f"{path!r} not found under the repo root or a source root")
    return path


def format_paths(where: list[str]) -> str:
    return ", ".join(f"`{w}`" for w in where)


def is_issue(tid: str) -> bool:
    return tid.lstrip("#").isdigit()


# ---- GitHub issues ----

def prio_label(tier: str) -> str:
    return f"Prio:{tier.upper()}"


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def issue_body(observed: str, where: list[str] | str, extra: str = "",
               repro: str | None = None, cause: str | None = None) -> str:
    """What was seen, how to see it again and the guessed cause are kept apart: they age differently."""
    paths = where if isinstance(where, str) else format_paths(where)
    parts = [f"**Where:** {paths}{extra}", f"### Observed\n{observed}"]
    if repro:
        parts.append(f"### Reproduce\n```bash\n{repro}\n```")
    if cause:
        parts.append(f"### Suspected cause\n{cause}")
    return "\n\n".join(parts + [LEAD_NOTE]) + "\n"


def parse_issue_body(body: str) -> dict[str, str | None]:
    """The `issue_body` arguments back from a body (a pre-sections body is all Observed)."""
    head, _, rest = body.replace("\r\n", "\n").partition("\n\n")
    m = re.match(r"\*\*Where:\*\* (.*?)(\n.*)?$", head, re.S)
    if not m:
        raise SystemExit("issue body is not in open_ticket.py's format; edit it on GitHub instead")
    rest = rest.replace(LEAD_NOTE, "").strip()
    sections = {k: v.strip() for k, v in re.findall(r"^### (.+?)\n(.*?)(?=^### |\Z)", rest, re.M | re.S)}
    repro = re.sub(r"^```\w*\n|\n?```$", "", sections.get("Reproduce", "")).strip()
    return {"where": m.group(1), "extra": m.group(2) or "",
            "observed": sections.get("Observed", rest if not sections else ""),
            "repro": repro or None, "cause": sections.get("Suspected cause") or None}


def title_of(issue: str) -> str:
    first = re.split(r"(?<=[.;:])\s|\s[-–—]\s", issue, maxsplit=1)[0].rstrip(".;:")
    if first.count("(") > first.count(")"):
        first = first.rsplit("(", 1)[0].rstrip()
    return first if len(first) <= 80 else first[:79].rsplit(" ", 1)[0] + "…"


def open_issue(tier: str, issue: str, where: list[str] | str, title: str | None = None, extra: str = "",
               repro: str | None = None, cause: str | None = None) -> str:
    require_github()
    url = gh("issue", "create", "--title", title or title_of(issue),
             "--body", issue_body(issue, where, extra, repro, cause), "--label", f"{prio_label(tier)},{TRIAGE}")
    return "#" + url.rsplit("/", 1)[1]


def require_github() -> None:
    try:
        ok = subprocess.run(["gh", "repo", "view"], cwd=REPO, capture_output=True).returncode == 0
    except FileNotFoundError:
        ok = False
    if not ok:
        raise SystemExit("this repo is not reachable on GitHub through gh: only ideas work here")


def issue_json(number: str, fields: str) -> dict:
    return json.loads(gh("issue", "view", number.lstrip("#"), "--json", fields))


# ---- ideas.md ----

def new_id(text: str) -> str:
    used = set(re.findall(r"^\| `([0-9a-z]{4})` \|", text, re.M))
    while (tid := "".join(secrets.choice(ALPHABET) for _ in range(4))) in used:
        pass
    return tid


def append_row(lines: list[str], cells: list[str]) -> None:
    """Insert a row of already-escaped cells into the ideas table, newest first."""
    rows = [i for i, l in enumerate(lines) if l.startswith("| `")]
    header_end = max(i for i, l in enumerate(lines) if l.startswith("|---"))
    at = next((i for i in rows if lines[i].split(" | ")[1] <= cells[1]), max(rows, default=header_end) + 1)
    lines.insert(at, "| " + " | ".join(cells) + " |")


def open_idea(issue: str, where: list[str] | str, noticed: str | None = None) -> str:
    lines = IDEAS.read_text().splitlines()
    tid = new_id("\n".join(lines))
    paths = where if isinstance(where, str) else format_paths(where)
    cells = [f"`{tid}`", noticed or datetime.date.today().isoformat(), paths, issue]
    append_row(lines, [c.replace("|", r"\|") for c in cells])
    IDEAS.write_text("\n".join(lines) + "\n")
    return tid


def open_ticket(tier: str, issue: str, where: list[str], title: str | None = None,
                repro: str | None = None, cause: str | None = None) -> str:
    return open_idea(issue, where) if tier == "idea" else open_issue(tier, issue, where, title, repro=repro, cause=cause)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tier", choices=TIERS)
    p.add_argument("issue", help="issue: what was observed, facts only (idea: what to change or investigate, and why)")
    p.add_argument("where", nargs="+", type=existing_path,
                   help="paths involved, relative to the repo root or a source root")
    p.add_argument("--title", help="issue title (default: the issue's first sentence)")
    p.add_argument("--repro", help="issue: command(s) that show it again with the exact options used")
    p.add_argument("--cause", help="issue: the suspected cause, kept apart from what was observed")
    a = p.parse_args()
    if a.tier == "idea" and (a.title or a.repro or a.cause):
        p.error("--title/--repro/--cause apply to issues only")
    print(open_ticket(**vars(a)))
