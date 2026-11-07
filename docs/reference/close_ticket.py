"""Close tickets: remove ideas from ideas.md by ID, close issues by number. An issue fixed
by a commit is better closed by `Fixes #N: <what fixed it>` in its message or PR."""

from __future__ import annotations

import argparse

from open_ticket import IDEAS, gh, is_issue


def close_ideas(ids: list[str]) -> None:
    lines = IDEAS.read_text().splitlines()
    rows = {tid: f"| `{tid}` |" for tid in ids}
    missing = [tid for tid, row in rows.items() if not any(l.startswith(row) for l in lines)]
    if missing:
        raise SystemExit(f"no idea with ID: {', '.join(missing)}")
    IDEAS.write_text("\n".join(l for l in lines if not l.startswith(tuple(rows.values()))) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ids", nargs="+", help="idea IDs and/or issue numbers (#12 or 12)")
    p.add_argument("--reason", choices=["completed", "not planned"], default="completed", help="issues only")
    p.add_argument("--comment", help="required for issues: why it is closed (what fixed it, or why it was wrong)")
    a = p.parse_args()
    if any(is_issue(t) for t in a.ids) and not a.comment:
        p.error("closing an issue needs --comment: what fixed it, or why it was wrong")
    if ideas := [t for t in a.ids if not is_issue(t)]:
        close_ideas(ideas)
    for n in (t.lstrip("#") for t in a.ids if is_issue(t)):
        gh("issue", "close", n, "--reason", a.reason, "--comment", a.comment)
