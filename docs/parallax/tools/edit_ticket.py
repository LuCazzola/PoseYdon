"""Change a ticket's tier, text or paths, or mark an issue verified. Moving between an idea
and an issue opens it in the new place and closes it in the old one."""

from __future__ import annotations

import argparse
import re

from open_ticket import (IDEAS, ISSUE_TIERS, TIERS, TRIAGE, append_row, existing_path, format_paths, gh,
                         is_issue, prio_label, issue_body, issue_json, open_idea, open_issue, parse_issue_body)
from close_ticket import close_ideas


def edit_issue(number: str, tier: str | None, issue: str | None, where: list[str] | None,
               verified: str | None, repro: str | None, cause: str | None) -> str:
    n = number.lstrip("#")
    data = issue_json(n, "body,labels,title")
    body = parse_issue_body(data["body"])
    changes = {"observed": issue, "where": where and format_paths(where), "repro": repro, "cause": cause}
    body.update({k: v for k, v in changes.items() if v is not None})
    if tier == "idea":
        cause_note = f" Suspected cause: {body['cause']}" if body["cause"] else ""
        tid = open_idea(f"{body['observed']}{cause_note} (was #{n}: {data['title']})", body["where"])
        gh("issue", "close", n, "--reason", "not planned", "--comment", f"Moved to ideas.md as `{tid}`.")
        return tid
    args = []
    if any(v is not None for v in changes.values()):
        args += ["--body", issue_body(**body)]
    if tier is not None:
        prios = {prio_label(t) for t in ISSUE_TIERS}
        stale = [label["name"] for label in data["labels"] if label["name"] in prios and label["name"] != prio_label(tier)]
        args += ["--add-label", prio_label(tier)] + [a for s in stale for a in ("--remove-label", s)]
    if verified is not None:
        args += ["--remove-label", TRIAGE]
        gh("issue", "comment", n, "--body", f"**Verified:** {verified}")
    if args:
        gh("issue", "edit", n, *args)
    return f"#{n}"


def edit_idea(tid: str, tier: str | None, issue: str | None, where: list[str] | None) -> str:
    lines = IDEAS.read_text().splitlines()
    row = next((i for i, l in enumerate(lines) if l.startswith(f"| `{tid}` |")), None)
    if row is None:
        raise SystemExit(f"no idea with ID: {tid}")
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", lines[row])[1:-1]]  # cells stay escaped
    if where is not None:
        cells[2] = format_paths(where).replace("|", r"\|")
    if issue is not None:
        cells[3] = issue.replace("|", r"\|")
    if tier in ISSUE_TIERS:
        text = cells[3].replace(r"\|", "|")
        number = open_issue(tier, text, cells[2].replace(r"\|", "|"), extra=f"\n**From idea:** `{tid}`")
        close_ideas([tid])
        return number
    lines[row] = "| " + " | ".join(cells) + " |"
    IDEAS.write_text("\n".join(lines) + "\n")
    return tid


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tid", metavar="ID", help="idea ID, or issue number (#12 or 12)")
    p.add_argument("--tier", choices=TIERS)
    p.add_argument("--issue", help="replaces the issue's Observed text (an idea's whole text)")
    p.add_argument("--where", nargs="+", type=existing_path,
                   help="replaces every path, relative to the repo root or a source root")
    p.add_argument("--repro", help="issues only: replaces the Reproduce command(s)")
    p.add_argument("--cause", help="issues only: replaces the Suspected cause")
    p.add_argument("--verified", metavar="EVIDENCE",
                   help="issues only: drops needs-triage and comments the evidence (red test, command + output)")
    a = p.parse_args()
    if all(v is None for v in (a.tier, a.issue, a.where, a.verified, a.repro, a.cause)):
        p.error("nothing to change: pass --tier, --issue, --where, --repro, --cause and/or --verified")
    if is_issue(a.tid):
        print(edit_issue(a.tid, a.tier, a.issue, a.where, a.verified, a.repro, a.cause))
    elif any(v is not None for v in (a.verified, a.repro, a.cause)):
        p.error("--verified/--repro/--cause apply to issues only")
    else:
        print(edit_idea(a.tid, a.tier, a.issue, a.where))
