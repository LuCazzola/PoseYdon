"""The verifier's report: one `Check` per finding, a verdict, `report.json` and `report.txt`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ALLOWED = "ALLOWED"
    NOT_RUN = "NOT RUN"


class Evidence(Enum):
    """What a line rests on; `pipeline` is never proof alone; `driven` is a probe the verifier made."""

    INDEPENDENT = "independent"
    DERIVED = "derived"
    PIPELINE = "pipeline"
    DRIVEN = "driven"


@dataclass
class Check:
    id: str
    status: Status
    evidence: Evidence | None  # None on a NOT RUN line that compared nothing: no evidence to label
    value: float | None = None
    limit: float | None = None
    where: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Report:
    run: dict
    checks: list[Check] = field(default_factory=list)

    @property
    def verdict(self) -> Status:
        """FAIL if any check failed, NOT RUN if none failed but one did not run,
        ALLOWED if every pass leans on an allowance somewhere, else PASS."""
        seen = {c.status for c in self.checks}
        for status in (Status.FAIL, Status.NOT_RUN, Status.ALLOWED):
            if status in seen:
                return status
        return Status.PASS if seen else Status.NOT_RUN

    @property
    def exit_code(self) -> int:
        return 0 if self.verdict in (Status.PASS, Status.ALLOWED) else 1

    def text(self) -> str:
        lines = [f"verdict: {self.verdict.value}", *(f"{k}: {v}" for k, v in self.run.items() if k != "sha256")]
        for c in self.checks:
            measured = "" if c.value is None else f" value={c.value:.4g} limit={c.limit}"
            where = {k: float(v) if isinstance(v, np.floating) else v for k, v in c.where.items()}
            label = "" if c.evidence is None else f" [{c.evidence.value}]"
            lines.append(f"{c.status.value:8} {c.id}{label}{measured} {where or ''} {c.note}".rstrip())
        return "\n".join(lines) + "\n"

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        body = {"verdict": self.verdict.value, "run": self.run,
                "checks": [{**asdict(c), "status": c.status.value, "evidence": c.evidence and c.evidence.value} for c in self.checks]}
        (directory / "report.json").write_text(json.dumps(body, indent=2, default=str))
        (directory / "report.txt").write_text(self.text())


def not_applicable(run: dict, text: str) -> None:
    """List a check that cannot apply to this input, with its reason, in ``run["not_applicable"]``."""
    run["not_applicable"] = f"{run['not_applicable']}; {text}" if "not_applicable" in run else text


def apply_allowances(report: Report, table: dict) -> Report:
    """Turn a FAIL inside the owner's budget into ALLOWED; an entry's ``joint`` and ``file`` must equal the
    line's `where` values. ``table`` is `allowances.yaml`:
    ``{rig: {clip: [{check, joint, budget, reason}]}}``, empty today, so nothing changes."""
    entries = table.get(report.run.get("rig"), {}).get(report.run.get("action"), []) or []
    if any(e["check"] == "C2a" for e in entries):
        raise ValueError("C2a takes no allowances")
    for c in report.checks:
        if c.status is Status.FAIL and c.value is not None and any(
            e["check"] == c.id and c.value <= e["budget"]
            and all(c.where.get(k) == e[k] for k in ("joint", "file") if e.get(k) is not None) for e in entries
        ):
            c.status = Status.ALLOWED
    return report
