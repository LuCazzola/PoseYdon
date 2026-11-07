"""The verifier's report: one `Check` per finding, a verdict, `report.json` and `report.txt`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ALLOWED = "ALLOWED"
    NOT_RUN = "NOT RUN"


class Evidence(Enum):
    """What a line rests on; `pipeline` is never proof alone."""

    INDEPENDENT = "independent"
    DERIVED = "derived"
    PIPELINE = "pipeline"


@dataclass
class Check:
    id: str
    status: Status
    evidence: Evidence
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
            lines.append(f"{c.status.value:8} {c.id} [{c.evidence.value}]{measured} {c.where or ''} {c.note}".rstrip())
        return "\n".join(lines) + "\n"

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        body = {"verdict": self.verdict.value, "run": self.run,
                "checks": [{**asdict(c), "status": c.status.value, "evidence": c.evidence.value} for c in self.checks]}
        (directory / "report.json").write_text(json.dumps(body, indent=2, default=str))
        (directory / "report.txt").write_text(self.text())


def apply_allowances(report: Report, table: dict) -> Report:
    """Turn a FAIL inside the owner's budget into ALLOWED. ``table`` is `allowances.yaml`:
    ``{rig: {clip: [{check, joint, budget, reason}]}}``, empty today, so nothing changes."""
    entries = table.get(report.run.get("rig"), {}).get(report.run.get("action"), []) or []
    for c in report.checks:
        if c.status is Status.FAIL and any(e["check"] == c.id and (c.value or 0) <= e["budget"] for e in entries):
            c.status = Status.ALLOWED
    return report
