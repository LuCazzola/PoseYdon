"""`--dataset <name>` picks the corpus the data tests probe.

A dataset `<name>` is its prepared corpus `data/<name>/`, its raw source
`data/_raw/<name>/` and its config `configs/dataset/<name>.yaml`.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--dataset", default="truebones", help="dataset under data/ to probe")


def pytest_terminal_summary(terminalreporter):
    """Say so when the FBX half was skipped: a green run is then BVH-only."""
    skipped = terminalreporter.stats.get("skipped", [])
    if any("fbx" in report.nodeid for report in skipped):
        terminalreporter.write_line(
            "NOTE: the FBX round trip was skipped (no FBX SDK here); this run checked "
            "BVH only. Run it in the `fbxsdk` container to check both formats.",
            yellow=True,
        )
