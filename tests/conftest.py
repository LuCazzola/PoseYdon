"""`--dataset <name>` picks the corpus the data tests probe.

A dataset `<name>` is its prepared corpus `data/<name>/`, its raw source
`data/_raw/<name>/` and its config `configs/dataset/<name>.yaml`.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--dataset", default="truebones", help="dataset under data/ to probe")
