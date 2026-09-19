import json

import numpy as np

from poseydon.build.prepare import PrepareChain
from poseydon.build.report import PreparationReport
from poseydon.core.authority import AuthorityPolicy
from tests.build.test_prepare import Double, Shift, _anim


def _run_report():
    report = PreparationReport.start(
        dataset="fixture", rig="Rig", policy=AuthorityPolicy()
    )
    chain = PrepareChain((Shift(), Double()))
    chain.apply(
        _anim(),
        resolved=None,
        rig_params={"shift": {"amount": np.array([5.0, 2.0, -3.0])}},
        audit=report,
    )
    return report


def test_report_semantics_and_hash_are_deterministic():
    first = _run_report()
    second = _run_report()

    assert first.semantic_dict() == second.semantic_dict()
    assert first.semantic_sha256 == second.semantic_sha256


def test_prepare_chain_records_executed_stages_in_order():
    report = _run_report()

    assert [operation.kind for operation in report.operations] == ["shift", "double"]
    assert [operation.scope for operation in report.operations] == ["rig", "clip"]
    assert [operation.operation_id for operation in report.operations] == [
        "Rig/apply/shift/0000",
        "Rig/apply/double/0001",
    ]
    assert all(operation.input_topology_sha256 for operation in report.operations)
    assert all(operation.output_topology_sha256 for operation in report.operations)



def test_fit_and_invert_are_audited_at_the_actual_execution_site():
    report = PreparationReport.start("fixture", "Rig", AuthorityPolicy())
    chain = PrepareChain((Shift(), Double()))
    source = _anim()

    rig_params = chain.fit_rig(source, resolved=None, audit=report)
    prepared, params = chain.apply(source, None, rig_params)
    chain.invert(prepared, params, audit=report)

    assert [operation.direction for operation in report.operations] == [
        "fit_rig", "fit_rig", "invert", "invert"
    ]

def test_report_write_uses_null_free_canonical_json(tmp_path):
    path = tmp_path / "report.json"
    report = _run_report()

    report.write(path)

    loaded = json.loads(path.read_text())
    assert loaded == report.semantic_dict()
    assert path.read_text().endswith("\n")
