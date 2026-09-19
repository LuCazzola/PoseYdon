import numpy as np
import pytest

from poseydon.build.skinning import SkinningInput, SkinningPolicyError, plan_skinning
from poseydon.features.reduce import COLLAPSE, DROP, RemovalOp


def test_weighted_animated_auxiliary_without_policy_fails():
    source = SkinningInput(
        joint_names=("Parent", "Cloth"),
        weights=np.array([[0.4, 0.6], [1.0, 0.0]]),
        animated=frozenset({"Cloth"}),
        removals=(RemovalOp("Cloth", "Parent", 1, DROP),),
    )
    with pytest.raises(SkinningPolicyError, match="weighted.*animated"):
        plan_skinning(source, policy=None)


def test_unweighted_leaf_drops_without_changing_weights():
    source = SkinningInput(
        joint_names=("Parent", "Helper"), weights=np.array([[1.0, 0.0]]),
        animated=frozenset(), removals=(RemovalOp("Helper", "Parent", 1, DROP),),
    )
    plan = plan_skinning(source, policy="exclude")
    assert plan.decisions[0].operation == "drop"
    np.testing.assert_allclose(plan.output_weights.sum(axis=1), 1.0)


def test_collapse_reports_exact_weight_transfer_metrics():
    source = SkinningInput(
        joint_names=("Parent", "Twist"),
        weights=np.array([[0.5, 0.5], [0.25, 0.75]]),
        animated=frozenset(),
        removals=(RemovalOp("Twist", "Parent", 1, COLLAPSE),),
    )
    plan = plan_skinning(source, policy="exclude")
    decision = plan.decisions[0]
    assert decision.destination == "Parent"
    assert decision.affected_vertices == 2
    assert decision.total_transferred_weight == pytest.approx(1.25)
    np.testing.assert_allclose(plan.output_weights, [[1.0], [1.0]])
