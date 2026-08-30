"""Skeleton graph relations.

These are what let one model span skeletons of different shapes: instead of
learning a fixed joint ordering, attention is biased by how two joints relate --
parent, child, sibling, or how many hops apart they are.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

# Number of hops beyond which joints are simply "far". Bounding this keeps the
# embedding table small and stops a 200-joint centipede from needing 200 buckets.
DEFAULT_MAX_PATH = 5


class EdgeType(IntEnum):
    SELF = 0
    PARENT = 1
    CHILD = 2
    SIBLING = 3
    NO_RELATION = 4
    END_EFFECTOR = 5
    TOKEN_CONNECTION = 6


def edge_relations(parents: np.ndarray) -> np.ndarray:
    """``(J, J)`` categorical relation between every ordered pair of joints.

    A joint with no children is marked END_EFFECTOR on its diagonal instead of
    SELF, so leaves are distinguishable without a separate feature.
    """
    parents = np.asarray(parents)
    n = len(parents)
    relations = np.full((n, n), EdgeType.NO_RELATION, dtype=np.int64)

    has_children = np.zeros(n, dtype=bool)
    has_children[parents[parents >= 0]] = True

    for i in range(n):
        for j in range(n):
            if i == j:
                relations[i, j] = EdgeType.SELF
            elif parents[j] == i:
                relations[i, j] = EdgeType.CHILD
            elif j == parents[i]:
                relations[i, j] = EdgeType.PARENT
            elif parents[j] == parents[i]:
                relations[i, j] = EdgeType.SIBLING
        if not has_children[i]:
            relations[i, i] = EdgeType.END_EFFECTOR

    return relations


def hop_distances(parents: np.ndarray, max_path: int = DEFAULT_MAX_PATH) -> np.ndarray:
    """``(J, J)`` tree distance in hops, saturating at ``max_path``.

    Joints are in topological order, so a joint's distance to any earlier joint
    is one more than its parent's distance to it.
    """
    parents = np.asarray(parents)
    n = len(parents)
    distances = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0
            elif j < i:
                distances[i, j] = distances[j, i]
            elif parents[j] == i:
                distances[i, j] = 1
            else:
                distances[i, j] = distances[i, parents[j]] + 1

    return np.minimum(distances, max_path)
