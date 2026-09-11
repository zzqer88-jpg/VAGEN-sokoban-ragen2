"""State legality and deterministic forward solving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .codec import StateSnapshot
from .transition import ACTIONS, apply_action, is_terminal


def validation_errors(state: StateSnapshot, num_boxes: int | None = None) -> list[str]:
    errors: list[str] = []
    if np.any(~np.isin(state.fixed, (0, 1, 2))):
        errors.append("fixed values must be 0, 1, or 2")
    if np.any(~np.isin(state.entity, (0, 1, 2))):
        errors.append("entity values must be 0, 1, or 2")
    if np.any((state.fixed == 0) & (state.entity != 0)):
        errors.append("entities cannot occupy walls")
    border = np.concatenate((state.fixed[0], state.fixed[-1], state.fixed[:, 0], state.fixed[:, -1]))
    if np.any(border != 0):
        errors.append("outer border must contain only walls")
    if int(np.sum(state.entity == 1)) != 1:
        errors.append("state must contain exactly one player")
    boxes = int(np.sum(state.entity == 2))
    targets = int(np.sum(state.fixed == 2))
    if boxes != targets:
        errors.append("box and target counts must match")
    if num_boxes is not None and (boxes != int(num_boxes) or targets != int(num_boxes)):
        errors.append(f"expected exactly {num_boxes} boxes and targets")
    return errors


def is_valid_state(state: StateSnapshot, num_boxes: int | None = None) -> bool:
    return not validation_errors(state, num_boxes)


@dataclass(frozen=True)
class SolverResult:
    status: str
    shortest_steps: int | None
    actions: tuple[str, ...] = ()
    expanded_nodes: int = 0


def solve_forward(state: StateSnapshot, max_nodes: int = 500_000) -> SolverResult:
    if validation_errors(state):
        return SolverResult("invalid", None)
    def key_of(item: StateSnapshot) -> bytes:
        # fixed is constant throughout one search; only the compact entity layer is
        # needed for identity. This avoids hashing/JSON-serialising the walls at every
        # expanded node.
        return item.entity.tobytes(order="C")

    start_key = key_of(state)
    queue = deque([state])
    parents: dict[bytes, tuple[bytes | None, str | None]] = {start_key: (None, None)}
    expanded = 0
    while queue:
        current = queue.popleft()
        current_key = key_of(current)
        if is_terminal(current):
            actions: list[str] = []
            cursor = current_key
            while parents[cursor][0] is not None:
                previous, action = parents[cursor]
                actions.append(str(action))
                cursor = previous  # type: ignore[assignment]
            actions.reverse()
            return SolverResult("solvable", len(actions), tuple(actions), expanded)
        if expanded >= max_nodes:
            return SolverResult("solver_unknown", None, expanded_nodes=expanded)
        expanded += 1
        for action in ACTIONS:
            nxt = apply_action(current, action).state
            key = key_of(nxt)
            if key not in parents:
                parents[key] = (current_key, action)
                queue.append(nxt)
    return SolverResult("unsolvable", None, expanded_nodes=expanded)
