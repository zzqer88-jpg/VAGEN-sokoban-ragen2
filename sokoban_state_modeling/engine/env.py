"""State-loading engine facade for replay and native VAGEN rendering."""

from __future__ import annotations

import numpy as np
from functools import lru_cache
import hashlib
from importlib.metadata import distribution
from pathlib import Path
from PIL import Image

from sokoban_state_modeling.state.codec import StateSnapshot, state_to_room
from sokoban_state_modeling.state.transition import apply_action, apply_actions

SPRITE_FILENAMES = (
    "wall.png", "floor.png", "box_target.png", "box_on_target.png",
    "box.png", "player.png", "player_on_target.png",
)


def _sprite_distribution():
    return distribution("gym-sokoban")


def _sprite_root() -> Path:
    return Path(_sprite_distribution().locate_file("gym_sokoban/envs/surface"))


def renderer_provenance() -> dict:
    """Return enough immutable metadata to detect a changed rendering domain."""
    root = _sprite_root()
    return {
        "gym_sokoban_version": _sprite_distribution().version,
        "sprite_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SPRITE_FILENAMES
        },
    }


@lru_cache(maxsize=1)
def _load_surfaces() -> tuple[np.ndarray, ...]:
    """Load immutable sprite pixels once per worker process."""
    package_root = _sprite_root()
    surfaces: list[np.ndarray] = []
    for name in SPRITE_FILENAMES:
        with Image.open(package_root / name) as source:
            pixels = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        pixels.setflags(write=False)
        surfaces.append(pixels)
    return tuple(surfaces)


def render_state(state: StateSnapshot) -> np.ndarray:
    fixed, room = state_to_room(state)
    room[(room == 5) & (fixed == 2)] = 6
    surfaces = _load_surfaces()
    result = np.zeros((state.shape[0] * 16, state.shape[1] * 16, 3), dtype=np.uint8)
    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            result[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = surfaces[int(room[r, c])]
    return result


class SokobanStateEngine:
    def __init__(self, state: StateSnapshot):
        self.load_state(state)

    def load_state(self, state: StateSnapshot) -> None:
        self.state = state.copy()

    def clone(self) -> "SokobanStateEngine":
        return SokobanStateEngine(self.state)

    def step(self, action: str):
        result = apply_action(self.state, action)
        self.state = result.state
        return result

    def replay(self, actions: list[str]):
        state, events, subtypes, terminated = apply_actions(self.state, actions)
        self.state = state
        return state, events, subtypes, terminated

    def render(self) -> np.ndarray:
        return render_state(self.state)
