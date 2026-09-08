"""RAGEN-2 sokoban engine: determinism, solvability and text rendering."""

import numpy as np

from vagen.envs.sokoban.ragen_engine import (
    RagenSokobanEngine,
    generate_room,
    get_shortest_action_path,
)
from vagen.envs.sokoban.utils.seeding import set_seed


def _gen(seed, **kwargs):
    """One first-attempt room generation, mirroring RagenSokobanEngine.reset."""
    with set_seed(seed):
        return generate_room(
            dim=(6, 6),
            num_steps=20,  # int(1.7 * (6 + 6)), gym_sokoban's num_gen_steps
            num_boxes=1,
            second_player=False,
            search_depth=300,
            **kwargs,
        )


def test_same_seed_generates_identical_rooms():
    for seed in (1, 2, 12345):
        fixed_a, state_a, _, _ = _gen(seed)
        fixed_b, state_b, _, _ = _gen(seed)
        assert np.array_equal(fixed_a, fixed_b)
        assert np.array_equal(state_a, state_b)


def test_generated_rooms_are_solvable_with_one_box():
    generated = 0
    for seed in range(1, 51):
        try:
            fixed, state, _, _ = _gen(seed)
        except (RuntimeError, RuntimeWarning):
            continue  # degenerate topology (score == 0) is retried by reset
        generated += 1
        assert np.argwhere(state == 5).shape[0] == 1, "exactly one player"
        assert np.argwhere((state == 3) | (state == 4)).shape[0] == 1, "exactly one box"
        path = get_shortest_action_path(fixed, state, MAX_DEPTH=200)
        assert len(path) >= 1, "room must be solvable"
    assert generated >= 25, "generator should succeed for most seeds"


def test_player_random_movement_changes_rooms_across_config():
    """The RAGEN-2 generator must not equal the gym_sokoban one byte-for-byte.

    add_random_player_movement repositions the player after reverse play, so
    at least some seeds must produce a different room_state than a generator
    without it would. We can't call the old generator here (removed); instead
    assert the player is sometimes NOT adjacent to the box, which the
    original artifact ("player always adjacent to a box") guarantees.
    """
    from vagen.envs.sokoban.ragen_engine.utils import CHANGE_COORDINATES

    non_adjacent = 0
    checked = 0
    for seed in range(1, 121):
        try:
            fixed, state, _, _ = _gen(seed)
        except (RuntimeError, RuntimeWarning):
            continue
        checked += 1
        player = tuple(np.argwhere(state == 5)[0])
        box = tuple(np.argwhere((state == 3) | (state == 4))[0])
        adjacent = any(
            (player[0] + dr, player[1] + dc) == box
            for dr, dc in CHANGE_COORDINATES.values()
        )
        if not adjacent:
            non_adjacent += 1
    assert checked >= 50
    assert non_adjacent >= 5, (
        "expected RAGEN-2's random player movement to decouple player and "
        f"box positions, got {non_adjacent}/{checked} non-adjacent"
    )


def test_engine_respects_difficulty_band_and_partition():
    env = RagenSokobanEngine(dim_room=(6, 6), max_steps=100, num_boxes=1)
    assert env.search_depth == 300
    env.reset(
        seed=2024,
        render_mode="rgb_array",
        min_solution_steps=(1, 5),
        reset_seed_max_tries=1000,
        map_partition=None,
    )
    path = get_shortest_action_path(env.room_fixed, env.room_state, MAX_DEPTH=200)
    assert 1 <= len(path) <= 5


def test_engine_renders_ragen_text_formats():
    env = RagenSokobanEngine(dim_room=(6, 6), max_steps=100, num_boxes=1)
    env.reset(seed=7, render_mode="rgb_array", min_solution_steps=(1, 5))

    grid = env.render("grid")
    assert set(grid) <= set("#_O√XPS\n")

    coord = env.render("coord")
    assert "Board size: 6 rows x 6 cols (zero-indexed)." in coord
    assert "Player: " in coord

    grid_coord = env.render("grid_coord")
    assert grid_coord.startswith("Coordinates:")
    assert "Grid Map:" in grid_coord


def test_search_depth_is_stored_and_threaded_through_construction():
    # gym_sokoban's __init__ auto-runs reset(), which already reads
    # search_depth -- successful construction proves it was set first.
    env = RagenSokobanEngine(dim_room=(6, 6), max_steps=100, num_boxes=1, search_depth=42)
    assert env.search_depth == 42
    env2 = RagenSokobanEngine(dim_room=(6, 6), max_steps=100, num_boxes=1)
    assert env2.search_depth == 300  # RAGEN default
