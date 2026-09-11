import asyncio
import json
from pathlib import Path

from sokoban_state_modeling import GENERATOR_VERSION, RENDERER_VERSION, SCHEMA_VERSION
from sokoban_state_modeling.env import SokobanStateModelingEnv
from sokoban_state_modeling.engine.env import _load_surfaces, renderer_provenance
from sokoban_state_modeling.generator.manifest_store import build_index
from sokoban_state_modeling.state.codec import grid_to_state, map_hash, state_hash, task_hash
from sokoban_state_modeling.state.transition import apply_actions
from vagen.envs import GymEnvAdapter, build_env


def test_single_turn_environment(tmp_path: Path):
    _load_surfaces.cache_clear()
    current = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    next_state, events, subtypes, terminal = apply_actions(current, ["Right"])
    row = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "renderer_version": RENDERER_VERSION,
        "task_id": "test_00000001", "task_index": 0, "accepted_attempt": 0, "split": "test",
        "resolved_generator_config": {"dim_room": [6, 6], "num_boxes": 1},
        "map_hash": map_hash(current), "current_state_hash": state_hash(current), "next_state_hash": state_hash(next_state),
        "query_actions": ["Right"], "task_hash": task_hash(current, ["Right"]),
        "current_state": current.to_dict(), "next_state": next_state.to_dict(), "event_sequence": events,
        "event_subtype_sequence": subtypes, "executed_action_count": 1, "terminated": terminal,
    }
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "test.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    index = manifests / "test.idx"
    build_index(manifest, index)

    async def run():
        env = SokobanStateModelingEnv({"manifest_path": str(manifest), "manifest_index_path": str(index)})
        obs, _ = await env.reset(0)
        assert obs["obs_str"].startswith("<image>") and len(obs["multi_modal_input"]["<image>"]) == 1
        first_image = obs["multi_modal_input"]["<image>"][0]
        assert first_image.size == (96, 96)
        assert _load_surfaces.cache_info().misses == 1
        repeated_obs, _ = await env.reset(0)
        repeated_image = repeated_obs["multi_modal_input"]["<image>"][0]
        assert repeated_image.tobytes() == first_image.tobytes()
        assert _load_surfaces.cache_info().hits >= 1
        answer = "<perception>\n" + "\n".join(current.to_dict()["grid"]) + "\n</perception>\n<prediction>\n" + "\n".join(next_state.to_dict()["grid"]) + "\n</prediction>"
        _, reward, done, info = await env.step(answer)
        assert reward == 1.0 and done and info["success"]
        assert info["reward_metrics"]["transition_consistency_rate"] == 1.0
        await env.close()

        adapter = GymEnvAdapter(
            build_env(SokobanStateModelingEnv, {
                "manifest_path": str(manifest), "manifest_index_path": str(index),
            }, max_turns=1),
            "SokobanStateModeling", {"seed": 0},
        )
        adapted_obs, adapted_info = await adapter.reset()
        assert adapted_info["task_index"] == 0
        assert adapted_obs["images"][0].size == (96, 96)
        await adapter.close()

    asyncio.run(run())


def test_renderer_provenance_covers_every_cached_sprite():
    provenance = renderer_provenance()
    assert provenance["gym_sokoban_version"]
    assert len(provenance["sprite_sha256"]) == len(_load_surfaces()) == 7
    assert all(len(value) == 64 for value in provenance["sprite_sha256"].values())
