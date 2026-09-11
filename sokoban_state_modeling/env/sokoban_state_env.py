"""One-response VAGEN environment for RGB state extraction and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from sokoban_state_modeling.engine import render_state
from sokoban_state_modeling.generator.manifest_store import ManifestStore
from sokoban_state_modeling.prompt.template import build_task_prompt, system_prompt
from sokoban_state_modeling.reward.metrics import METRIC_NAMES
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.state.codec import grid_to_state
from vagen.envs import GymImageEnv


@dataclass
class SokobanStateEnvConfig:
    manifest_path: str
    manifest_index_path: str | None = None
    reward_profile: str = "dense"
    image_placeholder: str = "<image>"
    verify_manifest_rows: bool = True

    def __post_init__(self) -> None:
        if self.reward_profile not in {"dense", "exact_only"}:
            raise ValueError("reward_profile must be dense or exact_only")


class SokobanStateModelingEnv(GymImageEnv):
    """The model answers once; the environment-supplied action block is immutable."""

    PUBLIC_REWARD_METRIC_NAMES = METRIC_NAMES
    REWARD_METRIC_NAMES = METRIC_NAMES + (
        "conditional_prediction_success_numerator",
        "conditional_prediction_success_denominator",
    )

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        self.config = SokobanStateEnvConfig(**env_config)
        self.store = ManifestStore(self.config.manifest_path, self.config.manifest_index_path)
        self.task: dict[str, Any] | None = None
        self._answered = False

    async def system_prompt(self) -> dict[str, Any]:
        return {"obs_str": system_prompt()}

    async def reset(self, seed: int):
        task_index = int(seed)
        self.task = self.store.read(task_index, verify=self.config.verify_manifest_rows)
        self._answered = False
        rows, cols = self.task["resolved_generator_config"]["dim_room"]
        prompt = build_task_prompt(rows, cols, self.task["query_actions"], self.config.image_placeholder)
        current = grid_to_state(self.task["current_state"]["grid"])
        image = Image.fromarray(render_state(current))
        return {
            "obs_str": prompt,
            "multi_modal_input": {self.config.image_placeholder: [image]},
        }, {"task_id": self.task["task_id"], "task_index": task_index}

    async def step(self, action_str: str):
        if self.task is None:
            raise RuntimeError("reset must be called before step")
        if self._answered:
            raise RuntimeError("SokobanStateModelingEnv accepts exactly one response")
        self._answered = True
        current = grid_to_state(self.task["current_state"]["grid"])
        next_state = grid_to_state(self.task["next_state"]["grid"])
        result = score_response(
            action_str, current, next_state, self.task["query_actions"],
            self.task["resolved_generator_config"]["num_boxes"], self.config.reward_profile,
        )
        info = {
            "success": bool(result.components["exact_all"]),
            "format_correct": bool(result.parsed.format_exact),
            "reward_metrics": result.metrics,
            "reward_components": result.components,
            "task_id": self.task["task_id"],
            "task_index": self.task["task_index"],
        }
        return {"obs_str": "State-modeling task complete."}, result.reward, True, info

    async def close(self) -> None:
        self.task = None
