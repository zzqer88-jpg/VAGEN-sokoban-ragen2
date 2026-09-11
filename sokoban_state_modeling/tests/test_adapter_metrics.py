import asyncio
from types import SimpleNamespace

from vagen.envs._common.adapter import GymEnvAdapter


class _MetricEnv:
    REWARD_METRIC_NAMES = ("metric_a", "metric_b")
    PUBLIC_REWARD_METRIC_NAMES = ("metric_a",)

    async def reset(self, seed=None):
        return {"obs_str": "ready"}, {}

    async def system_prompt(self):
        return {"obs_str": "system"}

    async def step(self, text):
        return {"obs_str": "done"}, 1.0, True, {
            "success": True,
            "reward_metrics": {"metric_a": 1.0, "metric_b": 0.25},
        }

    async def close(self):
        pass


def test_adapter_collects_declared_reward_metrics():
    async def run():
        adapter = GymEnvAdapter(_MetricEnv(), "MetricEnv", {})
        await adapter.reset(0)
        await adapter.step(SimpleNamespace(text="x", token_ids=[], tokenizer=None))
        assert adapter.finalized_reward_metrics() == {
            "metric_a": 1.0, "metric_b": 0.25, "reward_metric_error": 0.0,
        }
        assert adapter.finalized_public_reward_metrics() == {"metric_a": 1.0}

    asyncio.run(run())
