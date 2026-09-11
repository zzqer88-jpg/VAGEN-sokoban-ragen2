from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def test_split_configs_cover_exact_manifest_ranges():
    expected = {
        "train_qwen35_4b.yaml": (50_000, [0, 49_999, 1]),
        "eval_qwen35_4b.yaml": (2_000, [0, 1_999, 1]),
        "test_qwen35_4b.yaml": (10_000, [0, 9_999, 1]),
    }
    for filename, (count, seeds) in expected.items():
        row = yaml.safe_load((CONFIG_ROOT / filename).read_text(encoding="utf-8"))["envs"][0]
        assert row["n_envs"] == count
        assert row["seed"] == seeds
        assert row["max_turns"] == 1
        assert row["stop_strings"] == ["</prediction>"]
