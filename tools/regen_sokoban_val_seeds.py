"""Regenerate the validation seed manifest for the sokoban val yamls.

The RAGEN-2 room generator (vagen.envs.sokoban.ragen_engine) produces
different maps than the gym_sokoban generator the manifest was derived from,
so every seed in ``seed_list`` has to be re-derived.  A seed qualifies only
when its FIRST generation attempt (no retry walk) yields a room that

* solves within the ``min_solution_steps`` band (BFS, like reset's gate),
* hashes into the eval bucket of the map partition, and
* is a room no other kept seed produces.

All yamls passed on the command line must agree on the generation-relevant
env config; they receive the same seed list (tests assert the manifests are
identical across val files).

Usage:
    python tools/regen_sokoban_val_seeds.py \
        examples/train/sokoban/val_sokoban_vision.yaml \
        examples/train/sokoban/val_sokoban_vision_sr.yaml [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vagen.envs.sokoban.ragen_engine import (  # noqa: E402
    _room_matches_partition,
    generate_room,
    get_shortest_action_path,
)
from vagen.envs.sokoban.utils.seeding import set_seed  # noqa: E402

SEED_LIST_RE = re.compile(r"seed_list:\s*\[[^\]]*\]", re.S)
SEEDS_PER_LINE = 16

# Keys that decide which rooms a seed produces. Everything else in env config
# (reward scales, prompt format, ...) may differ between the val yamls.
GENERATION_KEYS = (
    "dim_room",
    "num_boxes",
    "min_solution_steps",
    "min_solution_bfs_max_depth",
    "search_depth",
    "map_partition_modulus",
    "map_partition_eval_bucket",
)


def _spec(path: Path):
    return yaml.safe_load(path.read_text())["envs"][0]


def _gen_config(spec) -> dict:
    """Generation-relevant config with the SokobanEnvConfig defaults filled in."""
    config = spec.get("config") or {}
    resolved = {
        "dim_room": tuple(config.get("dim_room", (6, 6))),
        "num_boxes": config.get("num_boxes", 1),
        "min_solution_steps": config.get("min_solution_steps"),
        "min_solution_bfs_max_depth": config.get("min_solution_bfs_max_depth", 200),
        "search_depth": config.get("search_depth", 300),
        "map_partition_modulus": config.get("map_partition_modulus", 4),
        "map_partition_eval_bucket": config.get("map_partition_eval_bucket", 0),
    }
    return resolved


def _num_gen_steps(dim_room) -> int:
    # gym_sokoban: num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
    return int(1.7 * (dim_room[0] + dim_room[1]))


def find_seeds(gen_cfg: dict, n_envs: int, start_seed: int):
    band = gen_cfg["min_solution_steps"]
    seeds, fingerprints = [], set()
    seed = start_seed
    tried = 0
    while len(seeds) < n_envs:
        tried += 1
        if tried % 5000 == 0:
            print(f"  ... {tried} seeds tried, {len(seeds)}/{n_envs} accepted", flush=True)
        try:
            with set_seed(seed):
                fixed, state, _, _ = generate_room(
                    dim=gen_cfg["dim_room"],
                    num_steps=_num_gen_steps(gen_cfg["dim_room"]),
                    num_boxes=gen_cfg["num_boxes"],
                    second_player=False,
                    search_depth=gen_cfg["search_depth"],
                )
        except (RuntimeError, RuntimeWarning):
            seed += 1
            continue
        solution_len = len(
            get_shortest_action_path(
                fixed, state, MAX_DEPTH=gen_cfg["min_solution_bfs_max_depth"]
            )
        )
        in_band = band is None or (band[0] <= solution_len <= band[1])
        if not in_band or solution_len < 1:
            seed += 1
            continue
        if not _room_matches_partition(
            fixed, state, "eval",
            gen_cfg["map_partition_modulus"], gen_cfg["map_partition_eval_bucket"],
        ):
            seed += 1
            continue
        fingerprint = fixed.tobytes() + state.tobytes()
        if fingerprint in fingerprints:
            seed += 1
            continue
        fingerprints.add(fingerprint)
        seeds.append(seed)
        seed += 1
    return seeds, tried


def rewrite_seed_list(path: Path, seeds) -> None:
    rows = [
        ", ".join(str(s) for s in seeds[i : i + SEEDS_PER_LINE])
        for i in range(0, len(seeds), SEEDS_PER_LINE)
    ]
    body = ",\n".join("      " + row for row in rows)
    replacement = "seed_list: [\n" + body + ",\n    ]"
    text = path.read_text()
    if not SEED_LIST_RE.search(text):
        raise SystemExit(f"{path}: no seed_list block found")
    path.write_text(SEED_LIST_RE.sub(replacement, text, count=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yamls", nargs="+", type=Path)
    parser.add_argument("--start-seed", type=int, default=10001)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = {p: _spec(p) for p in args.yamls}
    ref_path = args.yamls[0]
    ref_cfg = _gen_config(specs[ref_path])
    ref_n = specs[ref_path]["n_envs"]
    for path, spec in specs.items():
        if _gen_config(spec) != ref_cfg or spec["n_envs"] != ref_n:
            raise SystemExit(
                f"{path}: generation-relevant env config differs from {ref_path}; "
                "the val yamls must share one seed manifest."
            )

    print(f"Generating {ref_n} seeds from {ref_path} config: {ref_cfg}")
    seeds, tried = find_seeds(ref_cfg, ref_n, args.start_seed)
    print(f"Accepted {len(seeds)} seeds after trying {tried} candidates "
          f"([{seeds[0]} .. {seeds[-1]}])")

    if args.dry_run:
        print("dry-run: no files written")
        return
    for path in args.yamls:
        rewrite_seed_list(path, seeds)
        print(f"wrote {len(seeds)} seeds -> {path}")


if __name__ == "__main__":
    main()
