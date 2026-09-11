#!/usr/bin/env python3
"""Fail fast on dataset/runtime mistakes before allocating a training job."""

from __future__ import annotations

import argparse
from pathlib import Path

from sokoban_state_modeling.generator.manifest_store import ManifestStore


def _check_split(name: str, manifest: Path, index: Path, expected: int) -> None:
    store = ManifestStore(manifest, index)
    if len(store) != expected:
        raise RuntimeError(f"{name} manifest has {len(store)} rows; expected {expected}")
    first, last = store.read(0), store.read(expected - 1)
    if first["split"] != name or last["split"] != name:
        raise RuntimeError(f"{name} manifest contains the wrong split label")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    args = parser.parse_args()

    _check_split("train", args.train_manifest, args.train_index, 50_000)
    _check_split("validation", args.validation_manifest, args.validation_index, 2_000)
    trainer_config = args.verl / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
    if not trainer_config.is_file():
        raise RuntimeError(f"incomplete VERL checkout: missing {trainer_config}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the training environment") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA GPU is visible to PyTorch")
    print(
        f"preflight passed: train=50000 validation=2000 "
        f"cuda_devices={torch.cuda.device_count()} torch={torch.__version__}"
    )


if __name__ == "__main__":
    main()
