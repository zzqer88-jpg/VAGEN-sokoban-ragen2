#!/bin/bash
set -eo pipefail

V=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
PROJECT_NAME=${PROJECT_NAME:-sokoban_state_modeling}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_4b_gae_${SOKOBAN_STATE_REWARD_PROFILE:-dense}}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-$V/sokoban_state_modeling/outputs/$EXPERIMENT_NAME}

: "${SOKOBAN_STATE_TRAIN_MANIFEST:?set SOKOBAN_STATE_TRAIN_MANIFEST}"
: "${SOKOBAN_STATE_TRAIN_INDEX:?set SOKOBAN_STATE_TRAIN_INDEX}"
: "${SOKOBAN_STATE_VALIDATION_MANIFEST:?set SOKOBAN_STATE_VALIDATION_MANIFEST}"
: "${SOKOBAN_STATE_VALIDATION_INDEX:?set SOKOBAN_STATE_VALIDATION_INDEX}"

if [ -z "${VERL:-}" ]; then
  for d in "$V/verl" "$V/../verl"; do
    if [ -f "$d/verl/trainer/config/ppo_trainer.yaml" ]; then VERL=$(cd "$d" && pwd); break; fi
  done
fi
: "${VERL:?verl checkout not found; set VERL=/absolute/path/to/verl}"
mkdir -p "$EXPERIMENT_DIR"
export PYTHONPATH="$VERL:$V${PYTHONPATH:+:$PYTHONPATH}"
python3 -m sokoban_state_modeling.scripts.preflight_training \
  --train-manifest "$SOKOBAN_STATE_TRAIN_MANIFEST" \
  --train-index "$SOKOBAN_STATE_TRAIN_INDEX" \
  --validation-manifest "$SOKOBAN_STATE_VALIDATION_MANIFEST" \
  --validation-index "$SOKOBAN_STATE_VALIDATION_INDEX" \
  --verl "$VERL"
mapfile -t BASE < <(grep -vE '^\s*(#|$)' "$V/vagen/configs/training_defaults.flags" | sed "s|\$V|$V|g")

python3 -m vagen.training.main \
  --config-path="$V/vagen/configs" --config-name=vagen_multiturn \
  hydra.searchpath="[file://$VERL/verl/trainer/config]" \
  data.custom_cls.path="$V/vagen/training/dataset.py" \
  "${BASE[@]}" \
  data.train_files="$V/sokoban_state_modeling/configs/train_qwen35_4b.yaml" \
  data.val_files="$V/sokoban_state_modeling/configs/eval_qwen35_4b.yaml" \
  +data.sampler.class_path=sokoban_state_modeling.training.sampler.FullCoverageSampler \
  +data.sampler.full_coverage_cycle=true +data.sampler.pad_final_batch=true \
  actor_rollout_ref.model.path="$MODEL" critic.model.path="$MODEL" critic.enable=true \
  algorithm.adv_estimator=gae algorithm.gamma=1.0 algorithm.lam=1.0 \
  trainer.harness=concat actor_rollout_ref.rollout.n=1 \
  data.train_batch_size=128 data.max_prompt_length=3000 data.max_response_length=768 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  trainer.total_training_steps=400 trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" trainer.default_local_dir="$EXPERIMENT_DIR/checkpoints" \
  trainer.rollout_data_dir="$EXPERIMENT_DIR/rollouts" trainer.validation_data_dir="$EXPERIMENT_DIR/validation" \
  "$@" 2>&1 | tee "$EXPERIMENT_DIR/run.log"
