#!/usr/bin/env bash
# P0-3 ablation runner — implements the action described in
# docs/report/01_gate-detection-plan/plan.md, P0-3 section.
#
# Sequence (each step is idempotent — skips when its output already exists):
#   1. build_sam_dataset.py  → symlink train/val/test under data/sam/
#   2. generate_sam_masks.py → SAM2 ring-masks under data/sam_masked/
#   3. train cell A — gatenet baseline                  (configs/default.yaml)
#   4. train cell B — deeplabv3plus + imagenet weights  (configs/deeplabv3plus_imagenet_nonorm.yaml)
#   5. train cell C — deeplabv3plus + imagenet + norm   (configs/deeplabv3plus_imagenet.yaml)
#
# All three training cells use the same flight split (seed 42, deterministic
# in data/discovery.py) and the same training-time RNG seed (42, set via
# torch.manual_seed in trainer.train), so the only thing that varies across
# the three runs is the model + normalisation cell knob.
#
# Logs land in logs/p0_3/<timestamp>/{step}.log next to the repo root.
# Final summary is printed at the end and also saved to summary.txt.

set -euo pipefail

# ── tunables (override via env: EPOCHS=10 bash scripts/p0_3_ablation.sh) ───
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"

# ── paths ─────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/p0_3/${TS}"
mkdir -p "$LOGDIR" checkpoints

DATA_ROOT="datasets/drone-racing-dataset/data"
SAM_DIR="${DATA_ROOT}/sam"
SAM_MASKED_DIR="${DATA_ROOT}/sam_masked"

# All commands run inside the gate-detection conda env without sourcing it.
PY="/home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output python"

echo "=== P0-3 ablation @ ${TS} ==============================================="
echo "Repo   : $REPO"
echo "Epochs : $EPOCHS"
echo "Seed   : $SEED"
echo "Log dir: $LOGDIR"
echo

# ── 1) build_sam_dataset ──────────────────────────────────────────────────
if [[ -d "$SAM_DIR/train" && -d "$SAM_DIR/val" && -d "$SAM_DIR/test" ]]; then
  echo "[skip] step 1 (build_sam_dataset): $SAM_DIR/{train,val,test} already exist"
else
  echo "[run]  step 1: build_sam_dataset.py"
  $PY scripts/build_sam_dataset.py 2>&1 | tee "$LOGDIR/01_build_sam.log"
fi

# ── 2) generate_sam_masks ─────────────────────────────────────────────────
# Heuristic for "already done": >50 *_mask.png files in train/.
n_masks=$(find "$SAM_MASKED_DIR/train" -maxdepth 1 -name '*_mask.png' 2>/dev/null | wc -l | tr -d ' ' || echo 0)
if [[ "$n_masks" -gt 50 ]]; then
  echo "[skip] step 2 (generate_sam_masks): $SAM_MASKED_DIR/train already has $n_masks masks"
else
  echo "[run]  step 2: generate_sam_masks.py (~30 min on RTX 5090)"
  $PY scripts/generate_sam_masks.py 2>&1 | tee "$LOGDIR/02_generate_sam.log"
fi

# ── 3-5) the three training cells ─────────────────────────────────────────
run_cell () {
  local label="$1" config="$2" logfile="$3"
  echo "[run]  $label : $config (epochs=$EPOCHS, seed=$SEED)"
  $PY scripts/train.py --config "$config" --epochs "$EPOCHS" --seed "$SEED" \
      2>&1 | tee "$LOGDIR/$logfile"
}

run_cell "step 3 cell A (gatenet baseline)"            "configs/default.yaml"                       "03_train_default.log"
run_cell "step 4 cell B (deeplab + imagenet, no norm)" "configs/deeplabv3plus_imagenet_nonorm.yaml" "04_train_deeplab_nonorm.log"
run_cell "step 5 cell C (deeplab + imagenet + norm)"   "configs/deeplabv3plus_imagenet.yaml"        "05_train_deeplab_norm.log"

# ── 6) summary ────────────────────────────────────────────────────────────
SUMMARY="$LOGDIR/summary.txt"
{
  echo "=== P0-3 ablation summary @ ${TS} ==="
  echo "Epochs: $EPOCHS  Seed: $SEED"
  echo
  for tag in default deeplab_nonorm deeplab_norm; do
    log="$LOGDIR/0?_train_${tag}.log"
    line="$(grep -E 'best_(gatenet|deeplabv3plus).*val_iou=' $log 2>/dev/null | tail -1 || true)"
    if [[ -n "$line" ]]; then
      echo "$tag: $line"
    else
      echo "$tag: (no best-checkpoint line in $log)"
    fi
  done
} | tee "$SUMMARY"

echo
echo "=== done. Full logs in $LOGDIR/ ==="
