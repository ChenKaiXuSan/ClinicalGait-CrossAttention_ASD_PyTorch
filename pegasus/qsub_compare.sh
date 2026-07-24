#!/bin/bash
set -euo pipefail

# ============================================================================
# Batch driver for the ASD experiment matrix (3-fold, one fold per node).
# Default is DRY-RUN. Add --run to actually submit.
#
#   ./qsub_compare.sh prebuild            # 1) build the fold cache ONCE (required)
#   ./qsub_compare.sh main                # 2) preview main-result jobs
#   ./qsub_compare.sh main --run          #    submit them
#   ./qsub_compare.sh all --run           #    submit the whole matrix (78 sub-jobs)
#
# See pegasus/EXPERIMENTS.md for the full matrix and recommended order.
# ============================================================================

cd "$(dirname "$0")/.."   # repo root

# Must match the root_path used inside every run_train_*.sh
ROOT_PATH="/work/SKIING/chenkaixu/data/asd_dataset"
# Fold cache written by project.prepare_folds (class_num=3, sampling=over, K=3)
CACHE_INDEX="${ROOT_PATH}/pose_attn_map_dataset/index_mapping/3/over_K3/index.json"

MODE="${1:-help}"
ACTION="${2:-dry-run}"

usage() {
    cat <<'EOF'
Usage:
  ./pegasus/qsub_compare.sh <mode> [dry-run|--run]

Modes:
  prebuild   Build the 3-fold cache once (run BEFORE submitting; ignores --run,
             always executes). Parallel fold jobs otherwise race on cache build.
  main       Main result: PoseGated full + RGB baseline            (2 scripts,  6 sub-jobs)
  baseline   B1 RGB-only                                          (1 script,   3 sub-jobs)
  fusion     A1 early(add/mul/concat) + SE + cross-attn            (3 scripts, 33 sub-jobs)
  layers     A5 single-layer + multi-prefix(P1-P3)                (2 scripts, 24 sub-jobs)
  ablation   A2 bias(0,-1) + A3 no-sidehead + A4 no-bg/no-tmp      (5 scripts, 15 sub-jobs)
  all        Everything (12 scripts, 78 sub-jobs)
EOF
}

# --- script groups (must match files in pegasus/) ---------------------------
main_scripts=(
    pegasus/run_train_pose_gated_best.sh
    pegasus/run_train_3dcnn.sh
)
baseline_scripts=(
    pegasus/run_train_3dcnn.sh
)
fusion_scripts=(
    pegasus/run_train_early_fuse.sh
    pegasus/run_train_se_atn.sh
    pegasus/run_train_cross_atn.sh
)
layers_scripts=(
    pegasus/run_train_pose_atn_single.sh
    pegasus/run_train_pose_atn_multi.sh
)
ablation_scripts=(
    pegasus/run_train_pose_gated_bias0.sh
    pegasus/run_train_pose_gated_bias_neg1.sh
    pegasus/run_train_pose_gated_nosidehead.sh
    pegasus/run_train_pose_gated_nobgloss.sh
    pegasus/run_train_pose_gated_notmploss.sh
)
# 'all' = the 13 unique scripts (pose_gated_best + baseline + fusion + layers + ablation)
all_scripts=(
    pegasus/run_train_pose_gated_best.sh
    "${baseline_scripts[@]}"
    "${fusion_scripts[@]}"
    "${layers_scripts[@]}"
    "${ablation_scripts[@]}"
)

# --- prebuild: build fold cache once ----------------------------------------
if [[ "$MODE" == "prebuild" ]]; then
    echo "Building 3-fold cache (class_num=3, sampling=over, K=3)..."
    # conda activate / module load internals trip errexit, nounset AND pipefail;
    # relax all three around the source. prepare_folds below runs under strict mode.
    set +euo pipefail
    source pegasus/setup_env.sh
    set -euo pipefail
    python -m project.prepare_folds data.root_path="${ROOT_PATH}"
    echo "Cache ready: ${CACHE_INDEX}"
    exit 0
fi

case "$MODE" in
    main)     scripts=("${main_scripts[@]}") ;;
    baseline) scripts=("${baseline_scripts[@]}") ;;
    fusion)   scripts=("${fusion_scripts[@]}") ;;
    layers)   scripts=("${layers_scripts[@]}") ;;
    ablation) scripts=("${ablation_scripts[@]}") ;;
    all)      scripts=("${all_scripts[@]}") ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown mode: $MODE" >&2; usage; exit 2 ;;
esac

if [[ "$ACTION" != "dry-run" && "$ACTION" != "--run" ]]; then
    echo "Unknown action: $ACTION" >&2; usage; exit 2
fi

# --- guard: fold cache must exist before parallel fold jobs are submitted ----
if [[ "$ACTION" == "--run" && ! -f "$CACHE_INDEX" ]]; then
    echo "ERROR: fold cache not found: $CACHE_INDEX" >&2
    echo "Run './pegasus/qsub_compare.sh prebuild' first (parallel fold jobs would" >&2
    echo "otherwise all race to build it)." >&2
    exit 1
fi

echo "Mode: $MODE    Action: $ACTION"
echo
for script in "${scripts[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "Missing script: $script" >&2; exit 1
    fi
    if [[ "$ACTION" == "--run" ]]; then
        echo "qsub $script"
        qsub "$script"
        sleep 1
    else
        echo "[dry-run] qsub $script"
    fi
done

if [[ "$ACTION" != "--run" ]]; then
    echo
    echo "Dry run only. Re-run with --run to submit."
fi
