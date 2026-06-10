#!/bin/bash
set -euo pipefail

# Batch submit Pegasus comparison experiments.
# Default is dry-run. Add --run to actually submit jobs.

cd "$(dirname "$0")"

MODE="${1:-all}"
ACTION="${2:-dry-run}"

usage() {
    cat <<'EOF'
Usage:
  ./qsub_compare.sh [main|fusion|ablation|all] [dry-run|--run]

Examples:
  ./qsub_compare.sh
  ./qsub_compare.sh main
  ./qsub_compare.sh all --run

Modes:
  main      baseline + final/best PoseGated
  fusion    fusion-layer/method comparison
  ablation  bias / side-head / loss ablations
  all       all comparison training scripts
EOF
}

main_scripts=(
    run_train_3dcnn.sh
    run_train_pose_gated_best.sh
)

fusion_scripts=(
    run_train_pose_atn_single.sh
    run_train_pose_atn_multi.sh
    run_train_se_atn_single.sh
)

ablation_scripts=(
    run_train_pose_gated_bias0.sh
    run_train_pose_gated_bias_neg1.sh
    run_train_pose_gated_nosidehead.sh
    run_train_pose_gated_nobgloss.sh
    run_train_pose_gated_notmploss.sh
)

case "$MODE" in
    main)
        scripts=("${main_scripts[@]}")
        ;;
    fusion)
        scripts=("${fusion_scripts[@]}")
        ;;
    ablation)
        scripts=("${ablation_scripts[@]}")
        ;;
    all)
        scripts=("${main_scripts[@]}" "${fusion_scripts[@]}" "${ablation_scripts[@]}")
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        exit 2
        ;;
esac

if [[ "$ACTION" != "dry-run" && "$ACTION" != "--run" ]]; then
    echo "Unknown action: $ACTION" >&2
    usage
    exit 2
fi

echo "Mode: $MODE"
echo "Action: $ACTION"
echo

for script in "${scripts[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "Missing script: $script" >&2
        exit 1
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
