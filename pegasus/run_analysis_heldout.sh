#!/bin/bash
#PBS -A HP260146                       # HP260146 allocation (SKIING gpu budget busy with 892635)
#PBS -q gen_S                          # HP260146 -> general gen_S queue
#PBS -b 1
#PBS -l gpunum_job=1                   # gen_S default GPU=0; request 1 GPU explicitly
#PBS -l elapstim_req=4:00:00
#PBS -N heldout_analysis               # §5.6 alignment + §5.7 necessity on CLEAN held-out ckpts
#PBS -o logs/pegasus/heldout_analysis_out.log
#PBS -e logs/pegasus/heldout_analysis_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# These are TEST-TIME analyses (no retraining) on the CLEAN held-out checkpoints.
# data.heldout_test=True is REQUIRED so DefineCrossValidation rebuilds the same
# patient-disjoint test set the checkpoints were trained/selected on.

echo "===== §5.7 necessity: ROI perturbation on clean main model multi-[0,1] ====="
python -m analysis.attention_perturbation data.root_path=${root_path} \
    data.heldout_test=True \
    +pert.tag=heldout_pose_multi01 +pert.per_fold=750 \
    data.num_workers=${workers}

echo "===== §5.6 interpretability: across-depth alignment on clean multi-[0,1,2,3] ====="
python -m analysis.attention_alignment data.root_path=${root_path} \
    data.heldout_test=True \
    +align.run_glob='logs/train/heldout_pose_multi_P3_f*' \
    data.num_workers=${workers}

echo "===== done. outputs: analysis/attention_perturbation.md, analysis/alignment_out/ ====="
