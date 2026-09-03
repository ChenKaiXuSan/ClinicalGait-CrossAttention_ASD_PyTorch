#!/bin/bash
#PBS -A HP260146                       # HP260146 allocation (gen_S)
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1                   # gen_S default GPU=0; request 1 GPU explicitly
#PBS -l elapstim_req=2:00:00
#PBS -N heldout_nec_L3                 # §5.7 necessity (ROI perturbation) on the clean single-L3 main model
#PBS -o logs/pegasus/heldout_nec_L3_out.log
#PBS -e logs/pegasus/heldout_nec_L3_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# Main model switched to single-L3 (narrative B). Re-run the test-time ROI
# perturbation (real / shuffled / zero) on its clean checkpoints. Writes
# analysis/attention_perturbation.md (the multi-[0,1] result is kept as
# analysis/attention_perturbation_multi01.md).
python -m analysis.attention_perturbation data.root_path=${root_path} \
    data.heldout_test=True \
    +pert.tag=heldout_pose_single_L3 +pert.per_fold=750 \
    data.num_workers=${workers}

echo "===== done: analysis/attention_perturbation.md (single-L3) ====="
