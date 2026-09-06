#!/bin/bash
#PBS -A HP260146                       # HP260146 allocation (gen_S)
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1                   # gen_S default GPU=0; request 1 GPU explicitly
#PBS -l elapstim_req=2:00:00
#PBS -N heldout_decrule                # validation-tuned decision rules (post-hoc, no re-training)
#PBS -o logs/pegasus/heldout_decrule_out.log
#PBS -e logs/pegasus/heldout_decrule_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# Runs the single-L3 checkpoints on the INNER VALIDATION split, tunes (a) 3-class
# prior-correction weights and (b) the binary P(ASD) threshold on val only, and
# applies them to the saved TEST predictions. Writes analysis/decision_rule_heldout.md.
python -m analysis.decision_rule_heldout data.root_path=${root_path} \
    data.heldout_test=True +dr.tag=heldout_pose_single_L3 \
    data.num_workers=${workers}

echo "===== done: analysis/decision_rule_heldout.md ====="
