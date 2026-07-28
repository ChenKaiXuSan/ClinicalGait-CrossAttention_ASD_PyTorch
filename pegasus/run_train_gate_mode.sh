#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N gate_mode_train               # #6 gating-vs-plain-injection ablation
#PBS -t 0-5                           # array: 2 configs × 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_gate_mode_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_gate_mode_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 3 ))

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} → cfg ${cfg}, fold ${fold} / 3-fold  (gate_mode ablation)"

# Same architecture / layers / inputs as the main result (PoseGated multi-[0,1]);
# ONLY the fusion operator changes, isolating the learned gate from the shallow
# injection. The 'gated' reference is the main result (94.8) — not re-run here.
#   cfg 0: add   -> plain additive injection  (no gate)
#   cfg 1: fixed -> frozen 0.5 mix           (gate hard-set, no learning)
case ${cfg} in
  0) gmode=add ;;
  1) gmode=fixed ;;
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=1 model.ablation_study=multi \
    model.gate_mode=${gmode} \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=pose_gated_gate_${gmode}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# #6 gating mechanism ablation. All three share fusion at [0,1], side heads, and
# the doctor ROI input; only the mix operator differs:
#   gated (main, 94.8) | add | fixed.
# If add/fixed match gated -> the gain is from shallow injection, not the gate;
# if they drop -> the learned gate itself contributes.
###############################################################################
