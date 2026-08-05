#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N heldout_ablation              # A2/A3/A4/A6 gate-bias, loss, gate-mechanism — CLEAN held-out
#PBS -t 0-20                          # array: 7 configs x 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_heldout_ab_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_heldout_ab_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# --- gate/loss/gate-mechanism ablations, re-anchored on the CLEAN best config ---
# Old paper anchored these on the all-stage "full" model (its "full is sub-optimal"
# story). Under the clean protocol that story is gone: the reference is now the
# single-L3 PoseGated (b=2.0, all losses, learned gate) = heldout_pose_single_L3
# (already trained, 69.2% clip). Each config below changes ONE factor from it.
#
# NOTE: single-L3 is the strongest of the configs run so far; if the parallel
# location sweep (run_train_heldout_location.sh) crowns a different point, a few
# of these may be re-anchored — the L3 ablation stays a valid strong-config study.
#
# cfg -> (override, tag)   [all: fuse=pose_atn, single, fusion_layers=3, heldout]
# extra is a bash ARRAY expanded as "${extra[@]}" so the loss.selection=[...] value
# is passed as ONE literal arg (no word-splitting, no [ ] glob expansion).
case ${cfg} in
  0) extra=(model.gate_init_bias=0.0);                   tag=heldout_ab_bias0 ;;     # A2b
  1) extra=(model.gate_init_bias=-1.0);                  tag=heldout_ab_biasneg1 ;;  # A2c
  2) extra=('loss.selection=["cls","attn_loss","tmp"]'); tag=heldout_ab_nobg ;;      # A4a  -bg
  3) extra=('loss.selection=["cls","attn_loss","bg"]');  tag=heldout_ab_notmp ;;     # A4b  -tmp
  4) extra=(model.use_side_heads=False);                 tag=heldout_ab_noside ;;    # A3   -side head
  5) extra=(model.gate_mode=add);                        tag=heldout_ab_gateadd ;;   # A6   plain add
  6) extra=(model.gate_mode=fixed);                      tag=heldout_ab_gatefixed ;; # A6   fixed 0.5 mix
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} -> ${tag} [${extra[*]}], fold ${fold} (CLEAN held-out, base=single-L3)"

python -m project.train data.root_path=${root_path} \
    data.heldout_test=True \
    model.fuse_method=pose_atn \
    model.fusion_layers=3 model.ablation_study=single \
    "${extra[@]}" \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# Reference (b=2.0, all losses, gated) = heldout_pose_single_L3 (not re-run here).
# A2: gate bias {0.0, -1.0}        A3: no side head
# A4: {-bg, -tmp} loss             A6: gate mechanism {add, fixed} vs gated
# Aggregate to clip level with analysis/aggregate_heldout.py (extend HELDOUT_METHODS).
###############################################################################
