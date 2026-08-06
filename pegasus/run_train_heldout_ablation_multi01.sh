#!/bin/bash
#PBS -A HP260146                       # run on the HP260146 allocation (SKIING gpu budget was exhausted)
#PBS -q gen_S                          # HP260146 can't use the `gpu` queue; it uses the general gen_S queue
#PBS -b 1
#PBS -l gpunum_job=1                   # gen_S default GPU = 0, so GPUs MUST be requested explicitly
                                       # (proven directive: matches HP260146 user's freeman GPU job)
#PBS -l elapstim_req=24:00:00
#PBS -N heldout_abl_m01               # A2/A3/A4/A6 ablations re-anchored on multi-[0,1] — CLEAN held-out
#PBS -t 0-20                          # array: 7 configs x 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_heldout_abm_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_heldout_abm_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# --- gate/loss/gate-mechanism ablations anchored on multi-[0,1] -----------------
# Companion to run_train_heldout_ablation.sh (which anchors on single-L3). Here the
# reference is multi-[0,1] PoseGated (b=2.0, all losses, learned gate) =
# heldout_pose_multi01 (already trained). Each config changes ONE factor from it.
# Tags use the heldout_abm_ prefix to stay distinct from the single-L3 (heldout_ab_).
#
# cfg -> (override, tag)   [all: fuse=pose_atn, multi, fusion_layers=1, heldout]
case ${cfg} in
  0) extra=(model.gate_init_bias=0.0);                   tag=heldout_abm_bias0 ;;     # A2b
  1) extra=(model.gate_init_bias=-1.0);                  tag=heldout_abm_biasneg1 ;;  # A2c
  2) extra=('loss.selection=["cls","attn_loss","tmp"]'); tag=heldout_abm_nobg ;;      # A4a  -bg
  3) extra=('loss.selection=["cls","attn_loss","bg"]');  tag=heldout_abm_notmp ;;     # A4b  -tmp
  4) extra=(model.use_side_heads=False);                 tag=heldout_abm_noside ;;    # A3   -side head
  5) extra=(model.gate_mode=add);                        tag=heldout_abm_gateadd ;;   # A6   plain add
  6) extra=(model.gate_mode=fixed);                      tag=heldout_abm_gatefixed ;; # A6   fixed 0.5 mix
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} -> ${tag} [${extra[*]}], fold ${fold} (CLEAN held-out, base=multi-[0,1])"

python -m project.train data.root_path=${root_path} \
    data.heldout_test=True \
    model.fuse_method=pose_atn \
    model.fusion_layers=1 model.ablation_study=multi \
    "${extra[@]}" \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# Reference (b=2.0, all losses, gated) = heldout_pose_multi01 (not re-run here).
# A2: gate bias {0.0,-1.0}   A3: no side head   A4: {-bg,-tmp}   A6: {add,fixed}.
# Aggregate to clip level via analysis/aggregate_heldout.py (extend HELDOUT_METHODS).
###############################################################################
