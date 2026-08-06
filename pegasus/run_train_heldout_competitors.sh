#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N heldout_competitors           # A1 main-table fusion competitors — CLEAN held-out protocol
#PBS -t 0-35                          # array: 12 configs x 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_heldout_cmp_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_heldout_cmp_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# --- Main-table fusion competitors re-run under the CLEAN held-out protocol ------
# So PoseGated single-L3 (69.2 clip) is compared against competitors on the SAME
# leak-free protocol, not against the archived leaky (5-fold) numbers. Concat is
# already clean (heldout_early_concat) and excluded here.
#   fmethod : model.fuse_method       extra : method-specific overrides (bash array,
#             expanded as "${extra[@]}" so [3,4] lists survive shell globbing)
# SE uses int fusion_layers as a prefix map (0->[0] ... 4->[0..4]); cross takes an
# explicit layer list; early/late take no fusion_layers.
case ${cfg} in
  0)  fmethod=add;       extra=();                          tag=heldout_early_add ;;
  1)  fmethod=mul;       extra=();                          tag=heldout_early_mul ;;
  2)  fmethod=avg;       extra=();                          tag=heldout_early_avg ;;
  3)  fmethod=late;      extra=();                          tag=heldout_late ;;
  4)  fmethod=se_atn;    extra=(model.fusion_layers=0);     tag=heldout_se_prefix0 ;;
  5)  fmethod=se_atn;    extra=(model.fusion_layers=1);     tag=heldout_se_prefix1 ;;
  6)  fmethod=se_atn;    extra=(model.fusion_layers=2);     tag=heldout_se_prefix2 ;;
  7)  fmethod=se_atn;    extra=(model.fusion_layers=3);     tag=heldout_se_prefix3 ;;
  8)  fmethod=se_atn;    extra=(model.fusion_layers=4);     tag=heldout_se_prefix4 ;;
  9)  fmethod=cross_atn; extra=('model.fusion_layers=[3]');   tag=heldout_cross_L3 ;;
  10) fmethod=cross_atn; extra=('model.fusion_layers=[4]');   tag=heldout_cross_L4 ;;
  11) fmethod=cross_atn; extra=('model.fusion_layers=[3,4]'); tag=heldout_cross_L34 ;;
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} -> ${tag} (fuse=${fmethod} ${extra[*]}), fold ${fold} (CLEAN held-out)"

python -m project.train data.root_path=${root_path} \
    data.heldout_test=True \
    model.fuse_method=${fmethod} \
    "${extra[@]}" \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# 12 configs: early {add,mul,avg}, late, SE {prefix0..4}, cross {L3,L4,L34}.
# Concat already clean (heldout_early_concat). Aggregate all to clip level with
# analysis/aggregate_heldout.py (extend HELDOUT_METHODS). SE row = best prefix;
# cross row = best of {L3,L4,L34}, matching the main table's "(best)"/"(layer 4)".
###############################################################################
