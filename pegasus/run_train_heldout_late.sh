#!/bin/bash
#PBS -A HP260146                       # HP260146 allocation (gen_S; SKIING gpu budget is tight)
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1                   # gen_S default GPU=0; request 1 GPU explicitly
#PBS -l elapstim_req=24:00:00
#PBS -N heldout_late                   # A1 late (score-level) fusion — CLEAN held-out, batch 8
#PBS -t 0-2                            # array: fold 0-2  (SUBREQNO = fold)
#PBS -o logs/pegasus/train_heldout_late_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_heldout_late_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# Late (score-level) fusion trains TWO full backbones (RGB + attention map) and
# fuses at the logit level, so its memory footprint is ~2x the single-backbone
# competitors. At the default batch_size=16 it was hard-killed (OOM) on all three
# folds in the competitor array (892635, cfg 3). Re-run with batch_size=8 to fill
# the missing tab:main row. Same tag as before; the new run writes its own
# <date>/<time> dir, and the aggregator takes the most recent one with preds.
echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} -> heldout_late fold ${fold} (CLEAN held-out, batch_size=8)"

python -m project.train data.root_path=${root_path} \
    data.heldout_test=True \
    model.fuse_method=late \
    data.batch_size=8 \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=heldout_late_f${fold} \
    data.num_workers=${workers}
