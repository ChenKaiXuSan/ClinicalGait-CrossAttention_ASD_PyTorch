#!/bin/bash
#PBS -A HP260146                       # HP260146 allocation (gen_S)
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1                   # gen_S default GPU=0; request 1 GPU explicitly
#PBS -l elapstim_req=2:00:00
#PBS -N fig_qual_cache
#PBS -o logs/pegasus/fig_qual_cache_out.log
#PBS -e logs/pegasus/fig_qual_cache_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# Qualitative figure: (a) Grad-CAM across methods (METHODS in make_method_cam_fig.py,
# now the clean heldout_* tags) and (b) supervised side-head attention of the clean
# deep variant (SUP_TAG=heldout_pose_multi_P3). fold-0 checkpoints, as in the caption.
# data.heldout_test=True is REQUIRED so the rebuilt fold-0 test set matches the
# split these checkpoints were trained/selected on.
export COMB_CACHE=analysis/figures_out/comb_cache_fold0.pkl
python -m analysis.make_combined_fig data.root_path=${root_path} \
    data.heldout_test=True +qual.fold=0 \
    data.num_workers=${workers}

echo "===== done: paper/figures/fig_qual_combined.pdf ====="
