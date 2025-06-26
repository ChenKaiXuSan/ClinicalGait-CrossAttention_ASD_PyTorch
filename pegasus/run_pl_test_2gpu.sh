#!/bin/bash
#PBS -A SKIING
#PBS -q gen_S
#PBS -l elapstim_req=01:00:00
#PBS -l gpunum_job=2
#PBS -N pl_test_2gpu
#PBS -o logs/pl_test_out.log
#PBS -e logs/pl_test_err.log
#PBS -b 2
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda11.8.0
#PBS -v OMP_NUM_THREADS=48

cd /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
source /home/SKIING/chenkaixu/code/med_atn/bin/activate

python tests/pl_test_mnist.py