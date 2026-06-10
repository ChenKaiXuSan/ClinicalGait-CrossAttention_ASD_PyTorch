#!/bin/bash

# Shared Pegasus environment bootstrap for training jobs.
# Override when needed:
#   ASD_CONDA_ENV=/path/to/env qsub pegasus/run_train_3dcnn.sh

ASD_CONDA_ENV="${ASD_CONDA_ENV:-/home/SKIING/chenkaixu/miniconda3/envs/asd}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}-${PBS_JOBID:-manual}}"

mkdir -p "${MPLCONFIGDIR}"

module load intelpython/2022.3.1

if [[ -z "${CONDA_PREFIX:-}" || ! -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Could not find conda.sh after loading intelpython/2022.3.1" >&2
    echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}" >&2
    exit 1
fi

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true

if [[ ! -x "${ASD_CONDA_ENV}/bin/python" ]]; then
    echo "ERROR: Conda environment has no executable python: ${ASD_CONDA_ENV}" >&2
    echo "Available conda environments:" >&2
    conda info --envs >&2 || true
    exit 1
fi

conda activate "${ASD_CONDA_ENV}" || {
    echo "ERROR: Failed to activate conda environment: ${ASD_CONDA_ENV}" >&2
    conda info --envs >&2 || true
    exit 1
}

hash -r

echo "Python version: $(python --version)"
echo "Python executable: $(which python)"
