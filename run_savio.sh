#!/bin/bash
#SBATCH --job-name=cs289_pca
#SBATCH --account=ic_engin296f25
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A40:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -euo pipefail

mkdir -p logs results/features results/figures results/models

cd /global/scratch/users/ahmostafa/CS289

PY=$HOME/.conda/envs/cs289/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi || true
$PY -c "import torch, torchvision, sklearn, medmnist, numpy, scipy, matplotlib, tqdm, datasets; print('imports OK')"
$PY -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

$PY run.py

echo "=== done ==="
