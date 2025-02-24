#!/bin/bash
#SBATCH --job-name=slurm_few_shot_debug          # Job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeremiaslcc@gmail.com
#SBATCH --partition=multi
#SBATCH --ntasks=1                          # Number of tasks (single task)
#SBATCH --cpus-per-task=10                  # Use 10 CPU cores
#SBATCH --gpus=2                            # Use both GPUs
#SBATCH --time=01:00:00                     # Maximum runtime (hh:mm:ss)

echo "Running on node $(hostname)"
echo "Available GPUs:"
nvidia-smi -L

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deep-k-correct

# Debugging specific combination
TARGET_VARIABLE="KCORR01_SDSS_G"
Z_MIN=0.3031
Z_MAX=0.814

echo "Running debug for target variable: $TARGET_VARIABLE, z_min=$Z_MIN, z_max=$Z_MAX"
python train_few_shot.py model=residual_mlp data.target_variable=$TARGET_VARIABLE data.z_min=$Z_MIN data.z_max=$Z_MAX training.devices=-1