#!/bin/bash
#SBATCH --job-name=few_shot_architecture_experiment          # Job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeremiaslcc@gmail.com
#SBATCH --partition=multi
#SBATCH --ntasks=1                          # Number of tasks (single task)
#SBATCH --cpus-per-task=10                  # Use 10 CPU cores
#SBATCH --gpus=2                            # Use both GPUs
#SBATCH --time=24:00:00                     # Maximum runtime (hh:mm:ss)

echo "Running on node $(hostname)"
echo "Available GPUs:"
nvidia-smi -L

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deep-k-correct

# Debugging specific combination
TARGET_VARIABLE="KCORR01_SDSS_G"
Z_MIN=0
Z_MAX=0.3031

# Define array of network architectures to test
declare -a architectures=(
    "16"
    "1024"
    "512,512"
    "1024,1024"
    "512,256,128"
    "1024,512,256"
    "1024,512,512,256"
    "10000,7000,4000,2000,1000,500"
)

# Loop through architectures
for arch in "${architectures[@]}"; do
    echo "Testing architecture: $arch"
    python train_few_shot.py \
        data.target_variable=$TARGET_VARIABLE \
        data.z_min=$Z_MIN \
        data.z_max=$Z_MAX \
        training.devices=-1 \
        model=residual_mlp \
        model.n_hidden="[$arch]" \
        logging.experiment_tag="residual_mlp_architecture_experiment"
    
    echo "Completed architecture: $arch"
    echo "----------------------------------------"
done