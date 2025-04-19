#!/bin/bash
#SBATCH --job-name=few_shot_with_redshift          # Job name
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

# Load Conda environmsent
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deep-k-correct

# Define constants
MIN_REDSHIFT=0.0
MAX_REDSHIFT=2.0
threshold_r=0.3885
threshold_g1=0.3031
threshold_g2=0.814

# Define target variables and valid pairs
TARGET_VARIABLES=("KCORR01_SDSS_R" "KCORR01_SDSS_G" "KCORR01_SDSS_Z" "ABSMAG01_SDSS_R" "ABSMAG01_SDSS_G" "ABSMAG01_SDSS_Z")
VALID_PAIRS=(
    "K_G_G KCORR01_SDSS_G $MIN_REDSHIFT $threshold_g1"
    "K_G_R KCORR01_SDSS_G $threshold_g1 $threshold_g2"
    "K_G_Z KCORR01_SDSS_G $threshold_g2 $MAX_REDSHIFT"
    "K_R_G KCORR01_SDSS_R $MIN_REDSHIFT $threshold_r"
    "K_R_R KCORR01_SDSS_R $threshold_r $MAX_REDSHIFT"
    "K_Z_Z KCORR01_SDSS_Z $MIN_REDSHIFT $MAX_REDSHIFT"
    "absmag_G_G ABSMAG01_SDSS_G $MIN_REDSHIFT $threshold_g1"
    "absmag_G_R ABSMAG01_SDSS_G $threshold_g1 $threshold_g2"
    "absmag_G_Z ABSMAG01_SDSS_G $threshold_g2 $MAX_REDSHIFT"
    "absmag_R_R ABSMAG01_SDSS_R $MIN_REDSHIFT $threshold_r"
    "absmag_R_G ABSMAG01_SDSS_R $threshold_r $MAX_REDSHIFT"
    "absmag_Z_Z ABSMAG01_SDSS_Z $MIN_REDSHIFT $MAX_REDSHIFT"
)

# Run training jobs for each target variable with z_min=0 and z_max=2
for target in "${TARGET_VARIABLES[@]}"; do
    echo "Running training for target variable: $target with z_min=0 and z_max=2"
    python train_few_shot.py --config-name mlp_with_redshift.yaml data.target_variable=$target data.z_min=0 data.z_max=2 training.devices=-1 logging.experiment_tag=mlp_with_redshift
done

# Run training jobs for each valid pair
for pair in "${VALID_PAIRS[@]}"; do
    IFS=' ' read -r -a params <<< "$pair"
    name="${params[0]}"
    target="${params[1]}"
    z_min="${params[2]}"
    z_max="${params[3]}"
    echo "Running training for $name with target variable: $target, z_min=$z_min, z_max=$z_max"
    python train_few_shot.py --config-name mlp_with_redshift.yaml data.target_variable=$target data.z_min=$z_min data.z_max=$z_max training.devices=-1 logging.experiment_tag=mlp_with_redshift
done