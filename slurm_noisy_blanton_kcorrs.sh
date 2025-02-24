#!/bin/bash
#SBATCH --job-name=calculate_aclip_embeddings          # Job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeremiaslcc@gmail.com
#SBATCH --partition=multi
#SBATCH --ntasks=1                          # Number of tasks (single task)
#SBATCH --cpus-per-task=1                   # Restrict to a single CPU thread
#SBATCH --gpus=1                            # Assign one GPU
#SBATCH --time=24:00:00                     # Maximum runtime (hh:mm:ss)

echo "Running on node $(hostname)"
echo "Available GPUs:"
nvidia-smi -L

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deep-k-correct

 # Execute your Python script
python blanton_analysis/calculate_blanton_K_corrections.py --astroclip_path data/raw/AstroCLIP --desi_path  data/desi_edr_fluxes.h5 --noise_range 0.05 --save_path data/blanton_kcorrs_005.pickle 
