#!/bin/bash
#SBATCH --job-name=dcorr_mag_values
#SBATCH --output=dcorr_mag_values_%j.out
#SBATCH --error=dcorr_mag_values_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 
#SBATCH --time=24:00:00            

# Define your parameter arrays
z_lims=("0.100" "0.105" "0.110" "0.115" "0.120" "0.125" "0.130" "0.135" "0.140" "0.145" "0.150")
model_nos=("7655991_0") # "7655991_1" "7655991_2") #   # <-- Add all your model numbers here

# Get array lengths
NUM_Z=${#z_lims[@]}
NUM_MODELS=${#model_nos[@]}

# CRITICAL: Calculate total tasks dynamically
# E.g., 11 z_lims * 3 models = 33 tasks -> set #SBATCH --array=0-32 manually above if needed.
# Launch job with: sbatch --array=0-$((NUM_Z * NUM_MODELS - 1)) script.sh

# Map the 1D SLURM_ARRAY_TASK_ID to 2D indices
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_MODELS ))
Z_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_MODELS ))

# Extract values
z_lim=${z_lims[$Z_IDX]}
model_no=${model_nos[$MODEL_IDX]}

echo "Running on $SLURM_NODELIST"
echo "CPUs: $SLURM_CORES_PER_NODE"
echo "Job ID: $SLURM_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Evaluating Model: $model_no | z_lim: $z_lim"

# Activate environment and run
source $DATA/venvs/wwdc_spectra/bin/activate

python ./dcorr_mag_values_v2.py --z_lim "$z_lim" --model_no "$model_no"

echo "Job ID: $SLURM_JOB_ID Completed"