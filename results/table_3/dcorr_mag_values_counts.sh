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
model_nos=("7655991_0")

# Get array lengths
NUM_Z=${#z_lims[@]}
NUM_MODELS=${#model_nos[@]}


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

python ./dcorr_mag_values_counts.py --z_lim "$z_lim" --model_no "$model_no"

echo "Job ID: $SLURM_JOB_ID Completed"