#!/bin/bash
#SBATCH --job-name=z_mlp_predictor
#SBATCH --output=mlp_%A_%a.out
#SBATCH --error=mlp_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-17

echo "Running on $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

source $DATA/venvs/wwdc_spectra/bin/activate

# Define parameters
embed_types=("orig" "cond" "uncond")
base_models=("spender_I" "spender_II") # Note: removed the comma

# Map task ID → (idx, base_model, embed_type)
# Total combinations per idx = 3 embed_types * 2 base_models = 6
idx=$((SLURM_ARRAY_TASK_ID / 6))
base_idx=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
embed_idx=$((SLURM_ARRAY_TASK_ID % 3))

embed_type=${embed_types[$embed_idx]}
base_model=${base_models[$base_idx]}

echo "Running idx=$idx base_model=$base_model embed_type=$embed_type"
python z_mlp_prediction_v2.py --idx "$idx" --embed_type "$embed_type" --base_model "$base_model"
echo "Job completed"