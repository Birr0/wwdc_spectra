#!/bin/bash
#SBATCH --job-name=make_sersic
#SBATCH --output=make_sersic_%j.out
#SBATCH --error=make_sersic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # total RAM for the job
#SBATCH --time=24:00:00            # hh:mm:ss

# Optional: print job info
echo "Running on $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Job ID: $SLURM_JOB_ID"

# Run your program
source $DATA/venvs/wwdc_spectra/bin/activate

echo "Task ID: $SLURM_ARRAY_TASK_ID"
python ./make_sersic.py
echo "Job ID: $SLURM_JOB_ID Completed"