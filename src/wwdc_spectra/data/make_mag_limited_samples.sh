#!/bin/bash
#SBATCH --job-name=make_mag_limited_samples
#SBATCH --output=make_mag_limited_samples_%j.out
#SBATCH --error=make_mag_limited_samples_%j.err
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

z_idxs=(99 104 109 114 119 124 129 134 139 144 149) #159 169 179 189 199)
z_idx=${z_idxs[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID z_idx: $z_idx"
python ./make_mag_limited_samples.py --z_idx $z_idx
echo "Job ID: $SLURM_JOB_ID Completed"