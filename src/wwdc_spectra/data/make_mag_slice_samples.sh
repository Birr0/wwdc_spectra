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

# MAG values
mags=(-21.0 -21.2 -21.4 -21.6 -21.8 -22.0 -22.5 -23.0 -24.0)
mag=${mags[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID mags: $mag"
python ./make_mag_slice_samples.py --M_upper $mag
echo "Job ID: $SLURM_JOB_ID Completed"