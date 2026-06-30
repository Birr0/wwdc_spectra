#!/bin/bash
#SBATCH --job-name=make_vol_limited_samples
#SBATCH --output=make_vol_limited_samples_%j.out
#SBATCH --error=make_vol_limited_samples_%j.err
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

z_lims=("0.050" "0.075" "0.100" "0.125" "0.150" "0.175" "0.200" "0.225" "0.250")
z_lim=${z_lims[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID z_lim: $z_lim"
python ./save_vol_limited_samples.py --z_lim "$z_lim"
echo "Job ID: $SLURM_JOB_ID Completed"