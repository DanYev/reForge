#!/bin/bash
#SBATCH --time=0-01:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
##SBATCH --gres=gpu:1
#SBATCH -o slurm_jobs/output.%A.%a.out
#SBATCH -e slurm_jobs/error.%A.%a.err

# Get the Python script (first argument)
PYSCRIPT="$1"
shift  # Remove the script name from arguments

# Echo all job information to stderr so it appears in the error log file
echo "Job started at: $(date)" >&2
echo "Running script: $PYSCRIPT" >&2
echo "Working directory: $(pwd)" >&2
echo "Arguments: $@" >&2
echo "Environment variables:" >&2
echo "  SLURM_JOB_ID: $SLURM_JOB_ID" >&2
echo "  SLURM_JOB_NAME: $SLURM_JOB_NAME" >&2
echo "  HOSTNAME: $HOSTNAME" >&2
echo "----------------------------------------" >&2

# Run the Python script
python "$PYSCRIPT" "$@"

echo "----------------------------------------" >&2
echo "Job completed at: $(date)" >&2
