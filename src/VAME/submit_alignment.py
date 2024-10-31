#!/bin/bash
#SBATCH --job-name=grooming8k_egocentric_alignment
#SBATCH --time=01:00:00
#SBATCH --array=0-26 # 27 parallel jobs (50 videos per job)
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G


# Create sbatch_logs directory if it doesn't exist in the project root directory/logs
LOG_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/logs/sbatch_logs"
mkdir -p "$LOG_DIR"

# output and error paths to point to the log directory
#SBATCH --output="$LOG_DIR/grooming8k_egocentric_alignment_%A_%a.out"
#SBATCH --error="$LOG_DIR/grooming8k_egocentric_alignment_%A_%a.err"


# Calculate the starting video index for the current job
python3 -u egocentric_alignment.py $SLURM_ARRAY_TASK_ID

# For custom videos
#python3 -u egocentric_alignment.py 0 
