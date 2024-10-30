#!/bin/bash
#SBATCH --job-name=vame_infer8_groom
#SBATCH --time=01:00:00
#SBATCH --array=0-133
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Create sbatch_logs directory if it doesn't exist in the project root directory/logs
LOG_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/logs/sbatch_logs"
mkdir -p "$LOG_DIR"

# output and error paths to point to the log directory
#SBATCH --output="$LOG_DIR/vame_infer8_groom_%A_%a.out"
#SBATCH --error="$LOG_DIR/vame_infer8_groom_%A_%a.err"

# Calculate the starting video index for the current job
START=$((SLURM_ARRAY_TASK_ID * 10))

# Run the script
python3 -u vame_infer_cpu.py $START 10
