#!/bin/bash
#SBATCH --job-name=grooming8k_setup
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Create sbatch_logs directory if it doesn't exist in the project root directory/logs
LOG_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/logs/sbatch_logs"
mkdir -p "$LOG_DIR"

# output and error paths to point to the log directory
#SBATCH --output="$LOG_DIR/grooming8k_setup_%A_%a.out"
#SBATCH --error="$LOG_DIR/grooming8k_setup_%A_%a.err"

# Run the script
python3 -u project_setup.py
