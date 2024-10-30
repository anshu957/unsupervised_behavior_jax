#!/bin/bash
#SBATCH --job-name=grooming8k_train_model
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --qos=gpu_training
#SBATCH --mem=32G

# Create sbatch_logs directory if it doesn't exist in the project root directory/logs
LOG_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/logs/sbatch_logs"
mkdir -p "$LOG_DIR"

# output and error paths to point to the log directory
#SBATCH --output="$LOG_DIR/grooming8k_train_%A_%a.out"
#SBATCH --error="$LOG_DIR/grooming8k_train_%A_%a.err"

# Run the script
python3 -u train_n_eval.py 
