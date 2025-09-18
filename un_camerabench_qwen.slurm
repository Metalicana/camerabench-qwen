#!/bin/bash
#SBATCH -p gpu                  # use the main GPU partition
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --cpus-per-task=16      # reasonable CPU cores
#SBATCH --mem=64G               # memory for data loading + model
#SBATCH --time=6:00:00          # max run time
#SBATCH -J camerabench_qwen
#SBATCH -o logs/%x_%j.out       # log file

module purge
module load cuda/12.1           # adjust if CRCV uses different CUDA modules
source activate qwen-radi-camera

srun python evaluate_camerabench_qwen.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --split test \
  --local_data_root /home/ab575577/projects_fall_2025/t2v_metrics/camerabench/data \
  --out_file camerabench_summary.csv \
  --raw_file camerabench_raw_scores.csv
