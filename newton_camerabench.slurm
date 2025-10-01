#!/bin/bash
#SBATCH --job-name=radfm_test
#SBATCH --output=radfm_test_%j.out
#SBATCH --error=radfm_test_%j.err
#SBATCH --partition=normal                 # partition (queue)
#SBATCH --gres=gpu:nvidia_h100_pcie:1      # request 1 H100 GPU
#SBATCH --cpus-per-task=8                  # CPU cores
#SBATCH --mem=64G                          # memory
#SBATCH --time=01:00:00                    # max walltime (hh:mm:ss)

module load anaconda       # adjust if CRCV uses different CUDA modules
source activate qwen-radi

python evaluate_camerabench_qwen.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --split test \
  --local_data_root /home/ab575577/projects_fall_2025/t2v_metrics/camerabench/data \
  --out_file camerabench_summary.csv \
  --raw_file camerabench_raw_scores.csv