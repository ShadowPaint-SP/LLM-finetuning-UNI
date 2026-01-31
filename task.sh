#!/bin/bash
# Slurm submission script for the fine-tuning exercise with Qwen3-0.6B and LoRA.
# Edit paths (cache/output/venv/module names) to match your actual ones before submitting.

#SBATCH --job-name=seperate15RAG
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 1) Load modules (update for your site). Use `module spider Python` to find versions.
module purge

# 2) Activate the virtual environment created earlier (matches instructions in 07-finetuning.py)
source start_kernel.sh

# 3) Sanity check: ensure you see at least one GPU and the CUDA version
nvidia-smi

# 4) Keep all caches/checkpoints off $HOME
export HF_HOME=/data/horse/ws/davo303e-group-proj/hf_home
export TRANSFORMERS_CACHE=/data/horse/ws/davo303e-group-proj/hf_cache
export HF_DATASETS_CACHE=/data/horse/ws/davo303e-group-proj/hf_datasets

# 5) (Optional) speed up uploads to the Hub if you push artifacts later
# export HF_HUB_ENABLE_HF_TRANSFER=1

# 6) Run the full pipeline: baseline inference/eval -> LoRA fine-tune -> eval + generations.
#    Tweak max_steps/max_train_samples to ensure you finish on time.
python main.py

#7) Submit on Slurm (after editing 07-finetuning.sh):
#   sbatch 07-finetuning.sh

#   Helpful: squeue --me                #lists all your jobs (running, pending, etc.)
#            watch squeue --me          #monitors your jobs live and updates every 2 seconds
#            scancel <job_id>           #cancels the job with the given ID