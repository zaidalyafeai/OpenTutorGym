#!/bin/bash
#SBATCH --job-name=ai-tutor
#SBATCH --output=slurm/ai-tutor_%A_%a.out
#SBATCH --error=slurm/ai-tutor_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00

uv run examples/fine_tune.py
