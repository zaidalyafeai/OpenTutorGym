#!/bin/bash
#SBATCH --job-name=ai-tutor
#SBATCH --output=slurm/ai-tutor_%A_%a.out
#SBATCH --error=slurm/ai-tutor_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00

student_model="gemma-3-27b-it"
tutor_model=$1

bash scripts/start_tutor_server.sh $tutor_model 8788 &
bash scripts/start_student_server.sh $student_model 8787 &
uv run examples/generate_conversation.py --student_model $student_model --tutor_model $tutor_model
