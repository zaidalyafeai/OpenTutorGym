import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import TLLM
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="reciprocate/math_dpo_pairs")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train")
    model = TLLM(model_name=args.model_name)
    model.fine_tune(dataset, trainer_config_path="src/configs/dpo_config.yaml")
    model.save_model(f"{args.model_name}-dpo")

if __name__ == "__main__":
    main()
    