import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Datasets import load_dataset
from src.predict import TLLM
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="socra_teach")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    model = TLLM(model_name=args.model_name)
    model.fine_tune(dataset)
    model.save_model(f"{args.model_name}-lora")

if __name__ == "__main__":
    main()
    