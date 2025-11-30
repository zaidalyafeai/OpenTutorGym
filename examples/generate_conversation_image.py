"""
Generate conversational example from a vision dataset
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import LLMGenerator, APILLM
from src.Datasets import MathVision, ScienceQA
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_model", type=str, default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--tutor_model", type=str, default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()

def main():

    # load models
    args = parse_args()
    student_model = APILLM(args.student_model, backend="openrouter")
    tutor_model = APILLM(args.tutor_model, backend="openrouter")

    # load predictor
    predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)

    # load dataset
    dataset = MathVision(split = args.split)
    dataset = dataset.process_dataset()
    example = dataset[0]
    image = example["image"]['bytes'] if example["image"] is not None else None
    # generate conversation
    _, conversation = predictor.predict_conversation(
        example["question"],
        image = image,
        log = True
    )


    
if __name__ == "__main__":
    main()