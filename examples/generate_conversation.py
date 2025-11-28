import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import LLMGenerator, APILLM
from src.Datasets import GSM8K
from argparse import ArgumentParser
import json 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_model", type=str, default="gemma-3-27b-it")
    parser.add_argument("--tutor_model", type=str, default="Mistral-7B-Instruct-v0.2")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    student_model = APILLM(args.student_model, backend="vllm", port="8787")
    tutor_model = APILLM(args.tutor_model, backend="vllm", port="8788")

    predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)

    dataset = GSM8K(split = args.split)
    dataset = dataset.process_dataset()
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(50))

    conversations = []
    for problem in tqdm(dataset):
        _, conversation = predictor.predict_conversation(
            problem["question"],
            log = True
        )
        conversations.append(conversation)
    tutor_model_name = args.tutor_model.split("/")[-1]
    with open(f"examples/conversations_{tutor_model_name}.json", "w") as f:
        json.dump(conversations, f, indent=4)

    # evaluate
    
if __name__ == "__main__":

    # asyncio.run(main())
    main()