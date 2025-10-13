import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import LLMGenerator, APILLM
from src.Datasets import GSM8K
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_model", type=str, default="Qwen2.5-3B-Instruct")
    parser.add_argument("--tutor_model", type=str, default="Qwen2.5-3B-Instruct")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    student_model = APILLM(args.student_model, backend="vllm", port="8787")
    tutor_model = APILLM(args.tutor_model, backend="vllm", port="8788")

    predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)

    dataset = GSM8K(split = args.split)
    dataset = dataset.process_dataset()

    for i, problem in tqdm(enumerate(dataset), total=10):
        output = []
        for conversation in predictor.predict_conversation(
            problem["question"],
            log = True
        ):
            output.append(conversation)
        # print(output[-1])
        if i == 10:
            break

    # evaluate
    
if __name__ == "__main__":

    # asyncio.run(main())
    main()