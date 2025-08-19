import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import Evaluator
from src.Datasets import StepVerify
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="google/gemma-3-27b-it")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    judge = Evaluator(args.judge_model, metric='mistake_identification', eval_teacher=True, eval_student=False)
    dataset = StepVerify()
    dialouges = dataset.get_dialouge()[:10]
    sample_conversation = [
        [
            {"role": "assistant", "content": "What is 2 * 3 + 3?"},
            {"role": "user", "content": "I don't know. Can you help me?"},
            {"role": "assistant", "content": "Sure, think about the order of operations. What is 2 * 3?"},
            {"role": "user", "content": "6, what's next?"},
            {"role": "assistant", "content": "What is the next operation?"},
            {"role": "user", "content": "I need to add 3 to 6, so the answer is 9."},
            {"role": "assistant", "content": "Great! You got it right."},
        ],

        [
            {"role": "assistant", "content": "What is 2 * 3 + 3?"},
            {"role": "user", "content": "Solve the problem 2 * 3 + 3"},
            {"role": "assistant", "content": "Sure, think about the order of operations. What is 2 * 3?"},
            {"role": "user", "content": "Great attempt you need first to multiply 2 and 3, then add 3 to the result."},
            {"role": "assistant", "content": "Great! how to solve the problem?"},
            {"role": "user", "content": "Please solve the problem. I don't know how to solve it."},
            {"role": "assistant", "content": "Sure, the answer is 9."},
        ]
    ]
    results = judge.evaluate(sample_conversation)
    judge = Evaluator(args.judge_model, metric='revealing_of_the_answer', eval_teacher=True, eval_student=False)
    results = judge.evaluate(sample_conversation) 

    sample_conversation = [
        [
            {"role": "assistant", "content": "The answer is 3*3"},
            {"role": "user", "content": "Gold: 9"}
        ],

        [
            {"role": "assistant", "content": "The answer is 2*3 + 1"},
            {"role": "user", "content": "Gold: 9"},
        ]
    ]  
    judge = Evaluator(args.judge_model, metric='correct_answer', eval_answer=True)
    results = judge.evaluate(sample_conversation)   
    print(results)
if __name__ == "__main__":

    # asyncio.run(main())
    main()