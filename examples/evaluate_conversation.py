import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import LLM
from src.evaluate import Evaluator
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="openai/gpt-4o")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    judge = LLM(args.judge_model, backend="openrouter")
    evaluator = Evaluator(judge, ['answer_reveal'])
    sample_conversations = [
        [
            {"role": "assistant", "content": "What is 2 * 3 + 3?"},
            {"role": "user", "content": "I don't know. Can you help me?"},
            {"role": "assistant", "content": "Sure, think about the order of operations. What is 2 * 3?"},
            {"role": "user", "content": "7, what's next?"},
            {"role": "assistant", "content": "correct, what is the next operation?"},
            {"role": "user", "content": "I need to add 3 to 7, so the answer is 10."},
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
    results = evaluator.evaluate(sample_conversations)
    print(results)

    sample_conversation = [
        [
            {"role": "assistant", "content": "What is 2 * 3 + 3?"},
            {"role": "user", "content": "Answer: 9"}
        ],

        [
            {"role": "assistant", "content": "What is 2 * 3 + 1?"},
            {"role": "user", "content": "Answer: 7"},
        ],
        [
            {"role": "assistant", "content": "What is 2 * 3 + 2?"},
            {"role": "user", "content": "Answer: 8"},
        ],
        [
            {"role": "assistant", "content": "What is 25/5?"},
            {"role": "user", "content": "Answer: 7"},
        ],
        [
            {"role": "assistant", "content": "What is 25/5?"},
            {"role": "user", "content": "Answer: 5"},
        ],
    ]  
    evaluator = Evaluator(judge, ['correct_answer'])
    results = evaluator.evaluate(sample_conversation)   
    print(results)
if __name__ == "__main__":

    # asyncio.run(main())
    main()