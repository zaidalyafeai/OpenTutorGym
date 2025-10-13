import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import Evaluator, LLM, LLMGenerator 
from src.Datasets import StepVerify, GSM8K
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    
    dataset = GSM8K()
    dataset = dataset.process_dataset()
    student_model = LLM("google/gemma-3-27b-it", backend="openrouter")
    judge_model = LLM("openai/gpt-4o", backend="openrouter")
    generator = LLMGenerator(student_model=student_model, mode= "standard")
    examples = [dataset[i] for i in range(10)]
    answers = generator.predict(examples)
    judge = Evaluator(judge_model, metric='correct_answer', eval_mode='answer')
    
    # create conversations
    conversations = []
    for i in range(len(examples)):
        conversation = [
            {"role": "user", "content": "Answer: " + answers[i]["solution"]+"\n Gold: " + examples[i]["answer"]},
        ]
        conversations.append(conversation)
    results = judge.evaluate(conversations)
    print(results)
    
     
    
if __name__ == "__main__":

    # asyncio.run(main())
    main()