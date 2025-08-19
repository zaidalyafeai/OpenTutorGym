import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import Evaluator, APILLM, LLMGenerator 
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
    student_model = APILLM("google/gemma-3-27b-it")
    generator = LLMGenerator(student_model=student_model, mode= "standard")
    examples = [dataset[i] for i in range(10)]
    answers = generator.predict(examples)
    # print(examples)
    # print(answers)
    judge = Evaluator("openai/gpt-4o", metric='correct_answer', eval_answer=True)
    
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