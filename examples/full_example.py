import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import LLMGenerator, APILLM, TLLM
from src.Datasets import GSM8K, DialogueDataset, Problem, Conversation
from src.evaluate import Evaluator
from argparse import ArgumentParser 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--tutor_model", type=str, default="deepseek/deepseek-chat-v3-0324")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-4o")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()

def main():

    
    args = parse_args()

    # setup the student and tutor models
    student_model = APILLM(args.student_model, backend="openrouter")
    tutor_model = APILLM(args.tutor_model, backend="openrouter")

    # create the data generator class
    predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)

    # setup the seed dataset, we use only one example for testing
    dataset = GSM8K(split = args.split)
    dataset = dataset.process_dataset()
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(1))

    # generate the conversations
    conversations = []
    for problem in tqdm(dataset):
        _, conversation = predictor.predict_conversation(
            problem["question"],
            log = True
        )
        dialouge = []
        for turn in conversation:
            role = "user" if turn['role'].lower() == "student" else "assistant"
            dialouge.append({"role": role, "content": turn['content']})
        conversations.append(dialouge)

    # evaluate the conversations
    judge = APILLM(args.judge_model, backend="openrouter")
    evaluator = Evaluator(judge, ['correct_answer'])
    evaluator.evaluate(conversations)     

    # create a custom synthetic dataset
    class SyntheticDataset(DialogueDataset):
        def __init__(self, conversations):
            self.conversations = []
            for conversation in conversations:
                self.conversations.append(Conversation(problem = Problem(text = conversation[0]['content']), dialogue = conversation))
        
    dataset = SyntheticDataset(conversations)
    
    # fine tune the tutor model
    model = TLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    model.fine_tune(dataset)
    model.save_model("Qwen2.5-0.5B-Instruct-sft")
    
if __name__ == "__main__":
    main()