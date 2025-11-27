import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.predict import LLMGenerator, APILLM
from src.Datasets import GSM8K
import json

def main():

    dataset = GSM8K(split = "train")
    dataset = dataset.process_dataset()
    dataset = dataset.select(range(5))
    models = ["qwen/qwen-2.5-7b-instruct", "meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct", "allenai/olmo-3-7b-instruct"]
    
    for tutor_model in models:
        student_model = APILLM("google/gemma-3-27b-it", backend="openrouter")
        tutor_model = APILLM(tutor_model, backend="openrouter")

        predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)
        conversations = []
        for problem in tqdm(dataset):
            _, conversation = predictor.predict_conversation(
                problem["question"],
                log = True
            )
            conversations.append(conversation)
        tutor_model_name = tutor_model.split("/")[-1]
        with open(f"conversations_{tutor_model_name}.json", "w") as f:
            json.dump(conversations, f)
    
    
if __name__ == "__main__":

    main()