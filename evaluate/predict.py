import os
from dotenv import load_dotenv
from generate import predict_fewshot, predict_cot
import json
import hashlib

load_dotenv()

async def get_predictions(
    student_model = "qwen2.5:3b", prompt="", question_id = None, tutor_model = "gemma2:27b", question = None, mode = "fewshot:0"
):
    hash_path = hashlib.sha256(f"{student_model}-{tutor_model}".encode()).hexdigest()
    if question_id is not None and mode == "tutor":
        with open(f"results/{hash_path}/test/{question_id}.json", "r") as f:
            results = json.load(f)
        conversation = results["conversation"]
        if "Solution:" in conversation[-2]["content"]:
            response = conversation[-2]["content"].split("Solution:")[1].strip()
        else:
            response = conversation[-2]["content"].strip()
    elif "cot" in mode:
        response = predict_cot(question, student_model)
    elif "fewshot" in mode:
        fewshot_size = int(mode.split(":")[1])
        examples = []
        path = f"results/{hash_path}/train"

        for file in os.listdir(path):
            if "config" in file:
                continue
            with open(f"{path}/{file}", "r") as f:
                results = json.load(f)
                examples.append({"question": results["question"], "solution": results["solution"]})
        response = predict_fewshot(examples=examples, model=student_model, fewshot_size = fewshot_size, question = question)
    else:
        raise ValueError("Invalid question_id")
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0,
    }
    return response.strip(), usage
