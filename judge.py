import os
import json
from evaluate.predict import LLMTutorJudge

base_path = "examples/gemma2:27b/qwen2.5:3b/gsm8k/train"

for file in os.listdir(base_path):
    if file.endswith(".json"):
        with open(os.path.join(base_path, file), "r") as f:
            data = json.load(f)
            conversation = ""
            for item in data["conversation"]:
                conversation += f"{item['role']}: {item['content']}".strip()+"\n"
            print(conversation)
            judge = LLMTutorJudge(model="google/gemini-2.0-flash-exp:free")
            response = judge.get_response(conversation)
            print({
                "correctness": response.correctness,
                "helpfulness": response.helpfulness,
                "reasoning": response.reasoning
            })
            break



