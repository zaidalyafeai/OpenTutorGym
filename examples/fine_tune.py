import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.Datasets import load_dataset
from src.predict import TLLM

dataset = load_dataset('socra_teach')
model = TLLM(model_name="../models/Qwen2.5-7B-Instruct")
model.fine_tune(dataset, trainer_config_path="src/configs/sft_config.yaml", peft_config_path="src/configs/lora_config.yaml")
model.save_model("../models/Qwen2.5-7B-Instruct-lora")