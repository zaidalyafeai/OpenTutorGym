from Datasets import load_dataset
from predict import TLLM

dataset = load_dataset('comta')
model = TLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct")
model.fine_tune(dataset, trainer_config_path="./configs/sft_config.yaml", peft_config_path="./configs/lora_config.yaml")