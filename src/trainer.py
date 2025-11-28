from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import yaml
import os, json
from typing import Optional, Dict, Any
from peft import (
    get_peft_model,
    LoraConfig,
    AdaLoraConfig,
)
from trl import (
    SFTTrainer,
    SFTConfig,
    GRPOTrainer,
    GRPOConfig,
)
from datasets import Dataset
from src.Datasets import load_dataset

PEFT_TYPE_MAP = {
    "LORA": LoraConfig,
    "ADALORA": AdaLoraConfig,
}

TRAINER_TYPE_MAP = {
    "SFT": (SFTTrainer, SFTConfig),
    "GRPO": (GRPOTrainer, GRPOConfig),
}

def load_config_dict(config_path) -> Optional[Dict[str, Any]]:
    if not config_path:
        return None
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        elif config_path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        else:
            raise ValueError("Use .json or .yml/.yaml for PEFT config")

class PeftModelWrapper:
    def __init__(self, peft_config_path: Optional[str] = None):
        self.peft_config_path = peft_config_path

    def _build_peft_config_obj(self, cfg: Dict[str, Any]):
        peft_type = cfg.pop("peft_type").upper() if "peft_type" in cfg else "LORA"
        ConfigClass = PEFT_TYPE_MAP.get(peft_type)
        if ConfigClass is None:
            raise ValueError(f"Unsupported peft_type '{peft_type}'. "
                             f"Supported: {list(PEFT_TYPE_MAP.keys())}")

        cfg.setdefault("task_type", "CAUSAL_LM")

        return ConfigClass(**cfg)

    def load_model(self, model):
        cfg_dict = load_config_dict(self.peft_config_path)
        if cfg_dict:
            peft_config = self._build_peft_config_obj(cfg_dict)
            return get_peft_model(model, peft_config)

        # Fallback: a LoRA default
        default_cfg = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, default_cfg)
    

class CustomTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset = None, config_path = None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config_path = config_path
        self.kwargs = kwargs
        
        self.cfg_dict = load_config_dict(self.config_path)
        self.trainer_type = self.cfg_dict.pop("trainer_type").upper()
        
    def _build_trainer_config(self, cfg: Dict[str, Any]):
        _, ConfigClass = TRAINER_TYPE_MAP.get(self.trainer_type, (None, None))
        if ConfigClass is None:
            raise ValueError(f"Unsupported trainer_type '{self.trainer_type}'. "
                             f"Supported: {list(TRAINER_TYPE_MAP.keys())}")
        cfg['learning_rate'] = float(cfg.get('learning_rate', 2e-5))
        return ConfigClass(**cfg)

    def get_trainer(self):

        if self.cfg_dict:
            trainer_config = self._build_trainer_config(self.cfg_dict)
            TrainerClass, _ = TRAINER_TYPE_MAP.get(self.trainer_type)
            if TrainerClass is None:
                raise ValueError(f"Unsupported trainer_type '{self.trainer_type}'. "
                                 f"Supported: {list(TRAINER_TYPE_MAP.keys())}")

            return TrainerClass(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=trainer_config,
                **self.kwargs
            )
       
def main():
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        device_map="auto",
    )

    peft = PeftModelWrapper(peft_config_path="./configs/lora_config.yaml")
    model = peft.load_model(base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
    )

    dataset = load_dataset('comta')
    conversations = dataset.get_dialogue()
    conversations = [{"text": tokenizer.apply_chat_template(x, tokenize = False, num_proc=1)} for x in conversations]
    conversations = Dataset.from_list(conversations)
    print(conversations)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=conversations,
        eval_dataset=None, 
        config_path="./configs/sft_config.yaml",
    ).get_trainer()

    trainer.train()

    
    
if __name__ == "__main__":
    main()