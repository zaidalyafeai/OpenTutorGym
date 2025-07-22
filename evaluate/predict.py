import ollama
import os
import dotenv
from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel
from typing import Literal
from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing
import os
_original_cpu_count = multiprocessing.cpu_count
multiprocessing.cpu_count = lambda: 1

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 4
dotenv.load_dotenv()

class JudgeResponse(BaseModel):
    correctness: Literal[True, False]
    helpfulness: Literal[True, False]
    reasoning: str

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability

# Model type constants
GPT_LLMS = ["openai/gpt-4o-mini", "openai/gpt-4o"]
DEEPSEEK_LLMS = ["deepseek/deepseek-chat"]
GEMINI_LLMS = ["google/gemini-flash-1.5", "deepseek/deepseek-v3-base:free", "google/gemini-2.5-pro-exp-03-25:free"]
VLLM_LLMS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
API_LLMS = GPT_LLMS + DEEPSEEK_LLMS + GEMINI_LLMS
UNSLOTH_LLMS = ["unsloth/gemma-3-4B-it", "unsloth/Qwen3-4B", "unsloth/Qwen3-30B-A3B-GGUF", "unsloth/Qwen3-14B"]

class LLMTutorJudge:
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model
        self.system_prompt ="""You are given a conversation between a student and a tutor that solves a question. 
        You need to judge the correctness and helpfulness of the tutor's response. Also reason about the correctness and helpfulness of the tutor's response.
        {
        "correctness": <true or false answer>,
        "helpfulness": <true or false answer>,
        "reasoning": <reasoning>
        }"""

    def get_response(self, prompt: str) -> str:
        
        client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "")
        return JudgeResponse.parse_raw(content)
    
class LLMJudge:
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model

    def get_response_router(self, prompt: str) -> str:
        
        client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        return response.choices[0].message.parsed
    
    def get_response_vllm(self, prompt: str) -> str:
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        return response.choices[0].message.parsed
    
    def get_response(self, prompt: str) -> str:
        if "Qwen" in self.model:
            return self.get_response_vllm(prompt)
        else:
            return self.get_response_router(prompt)

def get_ollama_models(self):
        """Return available models from Ollama and API providers"""
        models = ["Human"]

        # Get Ollama models
        try:
            models_info = ollama.list()
            if "models" in models_info:
                # Sort models by size in decreasing order
                sorted_models = sorted(
                    models_info["models"], key=lambda x: x.get("size", 0), reverse=False
                )
                # Extract just the model names from the sorted list
                models.extend([model["model"] for model in sorted_models])
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")

        # Add API models
        try:
            models.extend(self.API_LLMS)
        except Exception as e:
            print(f"Error adding API models: {e}")

        return models

class UnslothLLM:
    def __init__(self, model_name: str = "unsloth/gemma-3-4B-it"):
        self.model_name = model_name
        print('loading checkpoint for ', self.model_name)
        self.model, _ = FastModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print('checkpoint loaded')

    def get_llm_response(self, contexts: List[Dict[str, Any]]) -> str:
        text = self.tokenizer.apply_chat_template(
            contexts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    def fine_tune(self, dataset: List[Dict[str, Any]]):
        conversations = dataset.get_dialouge()
        conversations = [{"text": self.tokenizer.apply_chat_template(x, tokenize = False, num_proc=1)} for x in conversations]
        conversations = Dataset.from_list(conversations)
        print(conversations)
        model = FastLanguageModel.get_peft_model(
            self.model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )

        trainer = SFTTrainer(
            model = model,
            tokenizer = self.tokenizer,
            train_dataset = conversations,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 1, # Set this for 1 full training run.
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats

    

class APILLM:

    def __init__(self, model: str = "openai/gpt-4o-mini", port: str = "8000"):
        self.model = model
        self.port = port

    def get_llm_response(self, str, contexts: List[Dict[str, Any]]) -> str:
        """Get response from LLM based on model type"""
        api_key = ""
        api_base = ""
        if self.model in API_LLMS:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            api_base = "https://openrouter.ai/api/v1"
        elif self.model in VLLM_LLMS:
            api_key = "EMPTY"
            api_base = f"http://localhost:{self.port}/v1"
        elif self.model in get_ollama_models():
            api_key = "ollama"
            api_base = f"http://localhost:{self.port}/v1"
        else:
            raise ValueError(f"Invalid model: {self.model}")
        
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        # print(f"Using model: {model}")
        # print(f"Request contexts: {contexts}")
        try:
            response = client.chat.completions.create(
                model=self.model, 
                messages=contexts
            )
            
            # print(f"API Response: {response}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            print(response)
            # raise  # Re-raise the exception after logging


class LLMGenerator:
    """Class to handle LLM conversations and predictions"""
    
    def __init__(self, student_model, tutor_model):
        self.mode = 'tutor'
        self.student_model = student_model
        self.tutor_model = tutor_model

    def predict_conversation(
        self,
        question: str,
        language: str = "English",
        student_level: str = "Middle School",
        max_turns: int = 20,
        log: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate a conversation between a student and a math tutor"""
        # System prompts for each agent
        student_system = f"""You speak in {language}.
        Never forget you are a Student in {student_level} and I am a Tutor. Never flip roles! You will always ask questions, never instruct me or ask me to solve the problem.
        I will help you to solve the problem. Here is the problem: {question}. Never forget the problem. You will ask for hints and I will provide you with some guidlines.  
        Try to understand the hints and guidance provided by me. 
        Once you have solved the problem, output the solution in the following format:
        Solution: <solution>
        Only output the final solution as a number do not include any units or other text.
        """
        
        tutor_system = f"""You speak in {language}.
        Never forget you are a Tutor and I am a Student in {student_level}. Never flip roles! You will always provide hints, never reveal the solution.
        You want to help me to answer the question: {question}. Provide hints and guidance rather than complete solutions. The hints and guidance MUST be according to the {student_level}.
        Ask questions to lead me to discover the answer myself. Only ask one question at a time. Be encouraging and supportive. If I provide the solution in the format:
        Solution: <solution> 
        You must verify that the solution is correct. If my solution is correct congratulate me and end the conversation using <END>.
        """

        # Initialize message history
        student_messages = [
            {"role": "system", "content": student_system},
            {"role": "user", "content": question},
        ]
        tutor_messages = [
            {"role": "system", "content": tutor_system},
            {"role": "assistant", "content": question},
        ]
        conversation = [{"role": "assistant", "content": question}]
        for turn_id in range(max_turns):
            response = self.student_model.get_llm_response(student_messages)
            conversation.append({"role": "Student", "content": response})
            student_messages.append({"role": "assistant", "content": response})
            tutor_messages.append({"role": "user", "content": response})

            if log:
                print("Turn ID: ", turn_id)
                print("Tutor: ", conversation[-2]["content"])
                print("Student: ", conversation[-1]["content"])
            response = self.tutor_model.get_llm_response(tutor_messages)
            conversation.append({"role": "Tutor", "content": response})
            student_messages.append({"role": "user", "content": response})
            tutor_messages.append({"role": "assistant", "content": response})

            if "<END>" in response:
                break
                
        return conversation[-2]["content"], conversation

        
    def predict_standard(self, examples: List[Dict[str, Any]], question: str = None):
        """Make a few-shot prediction based on examples"""

        system_prompt = """You are given a question. you need to answer the question step by step. Provide the answer in the following format:
        Reasoning: step by step reasoning to solve the problem
        Solution: final solution without any units or other text
        """
        if len(examples) >0:
            system_prompt += "Here are some examples:\n"


        contexts = [{"role": "system", "content": system_prompt}]
        for example in examples:
            contexts.append({"role": "user", "content": example["question"]})
            contexts.append({"role": "assistant", "content": "Solution: " + example["answer"]})
        
        contexts.append({"role": "user", "content": f"Question: {question}"})
        return self.get_llm_response(self.student_model, contexts)
    
    def process_conversation_examples(self, examples: List[Dict[str, Any]]):
        prompt = """You are a student. You are given a conversation between a student and a tutor that solves a question. 
        Use your knowledge of the conversation to answer the given Question. Answer the question in the following format:
        Reasoning: step by step reasoning to solve the problem
        Solution: final solution without any units or other text
        """
        conversations = [{"role": "system", "content": prompt}]
        for example in examples:

            for turn in example["conversation"][:-2]:
                if turn["role"] == "Student":
                    conversations.append({"role": "assistant", "content": turn["content"]})
                else:
                    conversations.append({"role": "user", "content": turn["content"]})
            conversations.append({"role": "assistant", "content": "Solution: " + example["answer"]})
        return conversations
    
    def predict_conversation_examples(self, examples: List[Dict[str, Any]], question: str = None):
        contexts= self.process_conversation_examples(examples)
        contexts.append({"role": "user", "content": "Question: " + question})
        return self.get_llm_response(self.student_model, contexts)
    
    def predict(self, examples: List[Dict[str, Any]], question: str = None, answer: str = None):
        if "tutor" in self.mode:
            if len(examples) > 0:
                return self.predict_conversation_examples(examples, question)
            else:
                return self.predict_conversation(question, answer)
        elif "cot" in self.mode or "fewshot" in self.mode:
            return self.predict_standard(examples, question)
        else:
            raise ValueError("Invalid mode")