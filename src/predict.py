import ollama
import os
import dotenv
from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel
from typing import Literal
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import multiprocessing
import os
import json
from tqdm import tqdm
import asyncio
import concurrent
from rich import print

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


class Evaluator:
    def __init__(self, model,  
    metric = 'mistake_identification',
    eval_teacher: bool = False,
    eval_student: bool = False,
    eval_answer: bool = False):


        self.model = model
        self.metric = metric

        if eval_teacher:
            if metric == 'mistake_identification':
                self.metrics = {'Mistake_Identification': []}
                self.system_prompt = """You are given a conversation. 
                Has the tutor identified a mistake in the student’s response?
                Return:
                {
                "Mistake_Identification": ["Yes/To some extent/No"]
                }"""

            elif metric == 'revealing_of_the_answer':
                self.metrics = {'Revealing_of_the_Answer': []}
                self.system_prompt = """You are given a conversation. 
                Does the tutor reveal the final answer?
                Return:
                {
                "Revealing_of_the_Answer": ["Yes (and the revealed answer is correct)/Yes (but the revealed answer is incorrect)/No"]
                }"""

            elif metric == 'providing_guidance':
                self.metrics = {'Providing_Guidance': []}
                self.system_prompt = """You are given a conversation. 
                Does the tutor offer correct and relevant guidance?
                Return:
                {
                "Providing_Guidance": ["Yes/To some extent/No"]
                }"""

            elif metric == 'tutor_tone':
                self.metrics = {'Tutor_Tone': []}
                self.system_prompt = """You are given a conversation. 
                What is the tone of the tutor’s response?
                Return:
                {
                "Tutor_Tone": ["Encouraging/Neutral/Offensive"]
                }"""

            elif metric == 'all':
                self.metrics = {
                    "Mistake_Identification": [],
                    "Revealing_of_the_Answer": [],
                    "Providing_Guidance": [],
                    "Tutor_Tone": []
                }
                self.system_prompt = """You are given a conversation. Evaluate the tutor's response:
                Return:
                {
                "reasoning": <explanation>,
                "Mistake_Identification": "Yes/To some extent/No",
                "Revealing_of_the_Answer": "Yes (and the revealed answer is correct)/Yes (but the revealed answer is incorrect)/No",
                "Providing_Guidance": "Yes/To some extent/No",
                "Tutor_Tone": "Encouraging/Neutral/Offensive"
                }"""

        elif eval_student:
            if metric == 'correct_answer':
                self.metrics = {'Correct_Answer': []}
                self.system_prompt = """You are given a conversation between a student and a tutor.  
                Provide the correct answer to the question discussed in the conversation.  
                Return:
                {
                "Correct_Answer": Yes/No
                }"""

            elif metric == 'concept_usage':
                self.metrics = {'Concept_Usage': [], 'Key_Concepts_Used': [], 'Key_Concepts_Missed': []}
                self.system_prompt = """You are given a conversation between a student and a tutor.
                Judge whether the student used the most relevant concepts, terms, formulas, or steps for solving the problem.
                Return:
                {
                "Concept_Usage": "Yes/Partially/No",
                "Key_Concepts_Used": ["<list key concepts used>"],
                "Key_Concepts_Missed": ["<list key concepts that should have been used>"]
                }"""

            elif metric == 'conciseness':
                self.metrics = {'Conciseness': [0.0], 'Extraneous_Notes': []}
                self.system_prompt = """You are given a conversation between a student and a tutor.
                Rate how concise the student's final answer/explanation is (higher = more concise, minimal redundancy while retaining substance).
                Return:
                {
                "Conciseness":[0-100],
                "Extraneous_Notes": ["<briefly note any redundancy or off-topic content>"]
                }"""

            elif metric == 'completeness':
                self.metrics = {'Completeness': [], 'Missing_Elements': []}
                self.system_prompt = """You are given a conversation between a student and a tutor.
                Evaluate whether the student's answer covers all required parts (sub-questions, units, proofs, steps).
                Return:
                {
                "Completeness": ["Complete/Partially complete/Incomplete"],
                "Missing_Elements": ["<list what is missing or underdeveloped>"]
                }"""


            elif metric == 'all':
                self.metrics = {
                    'Correct_Answer': [],
                    'Concept_Usage': [],
                    'Key_Concepts_Used': [""],
                    'Key_Concepts_Missed': [""],
                    'Conciseness': [0.0],
                    'Extraneous_Notes': [""],
                    'Completeness': [],
                    'Missing_Elements': [""]
                }
                self.system_prompt = """You are given a conversation between a student and a tutor.
                First, provide the correct answer to the question.
                Then, evaluate the student's response on concept usage, conciseness, and completeness.
                Return:
                {
                "Correct_Answer": ["<correct answer text>"],
                "Concept_Usage": ["Yes/Partially/No"],
                "Key_Concepts_Used": ["<key concepts used>"],
                "Key_Concepts_Missed": ["<key concepts missed>"],
                "Conciseness": [0-100],
                "Extraneous_Notes": ["<redundancy or off-topic notes>"],
                "Completeness": ["Complete/Partially complete/Incomplete"],
                "Missing_Elements": ["<what's missing>"]
                }"""
        
        elif eval_answer:
            if metric == 'correct_answer':
                self.metrics = {'Correct_Answer': []}
                self.system_prompt = """You are given a conversation conversation, you need to judge 
                if the provided answer matches the gold answer.
                Return:
                {
                "Correct_Answer": "Yes/No"
                }"""
        else:
            raise ValueError(f"Invalid metric: {metric}")
    
    def evaluate_conversation(self, contexts: List[Dict[str, Any]]) -> str:
        response = self.model.get_llm_response([{"role": "system", "content": self.system_prompt}] + contexts)
        try:
            return self.parse_response(response)
        except Exception as e:
            return {k: 0.0 if isinstance(v, float) else "No" for k, v in self.metrics.items()}

    def evaluate(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate multiple conversations in parallel and return aggregated metrics.
        
        Args:
            conversations: A list of conversation lists, where each conversation is a list of message dicts
                          with 'role' and 'content' keys.
                          
        Returns:
            Dict containing aggregated metrics for all conversations.
        """
        if not conversations:
            return {k: [] if isinstance(v, list) else v for k, v in self.metrics.items()}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create futures with their original indices
            future_to_index = {executor.submit(self.evaluate_conversation, conversations[i]): i for i in range(len(conversations))}
            
            # Initialize results list with None values
            results = [None] * len(conversations)
            
            pbar = tqdm(concurrent.futures.as_completed(future_to_index), total=len(conversations), desc="Evaluating")
            
            for future in pbar:
                try:
                    response = future.result()
                    original_index = future_to_index[future]
                    results[original_index] = response
                            
                except Exception as e:
                    print(f"Error evaluating conversation: {str(e)}")
                    # Add default values for failed evaluations
                    original_index = future_to_index[future]
                    failed_metrics = {k: 0.0 if isinstance(v, float) else "" for k, v in self.metrics.items()}
                    results[original_index] = failed_metrics
            
            # Update metrics in original order
            for response in results:
                if response:
                    for metric_name, metric_value in response.items():
                        if isinstance(self.metrics.get(metric_name), list):
                            self.metrics[metric_name].append(metric_value)
                    
        return self.metrics

    
    def parse_response(self, response: str) -> Dict[str, Any]:
        response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(response)
    

def get_ollama_models():
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

class TLLM:
    def __init__(self, model_name: str = "google/gemma-3-4B-it"):
        self.model_name = model_name
        print('loading checkpoint for ', self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
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
        model = peft.get_peft_model(
            self.model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
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

    def __init__(self, model: str = "openai/gpt-4o-mini", port: str = "8787", backend: str = "openrouter"):
        self.model = model
        self.port = port
        self.backend = backend
        print("Model: ", self.model)
        print("Backend: ", self.backend)
        print("==================================")

    def get_llm_response(self, contexts: List[Dict[str, Any]]) -> str:
        """Get response from LLM based on model type"""
        api_key = ""
        api_base = ""
        if self.backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            api_base = "https://openrouter.ai/api/v1"
        elif self.backend == "ollama":
            api_key = "ollama"
            api_base = f"http://localhost:{self.port}/v1"
        elif self.backend == "vllm":
            api_key = "EMPTY"
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


class LLM:
    def __init__(self, model_name: str = "google/gemma-3-4B-it", backend: str = "openrouter", **kwargs):
        self.backend = backend
        self.model_name = model_name
        if self.backend in ["openrouter", "ollama", "vllm"]:
            self.model = APILLM(model_name, backend=backend, **kwargs)
        elif self.backend == "transformers":
            self.model = TLLM(model_name)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
    
    def get_llm_response(self, contexts: List[Dict[str, Any]]):
        return self.model.get_llm_response(contexts)

class LLMGenerator:
    """Class to handle LLM conversations and predictions"""
    
    def __init__(self, student_model, tutor_model = None, mode: str = 'tutor'):
        self.mode = mode
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

        if log:
            print("Question: ", question)

        for turn_id in range(max_turns):
            if log:
                print("Turn ID: ", turn_id)
            response = self.student_model.get_llm_response(student_messages)
            conversation.append({"role": "Student", "content": response})
            student_messages.append({"role": "assistant", "content": response})
            tutor_messages.append({"role": "user", "content": response})

            if log:
                print("Student: ", conversation[-1]["content"])

            response = self.tutor_model.get_llm_response(tutor_messages)
            conversation.append({"role": "Tutor", "content": response})
            student_messages.append({"role": "user", "content": response})
            tutor_messages.append({"role": "assistant", "content": response})

            if log:
                print("Tutor: ", conversation[-1]["content"])

            if "<END>" in response:
                print("--------------------------------")
                break
                
        return conversation[-2]["content"], conversation
    
    def async_predict_conversation(
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

        if log:
            print("Question: ", question)

        for turn_id in range(max_turns):
            if log:
                print("Turn ID: ", turn_id)
            response = ""
            conversation.append({"role": "Student", "content": response})
            for chunk in self.student_model.get_llm_response(student_messages):
                response += chunk
                conversation[-1]["content"] = response
                yield conversation.copy()
            
            student_messages.append({"role": "assistant", "content": response})
            tutor_messages.append({"role": "user", "content": response})

            if log:
                print("Student: ", response)
            response = ""
            conversation.append({"role": "Tutor", "content": response})
            for chunk in self.tutor_model.get_llm_response(tutor_messages):
                response += chunk
                conversation[-1]["content"] = response
                yield conversation.copy()
            student_messages.append({"role": "user", "content": response})
            tutor_messages.append({"role": "assistant", "content": response})

            if log:
                print("Tutor: ", conversation[-1]["content"])

            if "<END>" in response:
                print("--------------------------------")
                break



    def parse_response(self, response: str) -> Dict[str, Any]:
        reasoning = response.split("### reasoning")[1].split("### solution")[0].strip()
        solution = response.split("### solution")[1].strip()
        return {"reasoning": reasoning, "solution": solution}
    
    def predict_standard(self, example: Dict[str, Any]):
        """Make a few-shot prediction based on examples"""
        system_prompt = """You are given a question. you need to answer the question step by step. Provide the answer in the following format:
        ### reasoning
        reasoning for the answer
        ### solution
        final solution without any units or other text
        """
        contexts = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question: {example['question']}"}]
        response = self.student_model.get_llm_response(contexts)
        try:
            parsed_response = self.parse_response(response)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(response)
            parsed_response = {"reasoning": "", "solution": ""}
        parsed_response["id"] = example["id"]
        return parsed_response
    
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
    
    def predict(self, examples: List[Dict[str, Any]]):
        if "tutor" in self.mode:
            return self.predict_conversation(example)
        elif "standard" in self.mode:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create futures with their original indices
                future_to_index = {executor.submit(self.predict_standard, examples[i]): i for i in range(len(examples))}
                
                # Initialize results list with None values
                results = [None] * len(examples)
                
                pbar = tqdm(concurrent.futures.as_completed(future_to_index), total=len(examples), desc="Predicting")
                
                for future in pbar:
                    try:
                        response = future.result()
                        original_index = future_to_index[future]
                        results[original_index] = response
                    except Exception as e:
                        print(f"Error processing example: {str(e)}")
                        original_index = future_to_index[future]
                        results[original_index] = {"reasoning": "", "solution": "", "id": examples[original_index].get("id", "")}
                        
                return results
        else:
            raise ValueError("Invalid mode")