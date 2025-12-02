import os
import dotenv
from openai import OpenAI
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import concurrent
from rich import print
from src.trainer import PeftModelWrapper, CustomTrainer
from peft import PeftModel
import time
from src.utils import image_to_base64

dotenv.load_dotenv()

class TLLM:
    """
    Transformer based LLM class, used to evaluate the model on a list of conversations
    """
    def __init__(self, model_name: str = "google/gemma-3-4B-it"):
        """
        Initialize the transformer based LLM.
        
        Args:
            model_name: Name of the model.
        """
        self.model_name = model_name
        print('loading checkpoint for ', self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            #max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            #full_finetuning = False, # [NEW!] We have full finetuning now!
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print('checkpoint loaded')

    def get_llm_response(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Get response from LLM based on model type
        
        Args:
            contexts: List of contexts.
            
        Returns:
            Response from the LLM.
        """
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

    def fine_tune(self, dataset: List[Dict[str, Any]], trainer_config_path: str = "./src/configs/sft_config.yaml", peft_config_path: str = "./src/configs/lora_config.yaml"):
        """
        Fine tune the transformer based LLM.
        
        Args:
            dataset: List of datasets.
            trainer_config_path: Path to the trainer config.
            peft_config_path: Path to the peft config.
        """
        if 'sft' in trainer_config_path:
            conversations = dataset.get_dialogue()
            conversations = [{"text": self.tokenizer.apply_chat_template(x, tokenize = False, num_proc=1)} for x in conversations]
            conversations = Dataset.from_list(conversations)
        else:
            conversations = dataset        

        if peft_config_path:
            peft = PeftModelWrapper(peft_config_path=peft_config_path)
            self.peft_model = peft.load_model(self.model)

        trainer = CustomTrainer(
        model=self.peft_model,
        tokenizer=self.tokenizer,
        train_dataset=conversations,  # Replace with your training dataset
        eval_dataset=None,   # Replace with your evaluation dataset
        config_path=trainer_config_path,
            ).get_trainer()
        
        trainer_stats = trainer.train()
        return trainer_stats
    
    def save_model(self, checkpoint_path: str):
        """
        Save the transformer based LLM.
        
        Args:
            checkpoint_path: Path to the checkpoint.
        """
        lora_adapter = "./lora_adapter"
        self.peft_model.save_pretrained(lora_adapter, save_adapter=True, save_config=True)

        model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(self.model_name).to("cuda"), lora_adapter)

        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)


class APILLM:
    """
    API based LLM class, used to evaluate the model on a list of conversations
    """
    def __init__(self, model: str = "openai/gpt-4o-mini", port: str = "8787", host: str = "localhost", backend: str = "openrouter"):
        """
        Initialize the API based LLM.
        
        Args:
            model: Name of the model.
            port: Port number.
            host: Host name.
            backend: Backend type.
        """
        self.model = model
        self.port = port
        self.host = host
        self.backend = backend
        print("Model: ", self.model)
        print("Backend: ", self.backend)
        print("==================================")

        self.api_key = ""
        self.api_base = ""
        if self.backend == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            self.api_base = "https://openrouter.ai/api/v1"
        elif self.backend == "ollama":
            self.api_key = "ollama"
            self.api_base = f"http://{self.host}:{self.port}/v1"
        elif self.backend == "vllm":
            self.api_key = "EMPTY"
            self.api_base = f"http://{self.host}:{self.port}/v1"
        else:
            raise ValueError(f"Invalid model: {self.model}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        ## blocking until connection is established
        while True:
            try:
                self.client.chat.completions.create(
                    model=self.model, 
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0,
                    max_tokens=1024
                )
                break
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                time.sleep(5)

    def get_llm_response(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Get response from LLM based on model type
        
        Args:
            contexts: List of contexts.
            
        Returns:
            Response from the LLM.
        """
        
        # print(f"Using model: {model}")
        # print(f"Request contexts: {contexts}")
        try:
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=contexts,
                max_tokens=1024
            )
            
            # print(f"API Response: {response}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            print(response)
            # raise  # Re-raise the exception after logging


class LLM:
    """
    Abstract LLM class, used to evaluate the model on a list of conversations
    """
    def __init__(self, model_name: str = "google/gemma-3-4B-it", backend: str = "openrouter", **kwargs):
        """
        Initialize the LLM.
        
        Args:
            model_name: Name of the model.
            backend: Backend type.
            **kwargs: Additional keyword arguments.
        """
        self.backend = backend
        self.model_name = model_name
        if self.backend in ["openrouter", "ollama", "vllm"]:
            self.model = APILLM(model_name, backend=backend, **kwargs)
        elif self.backend == "transformers":
            self.model = TLLM(model_name)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
    
    def get_llm_response(self, contexts: List[Dict[str, Any]]):
        """
        Get response from LLM based on model type
        
        Args:
            contexts: List of contexts.
            
        Returns:
            Response from the LLM.
        """
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
        image: bytes = None,
        language: str = "English",
        student_level: str = "Middle School",
        max_turns: int = 20,
        log: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate a conversation between a student and a tutor
        
        Args:
            question: Question to be asked.
            image: Image to be used.
            language: Language of the conversation.
            student_level: Level of the student.
            max_turns: Maximum number of turns.
            log: Whether to log the conversation.
            
        Returns:
            List of contexts.
        """

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
        if image is not None:
            # Convert image bytes to base64
            
            question = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_to_base64(image)}}
            ]

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
            try:
                response = self.student_model.get_llm_response(student_messages)
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                break
            conversation.append({"role": "Student", "content": response})
            student_messages.append({"role": "assistant", "content": response})
            tutor_messages.append({"role": "user", "content": response})

            if log:
                print("Student: ", conversation[-1]["content"])

            try:
                response = self.tutor_model.get_llm_response(tutor_messages)
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                break
            conversation.append({"role": "Tutor", "content": response})
            student_messages.append({"role": "user", "content": response})
            tutor_messages.append({"role": "assistant", "content": response})

            if log:
                print("Tutor: ", conversation[-1]["content"])

            if "<END>" in response or 'END' in response:
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
        """
        Generate a conversation between a student and a tutor
        
        Args:
            question: Question to be asked.
            language: Language of the conversation.
            student_level: Level of the student.
            max_turns: Maximum number of turns.
            log: Whether to log the conversation.
            
        Returns:
            List of contexts.
        """
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
        """
        Parse the response from the LLM
        
        Args:
            response: Response from the LLM.
            
        Returns:
            Dict containing reasoning and solution.
        """
        reasoning = response.split("### reasoning")[1].split("### solution")[0].strip()
        solution = response.split("### solution")[1].strip()
        return {"reasoning": reasoning, "solution": solution}
    
    def predict_standard(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction based on a single example
        
        Args:
            example: Example to be predicted.
            
        Returns:
            Dict containing reasoning and solution.
        """
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
        """
        Process conversation examples
        
        Args:
            examples: List of examples.
            
        Returns:
            List of contexts.
        """
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
        """
        Predict conversation examples
        
        Args:
            examples: List of examples.
            question: Question to be asked.
            
        Returns:
            List of contexts.
        """
        contexts= self.process_conversation_examples(examples)
        contexts.append({"role": "user", "content": "Question: " + question})
        return self.get_llm_response(self.student_model, contexts)
    
    def predict(self, examples: List[Dict[str, Any]]):
        """
        Predict examples
        
        Args:
            examples: List of examples.
            
        Returns:
            List of contexts.
        """
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