import ollama
import os
import dotenv
from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel
from typing import Literal
import asyncio
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

    async def get_response(self, prompt: str) -> str:
        
        client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        return response.choices[0].message.parsed
        
class LLMPredictor:
    """Class to handle LLM conversations and predictions"""
    
    # Model type constants
    GPT_LLMS = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    DEEPSEEK_LLMS = ["deepseek/deepseek-chat"]
    GEMINI_LLMS = ["google/gemini-flash-1.5", "google/gemini-2.0-pro-exp-02-05:free"]
    VLLM_LLMS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
    API_LLMS = GPT_LLMS + DEEPSEEK_LLMS + GEMINI_LLMS
    
    def __init__(self, mode: str = "standard", student_model: str = "gemma2:27b", tutor_model: str = "gemma2:27b"):
        self.mode = mode
        self.student_model = student_model
        self.tutor_model = tutor_model
        """Initialize the LLMConversationGenerator"""
        pass 

    async def get_llm_response(self, model: str, contexts: List[Dict[str, Any]], port: str = "8000") -> str:
        """Get response from LLM based on model type"""
        api_key = ""
        api_base = ""
        if model in self.API_LLMS:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            api_base = "https://openrouter.ai/api/v1"
        elif model in self.VLLM_LLMS:
            api_key = "EMPTY"
            api_base = f"http://localhost:{port}/v1"
        elif model in self.get_ollama_models():
            api_key = "ollama"
            api_base = f"http://localhost:{port}/v1"
        else:
            raise ValueError(f"Invalid model: {model}")
        
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        # print(f"Using model: {model}")
        # print(f"Request contexts: {contexts}")
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create, 
                model=model, 
                messages=contexts
            )
            # print(f"API Response: {response}")
            return response.choices[0].message.content, contexts
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            # raise  # Re-raise the exception after logging
         

    async def predict_conversation(
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
        
        conversation = []
        for _ in range(max_turns):
            response, _ = await self.get_llm_response(self.student_model, student_messages, port = "8000")
            if log:
                print("Student: ", response.strip())
            conversation.append({"role": "Student", "content": response})
            student_messages.append({"role": "assistant", "content": response})
            tutor_messages.append({"role": "user", "content": response})

            response, _ = await self.get_llm_response(self.tutor_model, tutor_messages, port = "8080")
            if log:
                print("Tutor: ", response.strip())
            conversation.append({"role": "Tutor", "content": response})
            student_messages.append({"role": "user", "content": response})
            tutor_messages.append({"role": "assistant", "content": response})

            if "<END>" in response:
                break
                
        return conversation[-2]["content"], conversation

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
        
    async def predict_standard(self, examples: List[Dict[str, Any]], question: str = None):
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
        return await self.get_llm_response(self.student_model, contexts)
    
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
    
    async def predict_conversation_examples(self, examples: List[Dict[str, Any]], question: str = None):
        contexts= self.process_conversation_examples(examples)
        contexts.append({"role": "user", "content": "Question: " + question})
        return await self.get_llm_response(self.student_model, contexts)
    
    async def predict(self, examples: List[Dict[str, Any]], question: str = None, answer: str = None):
        if "tutor" in self.mode:
            if len(examples) > 0:
                return await self.predict_conversation_examples(examples, question)
            else:
                return await self.predict_conversation(question, answer)
        elif "cot" in self.mode or "fewshot" in self.mode:
            return await self.predict_standard(examples, question)
        else:
            raise ValueError("Invalid mode")