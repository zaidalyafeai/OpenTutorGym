
import gradio as gr
import ollama
import time
from typing import List, Dict, Any, Generator
import os
import dotenv
from openai import OpenAI
import json
import argparse

dotenv.load_dotenv()

GPT_LLMS = ["openai/gpt-4o-mini", "openai/gpt-4o"]
DEEPSEEK_LLMS = ["deepseek/deepseek-chat"]
GEMINI_LLMS = ["google/gemini-1.5-flash", "google/gemini-2.0-pro-exp-02-05:free"]
API_LLMS = GPT_LLMS + DEEPSEEK_LLMS + GEMINI_LLMS

gr.set_static_paths(paths=["assets/"])

def get_ollama_response(model: str, contexts: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Get streaming response from Ollama model"""
    response = ollama.chat(
        model=model,
        messages=contexts,
        stream=False
    )
    
    return response['message']['content']

def get_chatgpt_response(model: str, contexts: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Get streaming response from ChatGPT API"""


    if model in API_LLMS:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Invalid model: {model}")

    

    chat_completion = client.chat.completions.create(
        model=model,
        messages= contexts,
        stream=False
    )

    return chat_completion.choices[0].message.content

def get_llm_response(model: str, contexts: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Get streaming response from LLM"""
    if model in API_LLMS:
        return get_chatgpt_response(model, contexts)
    else:
        return get_ollama_response(model, contexts)

def create_conversation(
    student_model: str,
    tutor_model: str, 
    question: str,
    solution: str,
    language: str,
    student_level: str = "Middle School",
    max_turns: int = 20
) -> Generator[List[Dict[str, Any]], None, None]:
    """Generate a conversation between a student and a math tutor"""
    
    conversation = []

    # System prompts for each agent
    student_system = f"""
    You speak in {language}.
    Never forget you are a Student in {student_level} and I am a Tutor. Never flip roles! You will always ask questions, never instruct me or ask me to solve the problem.
    I will help you to solve the problem. Here is the problem: {question}. Never forget the problem. You will ask for hints and I will provide you with some guidlines.  
    Try to understand the hints and guidance provided by me. 
    Once you have solved the problem, output the solution in the following format:
    Solution: <solution>
    """
    tutor_system = f"""
    You speak in {language}.
    Never forget you are a Tutor and I am a Student in {student_level}. Never flip roles! You will always provide hints, never reveal the solution.
    You want to help me to answer the question: {question}. Provide hints and guidance rather than complete solutions. The hints and guidance MUST be according to the {student_level}.
    Ask questions to lead me to discover the answer myself. Only ask one question at a time. Be encouraging and supportive. If I provide the solution in the format:
    Solution: <solution> 
    You must verify that the solution is correct by checking the solution: {solution}. If my solution is correct congratulate me and end the conversation using <END>.
    """

    # Initial student response
    conversation = []
    
    student_messages = [{"role": "system", "content": student_system}, {"role": "user", "content": question}]
    tutor_messages = [{"role": "system", "content": tutor_system}, {"role": "assistant", "content": question}]

    
    for turn in range(max_turns): 
        response = get_llm_response(student_model, student_messages)
        print('Student: ', response.strip())
        student_messages.append({"role": "assistant", "content": response})
        tutor_messages.append({"role": "user", "content": response})

        response = get_llm_response(tutor_model, tutor_messages)
        print('Tutor: ', response.strip())
        student_messages.append({"role": "user", "content": response})
        tutor_messages.append({"role": "assistant", "content": response})

        if '<END>' in response:
            break

    conversation = []
    for message in student_messages:
        if message["role"] == "assistant":
            conversation.append({"role": "Student", "content": message["content"]})
        elif message["role"] == "user":
            conversation.append({"role": "Tutor", "content": message["content"]})

    return conversation

def get_available_models():
    models = ['Human']
    
    # Get Ollama models
    try:
        models_info = ollama.list()
        if 'models' in models_info:
            # Sort models by size in decreasing order
            sorted_models = sorted(
                models_info['models'], 
                key=lambda x: x.get('size', 0), 
                reverse=False
            )
            # Extract just the model names from the sorted list
            models.extend([model['model'] for model in sorted_models])
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
    
    # Add ChatGPT models
    try:
        models.extend(API_LLMS)
        
    except Exception as e:
        print(f"Error adding ChatGPT models: {e}")
    
    return models

def main():
    # create args
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="gemma2:27b")
    parser.add_argument("--tutor_model", type=str, default="gemma2:27b")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--student_level", type=str, default="Middle School")
    args = parser.parse_args()
    
    # Define math problems with their solutions
    math_problems = [
        {"problem": "Solve for x: 2x - 5 = 11", "solution": "x = 8"},
        {"problem": "Find the derivative of f(x) = x^3 + 2x^2 - 4x + 7", "solution": "f'(x) = 3x^2 + 4x - 4"},
        {"problem": "Evaluate ∫(2x + 3)dx from x=1 to x=4", "solution": "24"},
        {"problem": "If P(A) = 0.3 and P(B) = 0.5 and A and B are independent, what is P(A and B)?", "solution": "0.15"},
        {"problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "solution": "72"},
        # in arabic {"problem": "Solve for x: 2x - 5 = 11", "solution": "x = 8"}
        # {"problem": "حل المعادلة: 2س - 5 = 11", "solution": "س = 8"},
    ]

    # create results directory
    os.makedirs("results", exist_ok=True)

    for i, problem in enumerate(math_problems):
        conversation = create_conversation(args.student_model, args.tutor_model, problem["problem"], problem["solution"], args.language, args.student_level)
        
        # save conversation to file
        with open(f"results/conversation_{i}.json", "w") as f:
            json.dump(conversation, f, indent=4)

if __name__ == "__main__":
    main()
