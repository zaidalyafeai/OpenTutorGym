import ollama
from typing import List, Dict, Any, Generator
import os
import dotenv
from openai import OpenAI
import json
import argparse
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

dotenv.load_dotenv()

GPT_LLMS = ["openai/gpt-4o-mini", "openai/gpt-4o"]
DEEPSEEK_LLMS = ["deepseek/deepseek-chat"]
GEMINI_LLMS = ["google/gemini-1.5-flash", "google/gemini-2.0-pro-exp-02-05:free"]
API_LLMS = GPT_LLMS + DEEPSEEK_LLMS + GEMINI_LLMS


def get_ollama_response(
    model: str, contexts: List[Dict[str, Any]]
) -> Generator[str, None, None]:
    """Get streaming response from Ollama model"""
    response = ollama.chat(model=model, messages=contexts, stream=False)

    return response["message"]["content"]


def get_chatgpt_response(
    model: str, contexts: List[Dict[str, Any]]
) -> Generator[str, None, None]:
    """Get streaming response from ChatGPT API"""

    if model in API_LLMS:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    else:
        raise ValueError(f"Invalid model: {model}")

    chat_completion = client.chat.completions.create(
        model=model, messages=contexts, stream=False
    )

    return chat_completion.choices[0].message.content


def get_llm_response(
    model: str, contexts: List[Dict[str, Any]], system_prompt: str = None
) -> Generator[str, None, None]:
    """Get streaming response from LLM"""
    if system_prompt is not None:
        contexts.insert(0, {"role": "system", "content": system_prompt})
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
    max_turns: int = 20,
    log: bool = True,
) -> Generator[List[Dict[str, Any]], None, None]:
    """Generate a conversation between a student and a math tutor"""

    # System prompts for each agent
    student_system = f"""
    You speak in {language}.
    Never forget you are a Student in {student_level} and I am a Tutor. Never flip roles! You will always ask questions, never instruct me or ask me to solve the problem.
    I will help you to solve the problem. Here is the problem: {question}. Never forget the problem. You will ask for hints and I will provide you with some guidlines.  
    Try to understand the hints and guidance provided by me. 
    Once you have solved the problem, output the solution in the following format:
    Solution: <solution>
    Only output the final solution as a number do not include any units or other text.
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
        response = get_llm_response(student_model, student_messages)
        if log:
            print("Student: ", response.strip())
        conversation.append({"role": "Student", "content": response})
        student_messages.append({"role": "assistant", "content": response})
        tutor_messages.append({"role": "user", "content": response})

        response = get_llm_response(tutor_model, tutor_messages)
        if log:
            print("Tutor: ", response.strip())
        conversation.append({"role": "Tutor", "content": response})
        student_messages.append({"role": "user", "content": response})
        tutor_messages.append({"role": "assistant", "content": response})

        if "<END>" in response:
            break
    return conversation


def get_available_models():
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

    # Add ChatGPT models
    try:
        models.extend(API_LLMS)

    except Exception as e:
        print(f"Error adding ChatGPT models: {e}")

    return models

def predict_cot(question: str, model: str):
    system_prompt = f"""
    You are given a question. you need to answer the question step by step. Provide the final answer in the following format:
    Solution: <solution>
    Only output the final solution as a number do not include any units or other text.
    """
    contexts = [{"role": "user", "content": question}]
    response = get_llm_response(model, contexts, system_prompt)
    return response
def predict_fewshot(examples: List[Dict[str, Any]], model: str, fewshot_size = 0, question:str = None):
    system_prompt = f"""
    You will given a list of examples. You will need to answer the question based on the examples. Please output your answer in following format:
    Solution: <solution>
    Only output the final solution as a number do not include any units or other text.
    """
    examples = examples[:fewshot_size]
    contexts = []
    for example in examples:
        contexts.append({"role": "user", "content": example["question"]})
        contexts.append({"role": "assistant", "content": example["solution"]})
    
    contexts.append({"role": "user", "content": question})
    response = get_llm_response(model, contexts, system_prompt)
    return response

def extract_answer(
    student_model: str = "gemma2:27b",
    tutor_model: str = "gemma2:27b",
    question: str = "Solve for x: 2x - 5 = 11",
    solution: str = "x = 8",
    language: str = "English",
    student_level: str = "Middle School",
    log: bool = False,
):
    conversation = create_conversation(
        student_model=student_model,
        tutor_model=tutor_model,
        question=question,
        solution=solution,
        language=language,
        student_level=student_level,
        log=log,
    )

    return conversation[-2]["content"]


def main():
    # create args
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="gemma2:27b")
    parser.add_argument("--tutor_model", type=str, default="gemma2:27b")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--student_level", type=str, default="Middle School")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_dataset_size", type=int, default=100)
    args = parser.parse_args()

    # load dataset
    dataset = load_from_disk(f"data/{args.dataset}.parquet")[args.split]
    dataset = dataset.select(range(args.max_dataset_size))
    dataset = dataset.map(lambda x: {
        "question": x["question"],
        "solution": x["answer"].split("####")[1].strip()
    })


    import hashlib
    import os

    hash_path = hashlib.sha256(f"{args.student_model}-{args.tutor_model}".encode()).hexdigest()
    results_path = f"results/{hash_path}/{args.split}"
    os.makedirs(results_path, exist_ok=True)

    for i, problem in tqdm(enumerate(dataset)):
        conversation = create_conversation(
            args.student_model,
            args.tutor_model,
            problem["question"],
            problem["solution"],
            args.language,
            args.student_level,
            log=False
        )
        config = {
            "student_model": args.student_model,
            "tutor_model": args.tutor_model,
            "language": args.language,
            "student_level": args.student_level,
        }
        # save conversation to file
        question_id = problem['id']
        print(problem["question"])
        print(question_id)
        results = {
            "id": question_id,
            "question": problem["question"],
            "solution": problem["solution"],
            "conversation": conversation,
        }
        with open(f"{results_path}/{question_id}.json", "w") as f:
            json.dump(results, f, indent=4)
        with open(f"{results_path}/config.json", "w") as f:
            json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
