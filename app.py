import gradio as gr
import ollama
import time
from typing import List, Dict, Any, Generator
import os
import dotenv
from openai import OpenAI
dotenv.load_dotenv()

GPT_LLMS = ["openai/gpt-4o-mini", "openai/gpt-4o"]
DEEPSEEK_LLMS = ["deepseek/deepseek-chat"]
GEMINI_LLMS = ["google/gemini-1.5-flash", "google/gemini-2.0-pro-exp-02-05:free"]
API_LLMS = GPT_LLMS + DEEPSEEK_LLMS + GEMINI_LLMS

gr.set_static_paths(paths=["assets/"])

def get_ollama_response(model: str, prompt: str, system: str = "") -> Generator[str, None, None]:
    """Get streaming response from Ollama model"""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']
        time.sleep(0.01)  # Small delay for smoother streaming

def get_chatgpt_response(model: str, prompt: str, system: str = "") -> Generator[str, None, None]:
    """Get streaming response from ChatGPT API"""
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

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
        messages=messages,
        stream=True
    )
    for chunk in chat_completion:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def get_llm_response(model: str, prompt: str, system: str = "") -> Generator[str, None, None]:
    """Get streaming response from LLM"""
    if model in API_LLMS:
        return get_chatgpt_response(model, prompt, system)
    else:
        print('using ollama')
        return get_ollama_response(model, prompt, system)

def agent_conversation(
    student_model: str,
    tutor_model: str, 
    initial_prompt: str,
    solution: str,
    max_turns: int
) -> Generator[List[Dict[str, Any]], None, None]:
    """Generate a conversation between a student and a math tutor"""
    
    conversation = []
    
    # System prompts for each agent
    student_system = f"""You are a student struggling with a Math problem: { initial_prompt }. You should ask for hints to solve the problem at hand.  
    A Tutor will help you solve the problem. Try to understand the hints and guidance provided by the tutor. Once you have solved the problem output the final solution in the following format:
    Solution: <solution>
    """
    tutor_system = f"""You are a helpful math tutor. You want to help the student to answer the question: {initial_prompt}.Provide hints and guidance rather than complete solutions. Ask questions to lead the student to discover the answer themselves. 
    Be encouraging and supportive while helping them understand the underlying concepts. Once the student has solved in the following solution format:
    Solution: <solution> verify that the solution is correct by checking {solution}. If the solution is correct congratulate the student and end the conversation.
    """
    
    # Initial student response
    student_message = {"role": "Student", "content": ""}
    conversation.append(student_message)
    
    full_response = ""
    for chunk in get_llm_response(student_model, initial_prompt, student_system):
        full_response += chunk
        student_message["content"] = full_response
        yield conversation.copy()
    
    # Conversation loop
    current_prompt = full_response
    
    for turn in range(max_turns - 1):  # -1 because we already did the first turn
        # Tutor's turn
        tutor_message = {"role": "Tutor", "content": ""}
        conversation.append(tutor_message)
        
        full_response = ""
        for chunk in get_llm_response(tutor_model, current_prompt, tutor_system):
            full_response += chunk
            tutor_message["content"] = full_response
            yield conversation.copy()
        
        current_prompt = full_response
        
        # Break if this was the last turn
        if turn == max_turns - 2 or max_turns == 2:  # Last iteration
            break
            
        # Student's turn
        student_message = {"role": "Student", "content": ""}
        conversation.append(student_message)
        
        full_response = ""
        for chunk in get_llm_response(student_model, current_prompt, student_system):
            full_response += chunk
            student_message["content"] = full_response
            yield conversation.copy()
        
        current_prompt = full_response
        if 'Solution:' in current_prompt:
            max_turns = 2

def get_available_models():
    models = []
    
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

def create_app():
    # Get available models when app starts
    available_models = get_available_models()
    default_student_model = [model for model in available_models if "7b" in model][0]
    default_tutor_model = [model for model in available_models if "9b" in model][0]
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Math Tutoring Session")
        gr.Markdown("Watch as a student and tutor work through a math problem together. The tutor provides hints rather than complete solutions.")
        
        with gr.Row():
            with gr.Column():
                student_model = gr.Dropdown(
                    choices=available_models, 
                    value=default_student_model,
                    label="Student Model"
                )
                tutor_model = gr.Dropdown(
                    choices=available_models, 
                    value=default_tutor_model,
                    label="Tutor Model"
                )
            
            with gr.Column():
                initial_prompt = gr.Textbox(
                    placeholder="Enter a math problem...",
                    value="x^2 + 2x + 1 = 0",
                    label="Math Problem",
                    lines=1
                )
                solution = gr.Textbox(
                    label="Solution",
                    value="x = -1",
                    lines=1
                )
                max_turns = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=10, 
                    step=1, 
                    label="Max Turns"
                )
        
        start_btn = gr.Button("Start A Tutoring Session")
        
        chatbot = gr.Chatbot(
            avatar_images=('assets/student.png','assets/tutor.png'),
            height=600,
            show_copy_button=True,
            label="Tutoring Session"
        )
        
        def start_conversation(student, tutor, prompt, solution,turns):
                        
            for messages in agent_conversation(student, tutor, prompt, solution, turns):
                formatted_messages = []
                for msg in messages:
                    if msg["role"] == "Student":
                        formatted_messages.append((msg["content"], None))
                    else:  # Tutor
                        formatted_messages.append((None, msg["content"]))
                yield formatted_messages
        
        start_btn.click(
            start_conversation,
            inputs=[student_model, tutor_model, initial_prompt, solution, max_turns],
            outputs=chatbot
        )
    
    return demo

demo = create_app()

if __name__ == "__main__":
    demo.launch(
        debug=True,         # Enables debugging features
        server_name="0.0.0.0",  # Makes the server accessible from other devices on the network
        server_port=7860
    )
