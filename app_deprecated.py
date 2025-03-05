import gradio as gr
import time
import json
import requests
from typing import List, Dict, Generator, Tuple, Optional

# Ollama API Integration
def get_available_ollama_models() -> List[str]:
    """Get list of available models from Ollama API."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data["models"]]
        else:
            return ["llama2", "mistral", "vicuna"]  # Fallback to common models
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return ["llama2", "mistral", "vicuna"]  # Fallback to common models

def generate_ollama_response(prompt: str, model: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
    """Generate streaming response from Ollama API."""
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    if system_prompt:
        data["system"] = system_prompt
    
    try:
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            response_text = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    response_text += json_response.get("response", "")
                    yield response_text
    except Exception as e:
        yield f"Error: Could not connect to Ollama API. Make sure Ollama is running on your machine. Details: {str(e)}"

# Collaborative problem-solving with agent chat
def agent_chat(problem: str, model: str, max_turns: int) -> Generator[str, None, None]:
    """Have agents chat with each other to solve a problem."""
    system_prompt = """You are a collaborative problem-solving system with two distinct personas:
    1. Brainstorming Agent: Creative, explores multiple approaches, considers different angles
    2. Solving Agent: Practical, develops concrete solutions, focuses on implementation details
    
    These two personas will work together through dialogue to solve a problem. Each message should be
    prefixed with either "Brainstorming Agent:" or "Solving Agent:" to indicate which persona is speaking.
    """
    
    # Initialize the chat with the problem statement
    chat_history = f"Problem to solve: {problem}\n\n"
    chat_history += "Brainstorming Agent: Let me explore some possible approaches to this problem...\n\n"
    
    # First turn is always from the brainstorming agent
    current_turn = 1
    current_agent = "brainstorming"
    
    while current_turn <= max_turns:
        if current_agent == "brainstorming":
            prompt = f"""Continue the following problem-solving dialogue. Respond ONLY as the Brainstorming Agent.
            
Chat history:
{chat_history}

Brainstorming Agent: """
            
            next_agent = "solving"
            agent_prefix = "Brainstorming Agent: "
        else:
            prompt = f"""Continue the following problem-solving dialogue. Respond ONLY as the Solving Agent.
            
Chat history:
{chat_history}

Solving Agent: """
            
            next_agent = "brainstorming"
            agent_prefix = "Solving Agent: "
        
        # Get response from Ollama
        response_text = ""
        for chunk in generate_ollama_response(prompt, model, system_prompt):
            # Extract just the new agent's response
            if isinstance(chunk, str) and chunk.startswith("Error:"):
                response_text = chunk
                break
                
            response_text = chunk
            # Update the chat history for display
            display_history = chat_history
            if response_text:
                display_history += agent_prefix + response_text
            
            yield display_history
        
        # Update chat history with the new response
        chat_history += agent_prefix + response_text + "\n\n"
        
        # Switch to the next agent
        current_agent = next_agent
        current_turn += 1
        
        # If this is the last turn, add a conclusion prompt
        if current_turn == max_turns:
            prompt = f"""Continue the following problem-solving dialogue. You are the Solving Agent.
            Provide a FINAL CONCLUSION that summarizes the solution to the problem.
            
Chat history:
{chat_history}

Solving Agent (FINAL CONCLUSION): """
            
            response_text = ""
            for chunk in generate_ollama_response(prompt, model, system_prompt):
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    response_text = chunk
                    break
                    
                response_text = chunk
                display_history = chat_history
                if response_text:
                    display_history += "Solving Agent (FINAL CONCLUSION): " + response_text
                
                yield display_history
            
            chat_history += "Solving Agent (FINAL CONCLUSION): " + response_text
            break
    
    yield chat_history

# Create the Gradio interface
def create_app():
    # Get available models
    available_models = get_available_ollama_models()
    
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# Collaborative Agent Problem-Solving with Ollama")
        gr.Markdown("""
        This app uses two Ollama-powered LLM agents chatting with each other to solve a problem:
        
        1. **Brainstorming Agent**: Explores possible approaches and ideas
        2. **Solving Agent**: Refines those ideas into a concrete solution
        
        Select a model and enter your problem below.
        
        **Note**: Ensure Ollama is running locally on the default port (11434).
        """)
        
        with gr.Row():
            problem_input = gr.Textbox(
                label="Problem Description",
                placeholder="Describe the problem you want the agents to solve...",
                lines=3
            )
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else "llama2",
                label="Model"
            )
            
            max_turns_slider = gr.Slider(
                minimum=2,
                maximum=10,
                value=4,
                step=1,
                label="Maximum Chat Turns",
                info="Higher values allow more back-and-forth but take longer"
            )
            
            refresh_button = gr.Button("Refresh Models")
            solve_button = gr.Button("Solve Problem", variant="primary")
        
        with gr.Row():
            chat_output = gr.Markdown(
                label="Agent Conversation",
                show_copy_button=True
            )
        
        def refresh_models():
            models = get_available_ollama_models()
            return gr.Dropdown.update(choices=models, value=models[0] if models else "llama2")
        
        refresh_button.click(
            fn=refresh_models,
            inputs=[],
            outputs=[model_dropdown]
        )
        
        solve_button.click(
            fn=agent_chat,
            inputs=[problem_input, model_dropdown, max_turns_slider],
            outputs=[chat_output]
        )
        
        gr.Examples(
            [
                ["Design a recommendation system for an e-commerce website"],
                ["Create an algorithm to detect fraudulent transactions"],
                ["Build a scheduling system for a hospital's appointments"]
            ],
            problem_input
        )
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()