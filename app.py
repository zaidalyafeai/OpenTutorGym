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

def get_ollama_response(model: str, contexts: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Get streaming response from Ollama model"""
    response = ollama.chat(
        model=model,
        messages=contexts,
        stream=True
    )
    
    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']
        time.sleep(0.01)  # Small delay for smoother streaming

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
        stream=True
    )
    for chunk in chat_completion:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def get_llm_response(model: str, contexts: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Get streaming response from LLM"""
    if model in API_LLMS:
        return get_chatgpt_response(model, contexts)
    else:
        return get_ollama_response(model, contexts)

def agent_conversation(
    student_model: str,
    tutor_model: str, 
    question: str,
    solution: str,
    language: str,
    human_text: str,
    human_conversation: List[Dict[str, Any]],
    tutor_conversation: List[Dict[str, Any]],
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
    print(human_conversation)
    if human_text != "":
        if human_conversation != []:
            student_messages = human_conversation
            tutor_messages = tutor_conversation
            for message in student_messages:
                if message["role"] == "assistant":
                    conversation.append({"role": "Student", "content": message["content"]})
                elif message["role"] == "user":
                    conversation.append({"role": "Tutor", "content": message["content"]})
                else:
                    pass
                yield conversation.copy()
        else:
            student_messages = [{"role": "system", "content": student_system}, {"role": "user", "content": question}]
            tutor_messages = [{"role": "system", "content": tutor_system}, {"role": "assistant", "content": question}]
    else:
        student_messages = [{"role": "system", "content": student_system}, {"role": "user", "content": question}]
        tutor_messages = [{"role": "system", "content": tutor_system}, {"role": "assistant", "content": question}]

    
    for turn in range(max_turns): 
        response = ""
        student_message = {"role": "Student", "content": ""}
        conversation.append(student_message)
        
        if human_text != "":
            for chunk in human_text:
                response += chunk
                student_message["content"] = response
                yield conversation.copy()
        else:
            for chunk in get_llm_response(student_model, student_messages):
                response += chunk
                student_message["content"] = response
                yield conversation.copy()

        student_messages.append({"role": "assistant", "content": response})
        tutor_messages.append({"role": "user", "content": response})
        human_conversation.append({"role": "assistant", "content": response})
        tutor_conversation.append({"role": "user", "content": response})

        # Tutor's turn
        tutor_message = {"role": "Tutor", "content": ""}
        conversation.append(tutor_message)

        response = ""
        for chunk in get_llm_response(tutor_model, tutor_messages):
            response += chunk
            tutor_message["content"] = response
            yield conversation.copy()
        
        student_messages.append({"role": "user", "content": response})
        tutor_messages.append({"role": "assistant", "content": response})
        human_conversation.append({"role": "user", "content": response})
        tutor_conversation.append({"role": "assistant", "content": response})

        if '<END>' in response:
            break
        if human_text != "":
            break

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

def create_app():
    # Get available models when app starts
    available_models = get_available_models()
    default_student_model = "gemma2:27b"
    default_tutor_model = "gemma2:27b"
    
    # Define math problems with their solutions
    math_problems = [
        {"problem": "Solve for x: 2x - 5 = 11", "solution": "x = 8"},
        {"problem": "Find the derivative of f(x) = x^3 + 2x^2 - 4x + 7", "solution": "f'(x) = 3x^2 + 4x - 4"},
        {"problem": "Evaluate ∫(2x + 3)dx from x=1 to x=4", "solution": "24"},
        {"problem": "If P(A) = 0.3 and P(B) = 0.5 and A and B are independent, what is P(A and B)?", "solution": "0.15"},
        {"problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "solution": "72"},
        # in arabic {"problem": "Solve for x: 2x - 5 = 11", "solution": "x = 8"}
        {"problem": "حل المعادلة: 2س - 5 = 11", "solution": "س = 8"},
    ]
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Math Tutoring Session")
        gr.Markdown("Watch as a student and tutor work through a math problem together. The tutor provides hints rather than complete solutions.")
        
        human_conversation = gr.State([])
        tutor_conversation = gr.State([])

    
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
                student_level = gr.Dropdown(
                    choices=["Kindergarten", "Elementary School", "Middle School", "High School"],
                    value="Middle School",
                    label="Student Level"
                )
            
            with gr.Column():
                # Replace textbox with dropdown for math problems
                problem_dropdown = gr.Dropdown(
                    choices=[f"{p['problem']} | Solution: {p['solution']}" for p in math_problems],
                    value=f"{math_problems[0]['problem']} | Solution: {math_problems[0]['solution']}",
                    label="Math Problem",
                )
                # Remove solution field
                max_turns = gr.Slider(
                    minimum=1, 
                    maximum=20, 
                    value=20, 
                    step=1, 
                    label="Max Turns",
                    visible=False
                )
                language = gr.Dropdown(
                    choices=["English", "Arabic"],
                    value="English",
                    label="Language"
                )
        
        start_btn = gr.Button("Start A Tutoring Session")
        
        chatbot = gr.Chatbot(
            avatar_images=('assets/student.png','assets/tutor.png'),
            height=600,
            show_copy_button=True,
            label="Tutoring Session"
        )

        human_text = gr.Textbox(visible=0, placeholder="Type your response here...")

        def update_visibility(student_model):  # Accept the event argument, even if not used
            if student_model == "Human":
                return [gr.Textbox('', visible=1), gr.Button("Start A Tutoring Session", visible=False)] #make it visible
            else:
                return [gr.Textbox(visible=0), gr.Button(visible=1)]
        
        student_model.change(update_visibility, student_model, [human_text, start_btn])
        
        def start_conversation(student, tutor, problem_with_solution, language, human_text, human_conversation, tutor_conversation, student_level):
            # Parse the problem and solution from the dropdown value
            question, solution = problem_with_solution.split(" | Solution: ")
            
            for messages in agent_conversation(student, tutor, question, solution, language, human_text, human_conversation, tutor_conversation, student_level):
                formatted_messages = []
                for msg in messages:
                    if msg["role"] == "Student":
                        formatted_messages.append((msg["content"], None))
                    else:  # Tutor
                        formatted_messages.append((None, msg["content"]))
                if human_text != "":
                    yield formatted_messages, ""  # Return empty string to clear human_text
                else:
                    yield formatted_messages
        
        start_btn.click(
            start_conversation,
            inputs=[student_model, tutor_model, problem_dropdown, language, human_text, human_conversation, tutor_conversation, student_level],
            outputs=chatbot
        )

        human_text.submit(
            start_conversation,
            inputs=[student_model, tutor_model, problem_dropdown, language, human_text, human_conversation, tutor_conversation, student_level],
            outputs=[chatbot, human_text]  # Add human_text to outputs
        )
    
    return demo

demo = create_app()

if __name__ == "__main__":
    demo.launch(
        debug=True,         # Enables debugging features
        server_name="0.0.0.0",  # Makes the server accessible from other devices on the network
        server_port=7777
    )
