from openai import OpenAI
import asyncio
from typing import List

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Define multiple conversations
conversations = [
    [
        {"role": "system", "content": "Write a short story about a dog."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    [
        {"role": "system", "content": "Write a short story about a cat."},
        {"role": "user", "content": "What is the capital of Germany?"}
    ],
    [
        {"role": "system", "content": "Write a short story about a llama."},
        {"role": "user", "content": "What is the capital of Italy?"}
    ],
]

# Synchronous approach (one after another)
def process_conversations_sync(conversations: List[List[dict]]):
    for i, messages in enumerate(conversations):
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct",
            messages=messages
        )
        print(f"Conversation {i+1}:")
        print(response.choices[0].message.content)
        print("-" * 30)

# Asynchronous approach (parallel processing)
async def process_conversations_async(conversations: List[List[dict]]):
    async def get_completion(messages, idx):
        # Create an async-compatible client if needed
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="Qwen/Qwen2.5-3B-Instruct",
            messages=messages
        )
        return idx, response.choices[0].message.content

    tasks = [get_completion(messages, i) for i, messages in enumerate(conversations)]
    results = await asyncio.gather(*tasks)
    
    # Sort results by original index and print
    for idx, content in sorted(results):
        print(f"Conversation {idx+1}:")
        print(content)
        print("-" * 30)

# process_conversations_sync(conversations)  # Synchronous approach

if __name__ == "__main__":
    asyncio.run(process_conversations_async(conversations))
