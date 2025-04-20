from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from dotenv import load_dotenv
import asyncio
from pydantic import BaseModel

class CityLocation(BaseModel):
    city: str
    country: str

async def main():
    load_dotenv()

    model = OpenAIModel(
        'google/gemini-2.5-flash-preview',
        provider=OpenAIProvider(base_url='https://openrouter.ai/api/v1', api_key=os.environ.get("OPENROUTER_API_KEY")),
    )
    agent = Agent(model)

    response = await agent.run("What is the capital of France?", output_type=CityLocation)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())