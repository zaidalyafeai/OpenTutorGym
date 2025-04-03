from datasets import load_from_disk
import json
from tqdm import tqdm
from predict import LLMPredictor
import os
import asyncio
async def process_dataset(
        dataset_name: str,
        split: str,
        language: str = "English",
        student_level: str = "Middle School",
        max_dataset_size: int = 100,
        generator: LLMPredictor = None
    ):
        """Process a dataset and save the conversations"""
        # Load dataset
        dataset = load_from_disk(f"data/{dataset_name}.parquet")[split]
        dataset = dataset.select(range(max_dataset_size))

        # Create hash for results directory
        results_path = f"examples/{generator.student_model}/{generator.tutor_model}/{dataset_name}/{split}"
        os.makedirs(results_path, exist_ok=True)

        # Process each problem
        for i, problem in tqdm(enumerate(dataset)):
            answer, conversation = await generator.predict_conversation(
                problem["question"],
                language,
                student_level,
            )
            
            config = {
                "student_model": generator.student_model,
                "tutor_model": generator.tutor_model,
                "language": language,
                "student_level": student_level,
            }
            
            # Save conversation to file
            question_id = problem['id']
            
            results = {
                "id": question_id,
                "question": problem["question"],
                "answer": problem["answer"],
                "conversation": conversation,
            }
            
            with open(f"{results_path}/{question_id}.json", "w") as f:
                json.dump(results, f, indent=4)
            with open(f"{results_path}/config.json", "w") as f:
                json.dump(config, f, indent=4)

async def main():
    student_model = "Qwen/Qwen2.5-3B-Instruct"
    tutor_model = "Qwen/Qwen2.5-7B-Instruct"
    generator = LLMPredictor(mode="tutor", student_model=student_model, tutor_model=tutor_model)

    await process_dataset(
        "gsm8k",
        "train",
        generator = generator,
        max_dataset_size=16

        ),

if __name__ == "__main__":
    asyncio.run(main())