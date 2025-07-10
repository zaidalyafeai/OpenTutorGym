import json
from tqdm import tqdm
from predict import LLMPredictor
import os
import asyncio
from Datasets import GSM8K

async def process_dataset(
        dataset_name: str,
        split: str,
        language: str = "English",
        student_level: str = "Middle School",
        max_dataset_size: int = 100,
        predictor: LLMPredictor = None
    ):
        """Process a dataset and save the conversations"""
        # Load dataset
        dataset = GSM8K()
        dataset = dataset.process_dataset()

        # Create hash for results directory
        results_path = f"examples/output"
        os.makedirs(results_path, exist_ok=True)
        config = {
                "student_model": predictor.student_model.model_name,
                "tutor_model": predictor.tutor_model.model_name,
                "language": language,
                "student_level": student_level,
            }
        with open(f"{results_path}/config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        # Process each problem
        for i, problem in tqdm(enumerate(dataset)):
            answer, conversation = await predictor.predict_conversation(
                problem["question"],
                language,
                student_level,
                log = True
            )
            print("conversation length: ", len(conversation))
            
            
            # Save conversation to file
            question_id = problem['id']
            
            results = {
                "id": question_id,
                "question": problem["question"],
                "answer": problem["answer"],
                "model_answer": answer,
                "conversation": conversation,
            }
            
            with open(f"{results_path}/{question_id}.json", "w") as f:
                json.dump(results, f, indent=4)

async def main():
    student_model = "unsloth/Qwen3-4B"
    tutor_model = "unsloth/Qwen3-4B"
    predictor = LLMPredictor(mode="tutor", student_model=student_model, tutor_model=tutor_model)

    await process_dataset(
        "gsm8k",
        "train",
        predictor = predictor,
        max_dataset_size=16

        ),

if __name__ == "__main__":
    asyncio.run(main())