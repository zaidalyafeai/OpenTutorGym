from tqdm import tqdm
from predict import LLMGenerator, UnslothLLM
from Datasets import GSM8K, StepVerify
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_model", type=str, default="unsloth/Qwen3-4B")
    parser.add_argument("--tutor_model", type=str, default="unsloth/Qwen3-4B")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    student_model = UnslothLLM(args.student_model)
    tutor_model = UnslothLLM(args.tutor_model)

    predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)

    dataset = GSM8K(split = args.split)
    dataset = dataset.process_dataset()

    for i, problem in tqdm(enumerate(dataset), total=10):
        answer, conversation = predictor.predict_conversation(
            problem["question"],
            log = True
        )
        if i == 10:
            break

    # evaluate
    
if __name__ == "__main__":

    # asyncio.run(main())
    main()