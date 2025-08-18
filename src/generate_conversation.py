from tqdm import tqdm
from predict import LLMGenerator, LLM
from Datasets import GSM8K, StepVerify
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--tutor_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    student_model = LLM(args.student_model)
    tutor_model = LLM(args.tutor_model)

    predictor = LLMGenerator(student_model=student_model, tutor_model=tutor_model)

    dataset = GSM8K(split = args.split)
    dataset = dataset.process_dataset()

    for i, problem in tqdm(enumerate(dataset), total=10):
        output = []
        for conversation in predictor.predict_conversation(
            problem["question"],
            stream = False,
            log = True
        ):
            output.append(conversation)
        print(output[-1])
        if i == 10:
            break

    # evaluate
    
if __name__ == "__main__":

    # asyncio.run(main())
    main()