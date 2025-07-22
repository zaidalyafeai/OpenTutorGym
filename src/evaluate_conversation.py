from tqdm import tqdm
from predict import Evaluator
from Datasets import StepVerify
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="google/gemma-3-27b-it")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    judge = Evaluator(args.judge_model)

    dataset = StepVerify()
    dialouges = dataset.get_dialouge()
    results = judge.evaluate(dialouges[:10])
    print(results)

    
if __name__ == "__main__":

    # asyncio.run(main())
    main()