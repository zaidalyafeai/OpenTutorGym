import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import LLM
from src.evaluate import Evaluator
from argparse import ArgumentParser
import json 
import pandas as pd
import glob

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="openai/gpt-4o")
    return parser.parse_args()

def main():

    ## generate conversation
    args = parse_args()
    judge = LLM(args.judge_model, backend="openrouter")
    metrics = ['creativity', 'mistake_identification', 'overload', 'actionability', 'correct_answer']    

    files = glob.glob("conversations_*.json")
    results = []
    for file in files:
        model_name = file.split("_")[1].replace(".json", "")
        with open(file, "r") as f:
            conversations = json.load(f)[:50]
        agg_results = {metric: 0 for metric in metrics}
        for metric in metrics:
            evaluator = Evaluator(judge, [metric])
            output = evaluator.evaluate(conversations)
            for result in output:
                agg_results[metric] += result[metric]/len(conversations)
        row = {"model": model_name, **agg_results}
        row['average'] = sum(agg_results.values())/len(metrics)
        results.append(row)
        print(row)
    
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()