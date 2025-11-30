import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Datasets import load_dataset

def main():
    dataset = load_dataset("math_vision")
    dataset = dataset.process_dataset()
    print(dataset[0])

if __name__ == "__main__":
    main()
