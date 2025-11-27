# exam.py
import sys
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# allow running this file from project root similar to your original script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import Evaluator, LLM, LLMGenerator
from src.Datasets import GSM8K

class Exam:
    """
    Exam orchestrates generating student answers for GSM8K problems and evaluating them.
    Usage:
        exam = Exam(student_model_spec, judge_model_spec)
        results = exam.run(num_examples=50)
    Results is the raw output returned by Evaluator.evaluate plus a summary dict.
    """

    def __init__(
        self,
        student_model_name: str = "gemma-3-12b-it",
        student_backend: str = "vllm",
        judge_model_name: str = "openai/gpt-4o",
        judge_backend: str = "openrouter",
        generator_mode: str = "standard",
        metric: str = "correct_answer",
        eval_answer: bool = True,
    ):
        # initialize models + generator + judge
        self.student_model = LLM(student_model_name, backend=student_backend)
        self.judge_model = LLM(judge_model_name, backend=judge_backend)
        self.generator = LLMGenerator(student_model=self.student_model, mode=generator_mode)
        self.judge = Evaluator(self.judge_model, metric=metric, eval_answer=eval_answer)

    def load_dataset(self, process: bool = True) -> List[Dict[str, Any]]:
        """Load GSM8K dataset and optionally call its process_dataset() method."""
        dataset = GSM8K()
        if process and hasattr(dataset, "process_dataset"):
            dataset = dataset.process_dataset()
        return dataset

    def _build_conversations(self, examples: List[Dict[str, Any]], answers: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
        """
        Build the conversation format expected by Evaluator.evaluate.
        Each conversation is a list of message dicts with 'role' and 'content'.
        """
        conversations = []
        for ex, ans in zip(examples, answers):
            # Compose a single message containing student answer and gold answer for the judge
            student_solution = ans.get("solution", str(ans))  # defensive fallback
            gold_answer = ex.get("answer", ex.get("label", ""))  # depending on dataset field names
            content = f"Answer: {student_solution}\nGold: {gold_answer}"
            conversations.append([{"role": "user", "content": content}])
        return conversations

    def run(self, num_examples: int = 10, verbose: bool = True, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run exam on the first `num_examples` examples from GSM8K.
        Returns a dict with keys: 'raw_results', 'conversations', 'examples', 'answers', 'summary'.
        """
        if seed is not None:
            try:
                import random
                random.seed(seed)
            except Exception:
                pass


        dataset = self.load_dataset()
        if num_examples <= 0:
            raise ValueError("num_examples must be > 0")
        if len(dataset) < num_examples:
            raise ValueError(f"Dataset length ({len(dataset)}) < num_examples ({num_examples})")

        examples = [dataset[i] for i in range(num_examples)]

        if verbose:
            print(f"[Exam] Generating answers for {num_examples} examples using {self.student_model}")
        answers = self.generator.predict(examples)

        conversations = self._build_conversations(examples, answers)

        if verbose:
            print(f"[Exam] Sending {len(conversations)} conversations to judge {self.judge_model} for evaluation...")

        raw_results = self.judge.evaluate(conversations)
        summary = self._summarize_results(raw_results)

        if verbose:
            print("[Exam] Summary:", summary)

        return {
            "raw_results": raw_results,
            "conversations": conversations,
            "examples": examples,
            "answers": answers,
            "summary": summary,
        }

    def _summarize_results(self, raw_results: Any) -> Dict[str, Any]:
        """
        Attempt to extract per-example correctness and produce counts.
        This is defensive because Evaluator.evaluate may have different return shapes.
        """
        summary = {"total": 0, "correct": 0, "incorrect": 0, "details": None}
        # If raw_results is a list/dict with per-example entries, try to parse it
        try:
            if isinstance(raw_results, list):
                summary["total"] = len(raw_results)
                correct_count = 0
                details = []
                for r in raw_results:
                    # Try several common keys where a correctness flag might appear
                    is_correct = None
                    if isinstance(r, dict):
                        for key in ("correct", "is_correct", "correct_answer", "label"):
                            if key in r:
                                val = r[key]
                                # normalize boolean-like values
                                if isinstance(val, bool):
                                    is_correct = val
                                elif isinstance(val, (int, float)) and val in (0, 1):
                                    is_correct = bool(val)
                                elif isinstance(val, str) and val.lower() in ("true", "yes", "correct"):
                                    is_correct = True
                                elif isinstance(val, str) and val.lower() in ("false", "no", "incorrect"):
                                    is_correct = False
                                if is_correct is not None:
                                    break
                        # fallback: if there's a 'score' numeric where 1 means correct
                        if is_correct is None and "score" in r and isinstance(r["score"], (int, float)):
                            is_correct = bool(round(r["score"]))
                    details.append({"raw": r, "is_correct": is_correct})
                    if is_correct:
                        correct_count += 1
                summary["correct"] = correct_count
                summary["incorrect"] = summary["total"] - correct_count
                summary["details"] = details
            elif isinstance(raw_results, dict):
                # maybe it's a dict with aggregated metrics
                summary["details"] = raw_results
                # try to pick up a 'accuracy' or 'correct' field
                acc = None
                for key in ("accuracy", "acc", "correct_rate"):
                    if key in raw_results:
                        acc = raw_results[key]
                        break
                if acc is not None and isinstance(acc, (int, float)):
                    summary["total"] = 1
                    summary["correct"] = acc
            else:
                summary["details"] = raw_results
        except Exception:
            summary["details"] = raw_results
        return summary



# if __name__ == "__main__":
#     exam = Exam()

#     results = exam.run(num_examples=10, verbose=True)
#     import json
#     try:
#         print(json.dumps(results["summary"], indent=2, default=str))
#     except Exception:
#         print(results["summary"])
