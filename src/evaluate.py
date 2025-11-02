from typing import List, Dict, Any
from src.metrics import load_metric
import concurrent.futures
from tqdm import tqdm
from rich import print

class Evaluator:
    def __init__(self, model, metric):
        self.model = model
        self.metric = load_metric(metric)
    
    def evaluate_conversation(self, contexts: List[Dict[str, Any]]) -> float:
        response = self.model.get_llm_response([{"role": "system", "content": self.metric.system_prompt}] + contexts)
        try:
            return self.metric.calculate_score(response)
        except Exception as e:
            return 0.0

    def evaluate(self, conversations: List[List[Dict[str, Any]]], aggregate = True) -> Dict[str, Any]:
        """
        Evaluate multiple conversations in parallel and return aggregated metrics.
        
        Args:
            conversations: A list of conversation lists, where each conversation is a list of message dicts
                          with 'role' and 'content' keys.
                          
        Returns:
            Dict containing aggregated metrics for all conversations.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create futures with their original indices
            future_to_index = {executor.submit(self.evaluate_conversation, conversations[i]): i for i in range(len(conversations))}
            
            # Initialize results list with None values
            results = [0.0] * len(conversations)
            
            pbar = tqdm(concurrent.futures.as_completed(future_to_index), total=len(conversations), desc="Evaluating")
            
            for future in pbar:
                try:
                    response = future.result()
                    original_index = future_to_index[future]
                    results[original_index] = response
                            
                except Exception as e:
                    print(f"Error evaluating conversation: {str(e)}")
                    # Add default values for failed evaluations
                    original_index = future_to_index[future]
                    results[original_index] = 0.0
                    
        return results if not aggregate else sum(results) / len(results)

