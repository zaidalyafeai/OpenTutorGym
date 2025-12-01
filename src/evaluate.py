from typing import List, Dict, Any
from src.metrics import load_metric
import concurrent.futures
from tqdm import tqdm
from rich import print

class Evaluator:
    """
    Evaluator class, used to evaluate the model on a list of conversations
    """
    def __init__(self, model, metric_names: list[str]):
        self.model = model
        self.metric_names = metric_names
        self.metric = load_metric(metric_names)
    
    def evaluate_conversation(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Evaluate a single conversation.
        
        Args:
            contexts: A list of message dicts with 'role' and 'content' keys.
            
        Returns:
            Dict containing metrics for the conversation.
        """
        content = ""
        for turn in contexts:
            role = "student" if turn['role'] == "user" else "tutor"
            content += f"{role}: {turn['content']}\n"
        response = self.model.get_llm_response([{"role": "system", "content": self.metric.system_prompt}, {"role": "user", "content": content}])
        return self.metric.calculate_score(response)

    def evaluate(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
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
            results = [{metric_name: 0.0 for metric_name in self.metric_names}] * len(conversations)
            
            pbar = tqdm(concurrent.futures.as_completed(future_to_index), total=len(conversations), desc="Evaluating")
            
            for future in pbar:
                response = future.result()
                original_index = future_to_index[future]
                results[original_index] = response
        return results

