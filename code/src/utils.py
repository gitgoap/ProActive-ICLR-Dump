import re
import json
import os
import math
from typing import Any, Dict, Optional, Union

def normalize_answer(raw_text: str, question_type: str = "yes_no") -> str:
    """
    Normalizes the raw generated answer from the MLLM.
    """
    if not raw_text:
        return ""
    
    # Lowercase and strip
    text = raw_text.lower().strip()
    
    # Remove standard punctuation
    text = re.sub(r'[.,!?]', '', text)
    
    # Map to standard answers based on question type
    if question_type in ["yes_no", "true_false"]:
        if "yes" in text or "true" in text or "correct" in text:
            return "yes"
        elif "no" in text or "false" in text or "incorrect" in text:
            return "no"
        else:
            return text  # fallback for ambiguous answers
            
    return text

def extract_confidence(model_output: Any, logits: Any = None) -> Optional[float]:
    """
    Extracts the scalar confidence from the model's output.
    Depending on the model, this could be the probability of the first generated token.
    (Mock implementation: expects either a direct probability score or calculates it from logits)
    """
    if logits is not None:
        # Example logic if logits are provided for the generated tokens
        import torch
        # Assuming logits is a 1D tensor for the first generated token
        probs = torch.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob
    
    # Fallback/mock if logits aren't provided in the clean harness yet
    return 0.95

class JSONLLogger:
    """
    Logs instances and probe observations to JSONL format for easy processing.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def log(self, data: Union[Dict, Any]):
        """Append a single record to the JSONL file."""
        # Convert objects with to_dict method (like MultimodalInstance)
        if hasattr(data, 'to_dict'):
            record = data.to_dict()
        else:
            record = data
            
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
