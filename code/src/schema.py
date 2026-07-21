from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json

@dataclass
class MultimodalInstance:
    """
    Common schema for a multimodal instance across all datasets.
    Based on the instance definition x = (I, q, y_hat, M) from the plan.
    """
    id: str
    dataset_name: str
    image_path: str
    question: str
    question_type: str  # e.g., "yes_no", "open_ended", "true_false"
    ground_truth: Any
    
    # These will be populated after the clean pass prediction
    model_name: Optional[str] = None
    raw_answer: Optional[str] = None
    normalized_answer: Optional[str] = None
    confidence: Optional[float] = None
    
    # Metadata specific to the dataset (e.g., visual vs language split for HallusionBench)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
