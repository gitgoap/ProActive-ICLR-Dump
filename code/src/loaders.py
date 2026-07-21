import os
import json
from typing import List, Dict, Any
from .schema import MultimodalInstance

# Mock paths for illustration. These should point to the actual dataset directories.
DATASET_ROOTS = {
    "POPE": "data/POPE",
    "HallusionBench": "data/HallusionBench",
    "VizWiz-VQA": "data/VizWiz-VQA",
    "VSR": "data/VSR",
    "PRE-HAL": "data/PRE-HAL",
    "IllusionBench": "data/IllusionBench"
}

def load_pope(split: str = "val") -> List[MultimodalInstance]:
    """
    Loads the POPE dataset.
    POPE usually comes in JSONL format with 'image', 'question', 'answer'.
    """
    instances = []
    # Mocking data loading for scaffolding
    # In practice: read from os.path.join(DATASET_ROOTS["POPE"], f"{split}.jsonl")
    
    mock_data = [
        {"image": "COCO_val2014_000000123456.jpg", "question": "Is there a dog in the image?", "answer": "yes", "id": "pope_1"},
        {"image": "COCO_val2014_000000123457.jpg", "question": "Is there a car in the image?", "answer": "no", "id": "pope_2"}
    ]
    
    for item in mock_data:
        instances.append(
            MultimodalInstance(
                id=item["id"],
                dataset_name="POPE",
                image_path=os.path.join(DATASET_ROOTS["POPE"], "images", item["image"]),
                question=item["question"],
                question_type="yes_no",
                ground_truth=item["answer"]
            )
        )
    return instances

def load_hallusionbench() -> List[MultimodalInstance]:
    """
    Loads HallusionBench.
    Contains visual illusion and language hallucination stress tests.
    """
    instances = []
    # Mocking data
    mock_data = [
        {"image": "visual_illusion_1.jpg", "question": "Is the left line longer than the right line?", "answer": "no", "category": "visual_illusion", "id": "hb_1"},
        {"image": "lang_halluc_1.jpg", "question": "What is the color of the apple?", "answer": "red", "category": "language_hallucination", "id": "hb_2"}
    ]
    
    for item in mock_data:
        instances.append(
            MultimodalInstance(
                id=item["id"],
                dataset_name="HallusionBench",
                image_path=os.path.join(DATASET_ROOTS["HallusionBench"], item["image"]),
                question=item["question"],
                question_type="yes_no" if item["answer"] in ["yes", "no"] else "open_ended",
                ground_truth=item["answer"],
                metadata={"category": item["category"]}
            )
        )
    return instances

def load_vizwiz(split: str = "val") -> List[MultimodalInstance]:
    """
    Loads VizWiz-VQA. Focuses on visual fragility and answerability.
    """
    instances = []
    # Mocking data
    mock_data = [
        {"image": "vizwiz_val_001.jpg", "question": "What does this bottle say?", "answers": [{"answer": "milk", "answer_confidence": "yes"}], "answerable": 1, "id": "vw_1"}
    ]
    
    for item in mock_data:
        # Take the most common answer for ground truth, or keep list
        gt = item["answers"][0]["answer"]
        instances.append(
            MultimodalInstance(
                id=item["id"],
                dataset_name="VizWiz-VQA",
                image_path=os.path.join(DATASET_ROOTS["VizWiz-VQA"], item["image"]),
                question=item["question"],
                question_type="open_ended",
                ground_truth=gt,
                metadata={"answerable": item["answerable"]}
            )
        )
    return instances

def load_vsr(split: str = "val") -> List[MultimodalInstance]:
    """
    Loads Visual Spatial Reasoning (VSR). Relation binding stress.
    """
    instances = []
    # Mocking data
    mock_data = [
        {"image": "vsr_001.jpg", "caption": "The cup is to the left of the laptop.", "label": 1, "id": "vsr_1"}
    ]
    
    for item in mock_data:
        instances.append(
            MultimodalInstance(
                id=item["id"],
                dataset_name="VSR",
                image_path=os.path.join(DATASET_ROOTS["VSR"], item["image"]),
                question=item["caption"],
                question_type="true_false",
                ground_truth="yes" if item["label"] == 1 else "no"
            )
        )
    return instances

def load_pre_hal() -> List[MultimodalInstance]:
    """
    Loads PRE-HAL dataset. Perception-reasoning shift.
    """
    instances = []
    # Mocking data
    mock_data = [
        {"image": "pre_hal_001.jpg", "question": "Are there more than 3 cats?", "answer": "yes", "type": "reasoning", "id": "ph_1"}
    ]
    for item in mock_data:
        instances.append(
            MultimodalInstance(
                id=item["id"],
                dataset_name="PRE-HAL",
                image_path=os.path.join(DATASET_ROOTS["PRE-HAL"], item["image"]),
                question=item["question"],
                question_type="yes_no",
                ground_truth=item["answer"],
                metadata={"type": item["type"]}
            )
        )
    return instances

def load_illusionbench() -> List[MultimodalInstance]:
    """
    Loads IllusionBench. Held-out visual illusion dataset.
    """
    instances = []
    # Mocking data
    mock_data = [
        {"image": "illusion_001.jpg", "question": "Are the horizontal lines parallel?", "answer": "yes", "id": "ib_1"}
    ]
    for item in mock_data:
        instances.append(
            MultimodalInstance(
                id=item["id"],
                dataset_name="IllusionBench",
                image_path=os.path.join(DATASET_ROOTS["IllusionBench"], item["image"]),
                question=item["question"],
                question_type="yes_no",
                ground_truth=item["answer"]
            )
        )
    return instances

def load_all_datasets() -> Dict[str, List[MultimodalInstance]]:
    """
    Convenience function to load all datasets into a dictionary.
    """
    return {
        "POPE": load_pope(),
        "HallusionBench": load_hallusionbench(),
        "VizWiz-VQA": load_vizwiz(),
        "VSR": load_vsr(),
        "PRE-HAL": load_pre_hal(),
        "IllusionBench": load_illusionbench()
    }
