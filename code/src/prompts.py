"""
Prompt templates for clean inference and basic diagnostic probes.
"""

CLEAN_INFERENCE_PROMPTS = {
    "yes_no": "{question}\nPlease answer with exactly 'Yes' or 'No'.",
    "true_false": "Statement: {question}\nIs this statement True or False?",
    "open_ended": "{question}\nPlease provide a brief and concise answer.",
    "multiple_choice": "{question}\nOptions:\n{options}\nPlease output the correct option letter."
}

# These are for Week 4, but scaffolding them now as part of the overall design
PROBE_PROMPTS = {
    "grounding": "Look closely at the image again. Are you absolutely sure about your previous answer regarding: '{question}'? Answer Yes or No.",
    "relation": "Consider the spatial and semantic relationships between the objects carefully. {question} Answer Yes or No.",
}

def get_prompt(question: str, q_type: str = "yes_no", options: str = "") -> str:
    template = CLEAN_INFERENCE_PROMPTS.get(q_type, CLEAN_INFERENCE_PROMPTS["open_ended"])
    if q_type == "multiple_choice":
        return template.format(question=question, options=options)
    return template.format(question=question)
