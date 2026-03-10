import random
import re

# MBTI Labels

labels = [
    'INTP', 'INTJ', 'INFJ', 'INFP', 'ENTP', 'ENFP', 'ISTP', 'ENTJ',
    'ENFJ', 'ESTP', 'ISTJ', 'ISFP', 'ISFJ', 'ESTJ', 'ESFP', 'ESFJ'
]

# Data Sampling by Class

def sample_data(dataset, num_samples_per_class, labels):
    """
    Randomly sample a fixed number of examples per MBTI class from the dataset.

    Args:
        dataset (list of dict): The dataset entries.
        num_samples_per_class (int): Number of samples to take for each class.
        labels (list): List of MBTI types.

    Returns:
        list: Balanced sampled data across all MBTI types.
    """
    sampled_data = []
    for label in range(len(labels)):
        class_data = [example for example in dataset if example["type"] == label]
        sampled_data.extend(random.sample(class_data, num_samples_per_class))
    random.shuffle(sampled_data)
    return sampled_data


# Formatting Instructions for LLM

def format_instruction(example):
    """
    Create a formatted dictionary with instruction, text, and output label.

    Args:
        example (dict): An example with 'posts' and class index in 'type'.

    Returns:
        dict: Structured prompt for training/inference.
    """
    instruction = """
    Below is a series of social media posts written by a person. Let's think step by step. 
    Determine this person's MBTI type. Please only answer the name of this MBTI type. Do nothing else.
    """

    text = example["posts"]
    output = labels[example["type"]]
    return {
        "instruction": instruction.strip(),
        "text": text,
        "output": output
    }

def make_instruction(dct):
    """
    Generate full LLM prompt with instruction, text, and answer.
    """
    return f"""
{dct['instruction']}

Text: {dct['text']}

Answer: {dct['output']}
""".strip()

def make_instruction_test(dct):
    """
    Generate inference-time prompt without answer.
    """
    return f"""
{dct['instruction']}

Text: {dct['text']}

Answer:""".strip()

# Extract MBTI Prediction

def extract_answer(prediction, valid_categories):
    """
    Extract MBTI label from model output text.

    Args:
        prediction (str): Raw output from model.
        valid_categories (list): List of valid MBTI types.

    Returns:
        str or None: Predicted MBTI type in lowercase if valid, else None.
    """
    normalized_categories = [cat.lower() for cat in valid_categories]

    match = re.search(r"Answer:\s*([\w/]+)", prediction)
    if match:
        extracted = match.group(1).strip().lower()
        if extracted in normalized_categories:
            return extracted

    # Fallback: match any valid type found in prediction text
    for category in normalized_categories:
        if category in prediction.lower():
            return category

    return None
