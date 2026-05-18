"""
Test baseline accuracy for Qwen3-4B-Base on GSM8K.

This script tests the model's accuracy without any ablation.
"""

import os
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
GSM8K_FILE = "/data/zjj/Synergistic_Core/data/gsm8k/json/test.json"
NUM_QUESTIONS = 50  # Test on 50 questions
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256


def load_gsm8k_samples(num_samples: int = 50) -> list:
    """Load GSM8K dataset samples."""
    print(f"Loading GSM8K dataset from {GSM8K_FILE}...")

    with open(GSM8K_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Sample questions
    import numpy as np
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]

    print(f"Loaded {len(samples)} questions")
    return samples


def format_qwen3_prompt(question: str) -> str:
    """Format question for Qwen3 chat template."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def extract_gsm8k_answer(text: str) -> str:
    """Extract final answer from GSM8K output."""
    # First try: Look for #### answer format
    if "####" in text:
        return text.split("####")[-1].strip().split()[0]

    # Second try: Look for "The answer is:" pattern
    if "The answer is:" in text or "the answer is:" in text:
        parts = re.split(r'[Tt]he answer is:', text)
        if len(parts) > 1:
            answer_text = parts[-1].strip()
            numbers = re.findall(r'\d+\.?\d*', answer_text)
            if numbers:
                return numbers[0]

    # Third try: Get the last number in the entire text
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return ""


def evaluate_baseline_accuracy(model, tokenizer, samples: list, batch_size: int = 8) -> float:
    """Evaluate model accuracy on GSM8K samples."""
    model.eval()
    correct = 0
    total = len(samples)

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch_samples = samples[i:i + batch_size]
        prompts = [format_qwen3_prompt(s['question']) for s in batch_samples]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode outputs
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted_answer = extract_gsm8k_answer(generated_text)
            correct_answer = extract_gsm8k_answer(batch_samples[j]['answer'])

            if predicted_answer and correct_answer:
                pred_norm = predicted_answer.replace(',', '').replace(' ', '')
                ans_norm = correct_answer.replace(',', '').replace(' ', '')
                if pred_norm == ans_norm:
                    correct += 1

    return correct / total


def main():
    print("=" * 60)
    print("Qwen3-4B-Base Baseline Accuracy Test")
    print("=" * 60)

    # Load data
    print("\n1. Loading GSM8K samples...")
    samples = load_gsm8k_samples(NUM_QUESTIONS)

    # Load model
    print("\n2. Loading Qwen3-4B-Base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   Model loaded")

    # Evaluate
    print(f"\n3. Evaluating on {NUM_QUESTIONS} GSM8K questions...")
    accuracy = evaluate_baseline_accuracy(model, tokenizer, samples, BATCH_SIZE)

    print(f"\n" + "=" * 60)
    print(f"Baseline Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {int(accuracy * NUM_QUESTIONS)}/{NUM_QUESTIONS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
