"""
Quick test to verify baseline accuracy for Qwen3-4B-Base.
Tests on 10 questions to quickly check if the prompting works correctly.
"""

import os
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
GSM8K_FILE = "/data/zjj/Synergistic_Core/data/gsm8k/json/test.json"
NUM_QUESTIONS = 10
MAX_NEW_TOKENS = 1024


def format_qwen3_prompt(question: str) -> str:
    """Format question with 8-shot CoT."""
    few_shot_prefix = """Question: There are 15 trees in the park. Park workers will plant 5 more trees today. How many trees will the park have when the workers are finished?
Answer: There are 15 trees originally. Then 5 trees were added. So 15 + 5 = 20. The park has 20 trees now.
#### 20

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are 3 cars originally. 2 more cars arrive. 3 + 2 = 5. There are 5 cars in the parking lot.
#### 5

Question: Leah had 32 apples and her sister gave her 15 more apples. How many apples does Leah have now?
Answer: Leah had 32 apples. Her sister gave her 15 more. 32 + 15 = 47. Leah has 47 apples now.
#### 47

Question: A box contains 6 red balls and 4 blue balls. How many balls are there in total?
Answer: There are 6 red balls and 4 blue balls. 6 + 4 = 10. There are 10 balls in total.
#### 10

Question: A restaurant had 23 customers for lunch and 18 customers for dinner. How many customers did the restaurant serve that day?
Answer: 23 customers for lunch plus 18 customers for dinner. 23 + 18 = 41. The restaurant served 41 customers.
#### 41

Question: Sarah has 15 notebooks. She buys 12 more notebooks. Then she gives 8 notebooks to her friend. How many notebooks does Sarah have now?
Answer: Sarah starts with 15 notebooks. She buys 12 more, so she has 15 + 12 = 27 notebooks. Then she gives 8 away, so she has 27 - 8 = 19 notebooks.
#### 19

Question: A baker made 56 cookies in the morning and 34 cookies in the afternoon. If he sold 45 cookies, how many cookies does he have left?
Answer: The baker made 56 + 34 = 90 cookies total. He sold 45 cookies, so he has 90 - 45 = 45 cookies left.
#### 45

Question: There are 48 students in a class. If they are divided into groups of 6, how many groups are there?
Answer: 48 students divided into groups of 6. 48 ÷ 6 = 8. There are 8 groups.
#### 8

Question: """

    return few_shot_prefix + question + "\nAnswer:"


def extract_gsm8k_answer(text: str) -> str:
    """Extract final answer from GSM8K output."""
    text = text.strip()

    # Method 1: Standard GSM8K #### format
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        numbers = re.findall(r'\d+\.?\d*', after_hash)
        if numbers:
            return numbers[0]

    # Method 2: Look for "The answer is" or similar patterns
    answer_patterns = [
        r'[Tt]he answer is\s+[:#]?\s*',
        r'[Aa]nswer\s*[:#]\s*',
        r'=\s*\$?',
        r'equals\s*\$?',
    ]

    for pattern in answer_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            after_pattern = text[match.end():].strip().split('\n')[0].split('.')[0]
            numbers = re.findall(r'\d+\.?\d*', after_pattern)
            if numbers:
                return numbers[0]

    # Method 3: Get the last number in the entire text
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return ""


def main():
    print("=" * 60)
    print("Qwen3-4B-Base Quick Accuracy Test")
    print("=" * 60)

    # Load data
    print("\nLoading GSM8K data...")
    with open(GSM8K_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    samples = dataset[:NUM_QUESTIONS]

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Test
    correct = 0
    for i, sample in enumerate(samples):
        question = sample['question']
        true_answer = extract_gsm8k_answer(sample['answer'])

        prompt = format_qwen3_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted_answer = extract_gsm8k_answer(generated)

        # Normalize for comparison
        true_norm = true_answer.replace(',', '').replace(' ', '')
        pred_norm = predicted_answer.replace(',', '').replace(' ', '')

        is_correct = true_norm == pred_norm
        if is_correct:
            correct += 1

        print(f"\n[{i+1}/{NUM_QUESTIONS}] {'✅' if is_correct else '❌'}")
        print(f"Q: {question[:100]}...")
        print(f"True: {true_answer} | Predicted: {predicted_answer}")

        # Show first 2 examples in full
        if i < 2:
            print(f"\nGenerated:\n{generated[:500]}...")

    accuracy = correct / NUM_QUESTIONS
    print(f"\n" + "=" * 60)
    print(f"Accuracy: {accuracy:.3f} ({correct}/{NUM_QUESTIONS})")
    print("=" * 60)


if __name__ == "__main__":
    main()
