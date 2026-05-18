"""
Quick test: Llama-3.1-8B baseline accuracy on GSM8K using Meta's official format.

Uses the same prompt format as lm-eval's gsm8k_cot_llama task:
- 8-shot CoT with "The final answer is X" format
- Llama chat template (multi-turn conversation)
- Stop tokens: <|eot_id|>, Q:

Usage:
    python src/test_llama_baseline.py
    python src/test_llama_baseline.py 50   # test 50 questions
"""

import os
import sys
import json
import re
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──
MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
GSM8K_FILE = "/data/zjj/Synergistic_Core/data/gsm8k/json/test.json"
NUM_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RANDOM_SEED = 42
MAX_NEW_TOKENS = 512

# ── 8-shot examples (from gsm8k-cot-llama.yaml) ──
FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
    },
]

# ── Llama chat template tokens ──
BOS = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"


def build_llama_prompt(question: str) -> str:
    """Build 8-shot CoT prompt using Llama chat template (multi-turn).

    Format matches lm-eval's gsm8k_cot_llama with --apply_chat_template --fewshot_as_multiturn.
    """
    messages = []

    # 8 few-shot examples as alternating user/assistant turns
    for ex in FEWSHOT_EXAMPLES:
        user_msg = (
            f'Given the following problem, reason and give a final answer to the problem.\n'
            f'Problem: {ex["question"]}\n'
            f'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
        )
        messages.append(("user", user_msg))
        messages.append(("assistant", ex["target"]))

    # Actual question
    user_msg = (
        f'Given the following problem, reason and give a final answer to the problem.\n'
        f'Problem: {question}\n'
        f'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    )
    messages.append(("user", user_msg))

    # Build the prompt string
    prompt = BOS
    for role, content in messages:
        prompt += f"{START_HEADER}{role}{END_HEADER}\n\n{content}{EOT}"
    # Add assistant header for generation
    prompt += f"{START_HEADER}assistant{END_HEADER}\n\n"

    return prompt


def extract_final_answer(text: str) -> str:
    """Extract answer from 'The final answer is X' format."""
    # Truncate at stop sequences
    for stop in ["<|eot_id|>", "<|start_header_id|>", "\nQ:", "</s>"]:
        if stop in text:
            text = text[:text.index(stop)]

    # Method 1: "The final answer is X"
    matches = list(re.finditer(r'[Tt]he final answer is\s*(-?[\$0-9.,]+)', text))
    if matches:
        answer = matches[-1].group(1)
        return answer.replace(',', '').replace('$', '').strip()

    # Method 2: "The answer is X"
    matches = list(re.finditer(r'[Tt]he answer is\s*(-?[\$0-9.,]+)', text))
    if matches:
        answer = matches[-1].group(1)
        return answer.replace(',', '').replace('$', '').strip()

    # Method 3: "#### X"
    if "####" in text:
        after = text.split("####")[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after.replace(',', ''))
        if numbers:
            return numbers[0]

    # Method 4: Last number
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]

    return ""


def extract_ground_truth(answer: str) -> str:
    """Extract ground truth answer from GSM8K format."""
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(',', '').replace(' ', '')
    numbers = re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))
    return numbers[-1] if numbers else ""


def main():
    print("=" * 60)
    print("Llama-3.1-8B Baseline Accuracy Test (Meta's Official Format)")
    print("=" * 60)

    # Load data
    with open(GSM8K_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(dataset), min(NUM_QUESTIONS, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    print(f"\nLoaded {len(samples)} questions from GSM8K")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded")

    # Get special token IDs for stopping
    eot_id = tokenizer.encode(EOT, add_special_tokens=False)
    print(f"  <|eot_id|> token id: {eot_id}")

    # Build prompts
    prompts = [build_llama_prompt(s['question']) for s in samples]
    ground_truths = [extract_ground_truth(s['answer']) for s in samples]

    print(f"\nFirst prompt (last 300 chars):")
    print(f"  ...{prompts[0][-300:]}")
    print(f"\nFirst ground truth: {ground_truths[0]}")

    # Evaluate
    print(f"\nEvaluating on {NUM_QUESTIONS} questions...")
    model.eval()
    correct = 0
    batch_size = 4  # Conservative for 8-shot prompt (long context)

    for i in range(0, len(samples), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_gt = ground_truths[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
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

        for j, output in enumerate(outputs):
            gen_text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_final_answer(gen_text)
            gt = batch_gt[j]

            match = predicted == gt
            if match:
                correct += 1

            # Print first 3 for debugging
            if i + j < 3:
                print(f"\n  Q{i+j}: GT={gt}, Pred={predicted}, {'CORRECT' if match else 'WRONG'}")
                print(f"  Generated (last 200 chars): ...{gen_text[-200:]}")

        current_acc = correct / min(i + batch_size, NUM_QUESTIONS)
        print(f"  Batch {i//batch_size + 1}: Running accuracy = {current_acc:.3f} ({correct}/{min(i + batch_size, NUM_QUESTIONS)})")

    accuracy = correct / NUM_QUESTIONS
    print(f"\n{'=' * 60}")
    print(f"Baseline Accuracy: {accuracy * 100:.1f}% ({correct}/{NUM_QUESTIONS})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
