"""
Quick test: Llama-3.1-8B baseline accuracy with improved evaluation.

Improvements:
- Custom StoppingCriteria to stop at "####" + number
- Better answer extraction
- Test both 3-shot and 8-shot formats

Usage:
    python src/test_llama_baseline_v3.py
    python src/test_llama_baseline_v3.py 50
"""

import os
import sys
import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from typing import List

MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
GSM8K_FILE = "/data/zjj/Synergistic_Core/data/gsm8k/json/test.json"
NUM_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 1024  # Match Meta's eval setup


class StopAtHashHash(StoppingCriteria):
    """Stop generation when ALL sequences have produced '####' followed by a number."""

    def __init__(self, tokenizer, prompt_lengths: List[int]):
        self.tokenizer = tokenizer
        self.prompt_lengths = prompt_lengths
        self._done = [False] * len(prompt_lengths)

    def __call__(self, input_ids, scores, **kwargs):
        all_done = True
        for i in range(input_ids.shape[0]):
            if self._done[i]:
                continue
            generated = input_ids[i, self.prompt_lengths[i]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            # Check if "####" followed by a number exists
            idx = text.find("####")
            if idx == -1:
                all_done = False
                continue
            after = text[idx + 4:].strip()
            if after:
                # Check if there's a number
                nums = re.findall(r'-?\d+\.?\d*', after.split('\n')[0])
                if nums:
                    self._done[i] = True
                else:
                    all_done = False
            else:
                all_done = False
        return all_done


# ── 3-shot prompt (#### format) ──
FEWSHOT_3 = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: It takes 2/2 = <<2/2=1>>1 bolt of white fiber.
So the total amount of fabric is 2 + 1 = <<2+1=3>>3 bolts of fabric.
#### 3

Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
A: The value of the house increased by 80,000 * 1.5 = $<<80,000*1.5=120,000>>120,000.
So the new value of the house is 80,000 + 120,000 = $<<80,000-80,000+120,000=200,000>>200,000.
The profit is 200,000 - 80,000 - 50,000 = $<<200,000-80,000-50,000=70,000>>70,000.
#### 70,000

Q: """


# ── 8-shot prompt (#### format, Wei et al. style) ──
FEWSHOT_8_EXAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
     "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
     "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
     "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
     "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
     "9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
     "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
     "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
     "8"),
]


def format_3shot(question: str) -> str:
    return FEWSHOT_3 + question + "\nA:"


def format_8shot_hash(question: str) -> str:
    """8-shot with #### format."""
    prompt = ""
    for q, reasoning, ans in FEWSHOT_8_EXAMPLES:
        prompt += f"Q: {q}\nA: {reasoning}\n#### {ans}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def format_8shot_wei(question: str) -> str:
    """8-shot with 'The answer is X.' format (Wei et al. 2022)."""
    prompt = ""
    for q, reasoning, ans in FEWSHOT_8_EXAMPLES:
        prompt += f"Q: {q}\nA: {reasoning} The answer is {ans}.\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    # Truncate at stop sequences
    for stop in ["\nQ:", "</s>", "<|im_end|>", "<|eot_id|>"]:
        if stop in text:
            text = text[:text.index(stop)]

    text = text.strip()

    # Method 1: "#### X" (highest priority, GSM8K standard)
    if "####" in text:
        after = text.split("####")[-1].strip()
        if after:
            nums = re.findall(r'-?\d+', after.split('\n')[0].replace(',', ''))
            if nums:
                return nums[0]

    # Method 2: "The answer is X"
    for pattern in [r'[Tt]he answer is\s+', r'[Tt]he final answer is\s+']:
        matches = list(re.finditer(pattern, text))
        if matches:
            match = matches[-1]
            after = text[match.end():].strip().split('\n')[0].split('.')[0]
            nums = re.findall(r'-?\d+', after.replace(',', '').replace('$', ''))
            if nums:
                return nums[0]

    # Method 3: Last number
    nums = re.findall(r'-?\d+', text.replace(',', ''))
    if nums:
        return nums[-1]
    return ""


def extract_gt(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(',', '').replace(' ', '')
    nums = re.findall(r'-?\d+', answer.replace(',', ''))
    return nums[-1] if nums else ""


def test_format(model, tokenizer, format_name, format_fn, samples, gts, batch_size=4):
    """Test one format and return accuracy."""
    print(f"\n{'='*60}")
    print(f"Testing: {format_name}")
    print(f"{'='*60}")

    prompts = [format_fn(s['question']) for s in samples]
    correct = 0
    wrong_samples = []

    for i in range(0, len(samples), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_gt = gts[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_lengths = [inputs['input_ids'].shape[1]] * len(batch_prompts)
        stop_criteria = StoppingCriteriaList([StopAtHashHash(tokenizer, prompt_lengths)])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=stop_criteria,
            )

        for j, output in enumerate(outputs):
            gen_text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen_text)
            gt = batch_gt[j]
            match = pred == gt
            if match:
                correct += 1
            elif len(wrong_samples) < 3:
                wrong_samples.append((i+j, gt, pred, gen_text[-200:]))

        running_acc = correct / min(i + batch_size, len(samples))
        print(f"  Batch {i//batch_size+1}: Running acc = {running_acc:.3f} ({correct}/{min(i+batch_size,len(samples))})")

    # Show wrong examples
    if wrong_samples:
        print(f"\n  Wrong examples:")
        for idx, gt, pred, gen in wrong_samples:
            print(f"    Q{idx}: GT={gt}, Pred={pred}")
            print(f"      Gen (last 100): ...{gen[-100:]}")

    accuracy = correct / len(samples)
    print(f"\n  >> {format_name} Accuracy: {accuracy*100:.1f}% ({correct}/{len(samples)})")
    return accuracy


def main():
    print("=" * 60)
    print("Llama-3.1-8B Base — Improved GSM8K Baseline Test")
    print(f"Questions: {NUM_QUESTIONS}")
    print("=" * 60)

    # Load data
    with open(GSM8K_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(dataset), min(NUM_QUESTIONS, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    gts = [extract_gt(s['answer']) for s in samples]
    print(f"Loaded {len(samples)} questions")

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
    print(f"Model loaded. EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    results = {}

    # Test 1: 3-shot #### (current ablation format) with stopping criteria
    results['3-shot ####'] = test_format(model, tokenizer, "3-shot #### (with stop criteria)",
                                          format_3shot, samples, gts, batch_size=8)

    # Test 2: 8-shot #### with stopping criteria
    results['8-shot ####'] = test_format(model, tokenizer, "8-shot #### (with stop criteria)",
                                          format_8shot_hash, samples, gts, batch_size=4)

    # Test 3: 8-shot Wei et al. with stopping criteria
    results['8-shot Wei'] = test_format(model, tokenizer, "8-shot Wei et al. (with stop criteria)",
                                         format_8shot_wei, samples, gts, batch_size=4)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v*100:.1f}%")
    print(f"{'='*60}")
    print(f"\nNote: Meta's 84.5% GSM8K is for the Instruct model (confirmed by Meta on HF).")
    print(f"The base model's expected GSM8K range is ~50-60% (per third-party evaluations).")


if __name__ == "__main__":
    main()
