"""
Quick comparison of different prompt formats for Llama-3.1-8B on GSM8K.

Tests 3 approaches:
  A) gsm8k_cot format (Q:/A: with 8 CoT examples ending "The answer is X.")
  B) Llama chat template (manually applied) with gsm8k_cot_llama format
  C) Simple 3-shot Q:/A: (current ablation format)

Usage:
    python src/test_llama_baseline_v2.py
    python src/test_llama_baseline_v2.py 20
"""

import os
import sys
import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
GSM8K_FILE = "/data/zjj/Synergistic_Core/data/gsm8k/json/test.json"
NUM_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RANDOM_SEED = 42
MAX_NEW_TOKENS = 512

# ── 8-shot examples (from gsm8k-cot.yaml / gsm8k-cot-llama.yaml) ──
COT_EXAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]

# ── Llama chat template components ──
LLAMA_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

LLAMA_EXAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"),
]


def format_a_qa_cot8(question: str) -> str:
    """Format A: Standard gsm8k_cot Q:/A: with 8-shot examples."""
    prompt = ""
    for q, a in COT_EXAMPLES:
        prompt += f"Q: {q}\nA: {a}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def format_b_chat_template(question: str, tokenizer) -> str:
    """Format B: Llama chat template with gsm8k_cot_llama format."""
    messages = []
    for q, a in LLAMA_EXAMPLES:
        messages.append({"role": "user", "content": f'Given the following problem, reason and give a final answer to the problem.\nProblem: {q}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=LLAMA_TEMPLATE)


def format_c_qa_3shot(question: str) -> str:
    """Format C: Current ablation format (3-shot Q:/A: with ####)."""
    prefix = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
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
    return prefix + question + "\nA:"


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    # Truncate at stop sequences
    for stop in ["\nQ:", "</s>", "<|im_end|>", "<|eot_id|>"]:
        if stop in text:
            text = text[:text.index(stop)]

    # "The final answer is X"
    matches = list(re.finditer(r'[Tt]he final answer is\s*(-?[\$0-9.,]+)', text))
    if matches:
        return matches[-1].group(1).replace(',', '').replace('$', '').strip()

    # "The answer is X"
    matches = list(re.finditer(r'[Tt]he answer is\s*(-?[\$0-9.,]+)', text))
    if matches:
        return matches[-1].group(1).replace(',', '').replace('$', '').strip()

    # "#### X"
    if "####" in text:
        after = text.split("####")[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after.replace(',', ''))
        if numbers:
            return numbers[0]

    # Last number
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return ""


def extract_gt(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(',', '').replace(' ', '')
    numbers = re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))
    return numbers[-1] if numbers else ""


def test_format(model, tokenizer, format_name, format_fn, samples, gts):
    """Test one format and return accuracy."""
    print(f"\n{'='*60}")
    print(f"Testing: {format_name}")
    print(f"{'='*60}")

    prompts = [format_fn(s['question']) for s in samples]
    print(f"First prompt (last 200 chars): ...{prompts[0][-200:]}")

    correct = 0
    batch_size = 2 if 'chat' in format_name else 4  # Chat template prompts are long

    for i in range(0, len(samples), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_gt = gts[i:i + batch_size]

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
            pred = extract_answer(gen_text)
            gt = batch_gt[j]
            match = pred == gt
            if match:
                correct += 1
            if i + j < 2:
                print(f"  Q{i+j}: GT={gt}, Pred={pred}, {'CORRECT' if match else 'WRONG'}")
                print(f"    Gen (last 150): ...{gen_text[-150:]}")

        running_acc = correct / min(i + batch_size, len(samples))
        print(f"  Batch {i//batch_size+1}: Running acc = {running_acc:.3f} ({correct}/{min(i+batch_size,len(samples))})")

    accuracy = correct / len(samples)
    print(f"\n  >> {format_name} Accuracy: {accuracy*100:.1f}% ({correct}/{len(samples)})")
    return accuracy


def main():
    print("=" * 60)
    print("Llama-3.1-8B GSM8K Format Comparison")
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
    print("Model loaded")

    results = {}

    # Test A: gsm8k_cot (Q:/A: with 8-shot)
    results['A'] = test_format(model, tokenizer, "A: Q:/A: 8-shot CoT (gsm8k_cot)",
                                format_a_qa_cot8, samples, gts)

    # Test B: Llama chat template with gsm8k_cot_llama format
    format_b_fn = lambda q: format_b_chat_template(q, tokenizer)
    results['B'] = test_format(model, tokenizer, "B: Llama chat template 8-shot (gsm8k_cot_llama)",
                                format_b_fn, samples, gts)

    # Test C: Current ablation format (3-shot)
    results['C'] = test_format(model, tokenizer, "C: 3-shot Q:/A: #### (current ablation)",
                                format_c_qa_3shot, samples, gts)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
