"""
Simple evaluation for BASE model on MedQA
Base model outputs: "D) Ceftriaxone\n\nExplanation:..."
Reports ONLY ACCURACY (standard for MedQA benchmarks)
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# ============================================================
# Load Model
# ============================================================

def load_base_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def generate_answer(model, tokenizer, device, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


# ============================================================
# Extract Answer from Base Model
# ============================================================

def extract_base_prediction(pred_text):
    """
    Base model outputs: "D) Ceftriaxone\n\nExplanation:..."
    Extract the letter (D) from beginning
    """
    # Get first line
    first_line = pred_text.split('\n')[0].strip()
    
    # Extract letter at the beginning (A/B/C/D/E)
    match = re.match(r'^([A-E])', first_line, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


# ============================================================
# Build Prompt
# ============================================================

def build_prompt(question, options):
    opts_txt = "\n".join([f"{opt['key']}) {opt['value']}" for opt in options])
    return f"Question: {question}\n{opts_txt}\nAnswer:"


# ============================================================
# Main Evaluation
# ============================================================

def evaluate():
    # Config
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    MAX_SAMPLES = None  # None for full dataset
    
    print("Loading model...")
    model, tokenizer, device = load_base_model(MODEL_ID)
    
    print("Loading dataset...")
    ds = load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
    eval_ds = ds["validation"]
    
    if MAX_SAMPLES:
        eval_ds = eval_ds.select(range(min(MAX_SAMPLES, len(eval_ds))))
    
    print(f"\nEvaluating BASE MODEL on {len(eval_ds)} examples...\n")
    
    correct = 0
    failed = 0
    
    for i, example in enumerate(tqdm(eval_ds, desc="Evaluating")):
        # Build prompt
        prompt = build_prompt(example["question"], example["options"])
        
        # Get prediction
        pred_text = generate_answer(model, tokenizer, device, prompt)
        
        # Extract letter
        pred_letter = extract_base_prediction(pred_text)
        
        # Get gold
        gold_letter = example["answer_idx"]
        
        if pred_letter is None:
            failed += 1
        elif pred_letter == gold_letter:
            correct += 1
        
        # Show first 5 examples
        if i < 5:
            print(f"\nExample {i+1}:")
            print(f"Question: {example['question'][:100]}...")
            print(f"Gold: {gold_letter}) {example['answer']}")
            print(f"Pred: {pred_text[:100]}...")
            print(f"Extracted: {pred_letter}")
            print(f"{'✓ CORRECT' if pred_letter == gold_letter else '✗ WRONG'}")
    
    # Calculate accuracy
    total = len(eval_ds)
    accuracy = (correct / total) * 100
    
    # Print results
    print("\n" + "="*60)
    print("BASE MODEL RESULTS")
    print("="*60)
    print(f"Total Examples:     {total}")
    print(f"Correct:            {correct}")
    print(f"Wrong:              {total - correct - failed}")
    print(f"Failed Extractions: {failed}")
    print(f"\n✓ ACCURACY: {accuracy:.2f}%")
    print("="*60)
    
    return accuracy


if __name__ == "__main__":
    evaluate()