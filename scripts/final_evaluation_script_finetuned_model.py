"""
Simple evaluation for FINE-TUNED model on MedQA
Fine-tuned model outputs: "Ciprofloxacin" (just the answer text)
Reports ONLY ACCURACY (standard for MedQA benchmarks)
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from difflib import SequenceMatcher

# ============================================================
# Load Model
# ============================================================

def load_finetuned_model(model_id, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
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
# Match Prediction to Options
# ============================================================

def normalize(text):
    """Simple normalization: lowercase and strip"""
    return text.lower().strip()


def fuzzy_similarity(str1, str2):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, str1, str2).ratio()


def match_prediction_to_option(pred_text, options):
    """
    Match predicted text to one of the options using exact or fuzzy matching.
    Returns the option letter (A/B/C/D/E) or None
    """
    pred_norm = normalize(pred_text)
    
    # Try exact match first
    for opt in options:
        if pred_norm == normalize(opt['value']):
            return opt['key']
    
    # Fuzzy match with all options - find best match
    best_match_letter = None
    best_similarity = 0.0
    
    for opt in options:
        opt_text_norm = normalize(opt['value'])
        similarity = fuzzy_similarity(pred_norm, opt_text_norm)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_letter = opt['key']
    
    return best_match_letter


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
    LORA_PATH = "./qlora_medqa"
    MAX_SAMPLES = None  # None for full dataset
    
    print("Loading fine-tuned model...")
    model, tokenizer, device = load_finetuned_model(MODEL_ID, LORA_PATH)
    
    print("Loading dataset...")
    ds = load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
    eval_ds = ds["validation"]
    
    if MAX_SAMPLES:
        eval_ds = eval_ds.select(range(min(MAX_SAMPLES, len(eval_ds))))
    
    print(f"\nEvaluating FINE-TUNED MODEL on {len(eval_ds)} examples...\n")
    
    correct = 0
    exact_matches = 0
    fuzzy_matches = 0
    failed = 0
    
    for i, example in enumerate(tqdm(eval_ds, desc="Evaluating")):
        # Build prompt
        prompt = build_prompt(example["question"], example["options"])
        
        # Get prediction
        pred_text = generate_answer(model, tokenizer, device, prompt)
        
        # Get gold
        gold_letter = example["answer_idx"]
        gold_text = example["answer"]
        options = example["options"]
        
        # Match prediction to option letter
        pred_letter = match_prediction_to_option(pred_text, options)
        
        if pred_letter is None:
            failed += 1
        elif pred_letter == gold_letter:
            correct += 1
            # Track if it was exact or fuzzy match
            if normalize(pred_text) == normalize(gold_text):
                exact_matches += 1
            else:
                fuzzy_matches += 1
        
        # Show first 5 examples
        if i < 5:
            is_correct = pred_letter == gold_letter
            match_type = ""
            if is_correct:
                if normalize(pred_text) == normalize(gold_text):
                    match_type = "(exact match)"
                else:
                    match_type = "(fuzzy match)"
            
            print(f"\nExample {i+1}:")
            print(f"Question: {example['question'][:100]}...")
            print(f"Gold: {gold_letter}) {gold_text}")
            print(f"Pred: {pred_text}")
            print(f"Matched to: {pred_letter}")
            print(f"{'✓ CORRECT' if is_correct else '✗ WRONG'} {match_type}")
    
    # Calculate accuracy
    total = len(eval_ds)
    accuracy = (correct / total) * 100
    
    # Print results
    print("\n" + "="*60)
    print("FINE-TUNED MODEL RESULTS")
    print("="*60)
    print(f"Total Examples:     {total}")
    print(f"Correct:            {correct}")
    print(f"  - Exact matches:  {exact_matches}")
    print(f"  - Fuzzy matches:  {fuzzy_matches}")
    print(f"Wrong:              {total - correct - failed}")
    print(f"Failed Extractions: {failed}")
    print(f"\n✓ ACCURACY: {accuracy:.2f}%")
    print("="*60)
    
    return accuracy


if __name__ == "__main__":
    evaluate()