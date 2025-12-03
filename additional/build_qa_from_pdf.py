import os
import json
import gc
import argparse

import torch
from datasets import Dataset
from json_repair import repair_json

import pdfplumber

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    BitsAndBytesConfig,
    pipeline,
)

# ------------------------------
# Config
# ------------------------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# PDF Loading & Chunking
# ------------------------------

def load_pdf_text(pdf_path: str) -> str:
    """Read all pages of a PDF into a single text string."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def text_to_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # overlap

    return chunks


# ------------------------------
# Mistral Loading & Cleanup
# ------------------------------

def load_mistral_pipeline(
    model_id: str = MODEL_ID,
    use_4bit: bool = True,
) -> TextGenerationPipeline:
    """
    Load Mistral-7B-Instruct as a text-generation pipeline.
    Uses accelerate (device_map='auto'), so we MUST NOT pass `device=` to pipeline().
    """

    if use_4bit and DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",           # accelerate manages devices
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return gen_pipe


def cleanup_pipeline(gen_pipe: TextGenerationPipeline):
    """
    Free GPU/CPU memory used by the model.
    """
    try:
        model = gen_pipe.model
        tokenizer = gen_pipe.tokenizer

        del gen_pipe
        del model
        del tokenizer
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------
# Prompt / JSON Helpers
# ------------------------------
def build_qa_prompt(chunk: str, batch_size: int = 20) -> str:
    """
    Ask the model for a clean JSON array of QA pairs.
    Similar questions with different wording are allowed.
    """
    return f"""
You are a dataset generator AI.

Your task: create EXACTLY {batch_size} short-answer exam-style questions based STRICTLY on the text below.

Rules:
- Questions must be factual and based ONLY on information from the text.
- Answers must be SHORT (1 sentence, ideally 5–20 words).
- It is OK if some questions are similar in meaning as long as the wording is different.
- Avoid exact duplicate question wordings.
- Output MUST be a valid JSON array of objects with this exact schema:
  [
    {{ "question": "string", "answer": "string" }},
    ...
  ]

Very important:
- NO explanations, commentary, or prose outside the JSON.
- NO markdown, NO code fences, NO backticks.
- Do NOT include keys other than "question" and "answer".

Text:
\"\"\"{chunk}\"\"\"
"""
import json
from json_repair import repair_json  # <- new import


def extract_json_from_text(text: str):
    """
    Try to robustly extract a JSON list from model output.
    Uses json_repair to fix common syntax issues.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    # 1) Try direct JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2) Try repair on full text
    try:
        repaired = repair_json(text)
        data = json.loads(repaired)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 3) Try to extract the array region [ ... ] and repair only that
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        candidate = text[start:end]

        # 3a) Raw slice
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except Exception:
            pass

        # 3b) Repaired slice
        repaired = repair_json(candidate)
        data = json.loads(repaired)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # If all fails, give up
    return None

def shorten_answer(text: str, max_words: int = 20) -> str:
    """
    Force answers to be short (for safety).
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ------------------------------
# Q&A Generation (with up to 100 per chunk)
# ------------------------------
def generate_qa_for_chunk(
    gen_pipe,
    chunk: str,
    target_count: int = 100,
    batch_size: int = 20,
    max_new_tokens: int = 700,
) -> list[dict]:

    all_pairs = []
    seen_questions = set()
    attempts = 0

    while len(all_pairs) < target_count and attempts < 10:
        attempts += 1

        prompt = build_qa_prompt(chunk, batch_size=batch_size)

        outputs = gen_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=gen_pipe.tokenizer.eos_token_id,
            pad_token_id=gen_pipe.tokenizer.pad_token_id,
        )

        full_text = outputs[0]["generated_text"]
        generated = full_text[len(prompt):]

        json_data = extract_json_from_text(generated)

        if not isinstance(json_data, list):
            print(f"[WARN] Invalid JSON (even after repair) on attempt {attempts}, retrying...")
            # Optional: debug output to inspect what the model gave:
            # print("==== RAW OUTPUT START ====")
            # print(generated)
            # print("==== RAW OUTPUT END ====")
            continue

        new_count_before = len(all_pairs)

        for item in json_data:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a:
                continue

            q_lower = q.lower()
            if q_lower in seen_questions:
                continue  # skip exact duplicates

            seen_questions.add(q_lower)
            a_short = shorten_answer(a, max_words=20)

            all_pairs.append(
                {
                    "question": q,
                    "answer": a_short,
                    "source_chunk": chunk,
                }
            )

        print(f"[INFO] Chunk: got {len(all_pairs)}/{target_count} QAs so far "
              f"(+{len(all_pairs) - new_count_before} from this attempt)")

        if len(all_pairs) - new_count_before == 0:
            print("[WARN] No new unique QAs in this attempt, stopping early for this chunk.")
            break

    return all_pairs[:target_count]


# ------------------------------
# Main Pipeline (with JSONL streaming)
# ------------------------------

def build_qa_dataset_from_pdf(
    pdf_path: str,
    num_questions_per_chunk: int = 100,
    batch_size: int = 20,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    use_4bit: bool = True,
    output_jsonl: str | None = None,
) -> Dataset:
    """
    - Load PDF
    - Chunk text
    - Generate QAs for each chunk
    - Append each QA to JSONL (if specified)
    - Return a HF Dataset in memory
    - Cleanup model
    """
    print(f"[INFO] Loading PDF: {pdf_path}")
    text = load_pdf_text(pdf_path)
    print(f"[INFO] PDF chars: {len(text)}")

    print("[INFO] Chunking text...")
    chunks = text_to_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
    print(f"[INFO] Chunks: {len(chunks)}")

    print("[INFO] Loading teacher (Mistral) ...")
    gen_pipe = load_mistral_pipeline(MODEL_ID, use_4bit=use_4bit)

    all_examples = []

    # If we want streaming output, open file once
    jsonl_file = None
    if output_jsonl is not None:
        jsonl_file = open(output_jsonl, "w", encoding="utf-8")

    try:
        for idx, chunk in enumerate(chunks):
            print(f"[INFO] Generating Q&A for chunk {idx + 1}/{len(chunks)}")

            qa_pairs = generate_qa_for_chunk(
                gen_pipe,
                chunk,
                target_count=num_questions_per_chunk,
                batch_size=batch_size,
            )

            # Append to in-memory list
            all_examples.extend(qa_pairs)

            # Stream to JSONL as we go
            if jsonl_file is not None:
                for ex in qa_pairs:
                    jsonl_file.write(json.dumps(ex, ensure_ascii=False) + "\n")
                jsonl_file.flush()

        if not all_examples:
            raise RuntimeError("No Q&A pairs generated. Check prompt/model output.")

        dataset = Dataset.from_list(all_examples)
        print(f"[INFO] Built dataset with {len(dataset)} examples.")
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        print("[INFO] Cleaning up model and freeing GPU memory...")
        cleanup_pipeline(gen_pipe)

    return dataset


# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA dataset from PDF using Mistral-7B")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to input PDF")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Where to save dataset as JSONL (streaming)")
    parser.add_argument("--questions_per_chunk", type=int, default=100, help="Target QAs per chunk")
    parser.add_argument("--batch_size", type=int, default=20, help="QAs to ask for per generation call")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")

    args = parser.parse_args()

    ds = build_qa_dataset_from_pdf(
        pdf_path=args.pdf_path,
        num_questions_per_chunk=args.questions_per_chunk,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_4bit=not args.no_4bit,
        output_jsonl=args.output_jsonl,
    )

    if args.output_jsonl is None:
        print("[INFO] No output_jsonl specified; showing first 3 examples:")
        for i in range(min(3, len(ds))):
            print("----")
            print("Q:", ds[i]["question"])
            print("A:", ds[i]["answer"])
