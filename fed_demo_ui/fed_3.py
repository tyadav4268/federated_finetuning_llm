import os
import random
import torch
import argparse
import glob
from copy import deepcopy
import numpy as np
import pypdf

import flwr as fl
from flwr.common import parameters_to_ndarrays # Added for Strategy
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ------------------------------
# Config Defaults
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_ADAPTER_DIR = os.path.join(BASE_DIR, "final_adapter")
CLIENT_ADAPTER_BASE_DIR = os.path.join(BASE_DIR, "client_adapters")
OUTPUT_DIR = "./qlora_output"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_LENGTH = 512
SEED = 42

# We remove the hardcoded global DEVICE default and use the argparse one
torch.manual_seed(SEED)
random.seed(SEED)

# Global tokenizer
tokenizer = None

# ------------------------------
# Helper: Data Loading (Kept from Script A)
# ------------------------------
def load_local_data(file_path):
    """Parses a PDF, Text, or JSONL file into a HuggingFace Dataset."""
    
    # CASE 1: JSONL (Structured QA with Context)
    if file_path.endswith('.jsonl'):
        try:
            print(f"Detected JSONL file: {file_path}")
            # Hugging Face datasets can load jsonl natively
            dataset = load_dataset("json", data_files=file_path, split="train")
            
            # If dataset is large, split it. If small, duplicate for train/test
            if len(dataset) > 1:
                dataset = dataset.train_test_split(test_size=0.1)
                return dataset["train"], dataset["test"]
            else:
                return dataset, dataset
        except Exception as e:
            print(f"Error reading JSONL: {e}")
            return None, None

    # CASE 2: PDF or TXT
    text_data = []
    try:
        if file_path.endswith('.pdf'):
            reader = pypdf.PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            chunk_size = 500
            chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
            text_data = [{"text": c} for c in chunks if len(c.strip()) > 50]
        else:
            # Assumes .txt or other text-based formats
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            text_data = [{"text": line.strip()} for line in lines if line.strip()]
            
        if not text_data:
            print(f"Warning: No text found in {file_path}")
            return None, None

        dataset = Dataset.from_list(text_data)
        
        if len(dataset) > 1:
            dataset = dataset.train_test_split(test_size=0.1)
            return dataset["train"], dataset["test"]
        else:
            return dataset, dataset 
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

# ------------------------------
# Helper: Formatting & Preprocessing (Standard CLM/SFT)
# ------------------------------
def format_prompt_standard(example):
    """
    Formats the JSONL entry into a single string for Standard SFT.
    Structure: ### Context -> ### Question -> ### Answer
    """
    context = example.get("source_chunk", "")
    question = example.get("question", "")
    answer = example.get("answer", "")
    
    # Simple hardcoded format (No Chat Templates)
    full_text = (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Answer:\n{answer}"
    )
    return {"text": full_text}

def preprocess_standard(batch, text_col="text"):
    """
    Standard SFT/CLM Preprocessing:
    - Tokenizes the full text.
    - COPIES input_ids to labels (No Masking).
    - The model trains on everything.
    """
    global tokenizer
    inputs = []
    for i in range(len(batch[text_col])):
        inputs.append(str(batch[text_col][i]))
             
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    
    # ---------------------------------------------------------
    # Standard SFT Logic (Script B Style):
    # We set labels = input_ids.
    # We do NOT use -100 to mask the question/context.
    # ---------------------------------------------------------
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# ------------------------------
# Model Loading
# ------------------------------
def load_and_init_model(model_name, device):
    global tokenizer
    # Ensure trust_remote_code=True for custom models/templates
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        trust_remote_code=True 
    ).to(device)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"], # Keep lightweight config
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    return model, tokenizer

def set_lora_parameters(model: PeftModel, ndarrays: list, device):
    """
    Updates the LoRA parameters of the model.
    """
    trainable_names = [n for n, v in model.named_parameters() if v.requires_grad]
    state_dict = model.state_dict()
    for name, arr in zip(trainable_names, ndarrays):
        state_dict[name] = torch.tensor(arr, dtype=torch.float32).to(device)
    model.load_state_dict(state_dict, strict=False)

# ------------------------------
# Flower Client
# ------------------------------
class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, tokenizer, train_dataset, val_dataset, device):
        self.cid = cid
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True)
        
        # --- LOGIC FOR JSONL DATA (Formatted SFT) ---
        if "source_chunk" in train_dataset.column_names and "question" in train_dataset.column_names:
            print(f"[Client {cid}] Formatting {len(train_dataset)} samples for Standard SFT (Context+Q+A)...")
            
            # 1. Format text into a single column "text"
            train_dataset = train_dataset.map(format_prompt_standard)
            
            # 2. Tokenize without masking
            self.train_tok = train_dataset.map(
                lambda x: preprocess_standard(x, "text"),
                batched=True,
                remove_columns=train_dataset.column_names
            )

            # 1. Format text into a single column "text"
            val_dataset = val_dataset.map(format_prompt_standard)
            
            # 2. Tokenize without masking
            self.val_tok = val_dataset.map(
                lambda x: preprocess_standard(x, "text"),
                batched=True,
                remove_columns=val_dataset.column_names
            )
            
        # --- LOGIC FOR RAW TEXT / PDF (Continued Pre-training) ---
        else:
            print(f"[Client {cid}] Formatting {len(train_dataset)} samples for Raw Text Training...")
            col_name = "text" if "text" in train_dataset.column_names else "question"
            
            self.train_tok = train_dataset.map(
                lambda x: preprocess_standard(x, col_name), 
                batched=True,
                remove_columns=train_dataset.column_names 
            )

            self.val_tok = val_dataset.map(
                lambda x: preprocess_standard(x, col_name), 
                batched=True,
                remove_columns=val_dataset.column_names 
            )

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.named_parameters() if val.requires_grad]

    def set_parameters(self, parameters):
        set_lora_parameters(self.model, parameters, self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"[Client {self.cid}] Training started on {self.device} with {len(self.train_dataset)} samples...")

        # Debug: compute a simple checksum of the first trainable tensor
        first_arr = parameters[0]
        checksum = float(np.mean(first_arr))
        print(f"[Client {self.cid}] Round {config.get('server_round')} "
              f"received weights checksum: {checksum:.6f}")
        
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}_c{self.cid}",
            per_device_train_batch_size=4,
            num_train_epochs=1,
            save_strategy="no",
            eval_strategy="steps",
            logging_steps=1,
            learning_rate=2e-4,
            use_cpu=(self.device == "cpu"), 
            remove_unused_columns=False,
            disable_tqdm=True, 
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tok,
            eval_dataset=self.val_tok,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
        metrics = trainer.evaluate()
        # print("Val Metrics: ", metrics)
        
        # Save adapter
        save_path = os.path.join(CLIENT_ADAPTER_BASE_DIR, f"client_{self.cid}")
        print(f"[Client {self.cid}] Saving local adapter to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        return self.get_parameters({}), len(self.train_tok), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

# ------------------------------
# 4. Server Strategy (Script B Style)
# ------------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, tokenizer, num_rounds, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = model
        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self.device = device

    def aggregate_fit(self, server_round, results, failures):
        # 1. Standard FedAvg Aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is None:
            return None

        parameters_aggregated, metrics_aggregated = aggregated
        
        if parameters_aggregated is None:
            return None

        # 2. Update Global Model
        # Convert parameters to list of numpy arrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        
        # Checksum for debugging
        first_arr = ndarrays[0]
        checksum = float(np.mean(first_arr))
        print(f"[Server] Round {server_round} aggregated weights checksum: {checksum:.6f}")

        # Load weights into the global model
        set_lora_parameters(self.global_model, ndarrays, self.device)

        # 3. Save Model (if last round)
        if server_round == self.num_rounds:
            print(f"[Server] Training complete. Saving final global model to {FINAL_ADAPTER_DIR}...")
            self.global_model.save_pretrained(FINAL_ADAPTER_DIR)
            self.tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

        return parameters_aggregated, metrics_aggregated

# ------------------------------
# Server Evaluation
# ------------------------------
def get_evaluate_fn(val_dataset, model_init: PeftModel, total_rounds, device):
    
    # Apply Standard SFT preprocessing to validation data too
    if "source_chunk" in val_dataset.column_names and "question" in val_dataset.column_names:
        print("[Server] Preprocessing validation data (Standard SFT)...")
        val_dataset = val_dataset.map(format_prompt_standard)
        val_tok = val_dataset.map(
            lambda x: preprocess_standard(x, "text"),
            batched=True,
            remove_columns=val_dataset.column_names
        )
    else:
        print("[Server] Preprocessing validation data (Raw Text)...")
        col_name = "text" if "text" in val_dataset.column_names else "question"
        val_tok = val_dataset.map(
            lambda x: preprocess_standard(x, col_name), 
            batched=True,
            remove_columns=val_dataset.column_names
        )

    val_tok = val_tok.select(range(min(10, len(val_tok))))

    def evaluate(server_round: int, parameters, config):
        # Note: Model is already updated in aggregate_fit strategy, 
        # but we re-set here to be safe/consistent with standard Flower flows
        
        set_lora_parameters(model_init, parameters, device)
        model_init.eval()

        total_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_tok)):
                inputs = torch.tensor(val_tok[i]["input_ids"]).unsqueeze(0).to(device)
                
                if "attention_mask" in val_tok[i]:
                     mask = torch.tensor(val_tok[i]["attention_mask"]).unsqueeze(0).to(device)
                else:
                     mask = None

                labels = torch.tensor(val_tok[i]["labels"]).unsqueeze(0).to(device)
                
                outputs = model_init(input_ids=inputs, attention_mask=mask, labels=labels)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(val_tok)
        print(f"[Server] Round {server_round} eval loss: {avg_loss:.4f} (on {device})")

        return float(avg_loss), {"loss": float(avg_loss)}

    return evaluate

def client_fn(cid: str, model_name, device, data_path=None):
    if data_path and os.path.exists(data_path):
        print(f"[Client {cid}] Loading local file: {data_path}")
        train_ds, val_ds = load_local_data(data_path)
    else:
        print(f"[Client {cid}] No file provided, using MedQA subset.")
        ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="train[:20]") 
        train_ds = ds

    if train_ds is None: 
        raise ValueError("Dataset could not be loaded.")

    model, tok = load_and_init_model(model_name, device)
    return LLMFlowerClient(cid, model, tok, train_ds, val_ds, device).to_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client"])
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_clients", type=int, default=2)
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use, e.g., 'cpu', 'cuda:0', 'cuda:1'")
    
    args = parser.parse_args()
    
    os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
    os.makedirs(CLIENT_ADAPTER_BASE_DIR, exist_ok=True)

    print(f"Initializing {args.mode} on device: {args.device}")

    if args.mode == "server":
        print(f"Starting Server (Expecting {args.min_clients} clients, {args.rounds} rounds)...")
        
        # if args.data_path:
        #      _, val_ds = load_local_data(args.data_path)
        # data_path = '/raid/home/tejpratapy/llm_training/final/temp_data/val.jsonl'
        # if os.path.exists(data_path):
        #          _, val_ds = load_local_data(data_path)
        # else:
        #      val_ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="validation[:20]")
        
        model, tok = load_and_init_model(args.model_name, args.device)
        
        # Use the Custom Strategy (Script B Style)
        strategy = SaveModelStrategy(
            model=model,
            tokenizer=tok,
            num_rounds=args.rounds,
            device=args.device,
            fraction_fit=1.0,
            min_fit_clients=args.min_clients,
            min_available_clients=args.min_clients,
            fraction_evaluate=0.0,
            min_evaluate_clients=0,
            # evaluate_fn=get_evaluate_fn(val_ds, model, args.rounds, args.device), 
            on_fit_config_fn=lambda r: {"server_round": r},
        )
        
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )

    elif args.mode == "client":
        fl.client.start_client(
            server_address=args.server_address,
            client=client_fn(str(args.cid), args.model_name, args.device, args.data_path),
        )