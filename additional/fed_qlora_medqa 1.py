import os
import random
import torch
import argparse
from copy import deepcopy
import numpy as np

import flwr as fl
from flwr.common import parameters_to_ndarrays, Parameters
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader

# ------------------------------
# Config
# ------------------------------
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_ID = "sshleifer/tiny-gpt2"
# MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_ID = "bigbio/med_qa"
DATASET_CONFIG = "med_qa_en_source"
OUTPUT_DIR = "./qlora_medqa"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 4
MAX_LENGTH = 512
SEED = 42
NUM_CLIENTS = 2
SERVER_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
random.seed(SEED)

# Global tokenizer for preprocess
tokenizer = None


# ------------------------------
# Helper: Format prompt and Preprocess
# ------------------------------
def format_prompt(example):
    question = example["question"]
    options = example.get("options", [])
    opts_txt = "\n".join([f"{opt['key']}) {opt['value']}" for opt in options])
    answer = example["answer"]
    return f"Question: {question}\n{opts_txt}\nAnswer: {answer}"


def preprocess(batch):
    global tokenizer
    inputs = []
    for i in range(len(batch["question"])):
        example = {k: batch[k][i] for k in batch.keys()}
        inputs.append(format_prompt(example))
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ------------------------------
# 1. Non-IID Data Partitioning for 2 Clients
# ------------------------------
def get_client_datasets(
    ds: DatasetDict,
    num_clients: int,
) -> tuple[list[Dataset], Dataset]:
    subset_size = "auto"
    full_train = ds["train"].shuffle(seed=SEED)
    total_size = len(full_train)

    # --- Determine subset size ---
    if subset_size == "full":
        use_size = total_size
    elif subset_size == "half":
        use_size = total_size // 2
    elif subset_size == "auto": 
        use_size = min(total_size, 5000)
    elif isinstance(subset_size, int):
        use_size = min(subset_size, total_size)
    else:
        raise ValueError("subset_size must be 'full', 'half', 'auto', or an integer")

    # --- Select subset ---
    train_subset = full_train.select(range(use_size))

    # --- Partition among clients ---
    split_size = use_size // num_clients
    client_train_datasets = []

    for cid in range(num_clients):
        start = cid * split_size
        end = (cid + 1) * split_size if cid < num_clients - 1 else use_size
        client_train_datasets.append(train_subset.select(range(start, end)))

    # Validation stays the same
    global_val_dataset = ds["validation"]

    print(f"[Data] Total_train={total_size} | subset={use_size} | "
          f"clients={num_clients} | samples_per_client≈{split_size}")

    return client_train_datasets, global_val_dataset



# ------------------------------
# 2. Model Loading (QLoRA)
# ------------------------------
def load_and_init_model(device):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model on a specific device
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if "cuda" in device and torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    return model, tokenizer, lora_config


# ------------------------------
# Shared helper for setting model parameters from a list of ndarrays
# ------------------------------
def set_lora_parameters_from_list(model: PeftModel, ndarrays: list):
    trainable_param_names = [name for name, val in model.named_parameters() if val.requires_grad]
    assert len(trainable_param_names) == len(
        ndarrays
    ), f"Mismatch: {len(trainable_param_names)} trainable params vs {len(ndarrays)} arrays"

    state_dict = model.state_dict()
    for name, arr in zip(trainable_param_names, ndarrays):
        state_dict[name] = torch.tensor(arr, dtype=torch.float16)
    model.load_state_dict(state_dict, strict=False)


# ------------------------------
# 3. The Flower Client
# ------------------------------
class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, tokenizer, train_dataset):
        self.cid = cid
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True)

        # Tokenize client's data once
        self.train_tok = self.train_dataset.map(
            preprocess,
            batched=True,
            remove_columns=self.train_dataset.column_names,
        )

    # NumPyClient: return list[np.ndarray]
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for name, val in self.model.named_parameters() if val.requires_grad]

    # NumPyClient: parameters is list[np.ndarray]
    def set_parameters(self, parameters):
        set_lora_parameters_from_list(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Debug: compute a simple checksum of the first trainable tensor
        first_arr = parameters[0]
        checksum = float(np.mean(first_arr))
        print(f"[Client {self.cid}] Round {config.get('server_round')} "
              f"received weights checksum: {checksum:.6f}")
        
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}_client{self.cid}",
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=1,
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            # bf16=False,
            gradient_accumulation_steps=8,
            remove_unused_columns=False,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tok,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
        return self.get_parameters({}), len(self.train_tok), {}

    def evaluate(self, parameters, config):
        # We do server-side evaluation, so keep this trivial
        return 0.0, 0, {}


# ------------------------------
# 4. Server Logic and Execution
# ------------------------------

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, tokenizer, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This is a PeftModel with LoRA already wrapped
        self.global_model = model
        self.tokenizer = tokenizer
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        # Let FedAvg do the usual aggregation first
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            print(f"[Server] Round {server_round}: no results to aggregate.")
            return None

        parameters_aggregated, metrics_aggregated = aggregated
        if parameters_aggregated is None:
            print(f"[Server] Round {server_round}: aggregation returned no parameters "
                  f"(probably all clients failed).")
            return None

        
        # Convert Flower Parameters -> list[np.ndarray]
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        # ---- checksum logging here ----
        first_arr = ndarrays[0]
        checksum = float(np.mean(first_arr))
        print(f"[Server] Round {server_round} aggregated weights checksum: {checksum:.6f}")
        # Load them into the global LoRA model
        set_lora_parameters_from_list(self.global_model, ndarrays)

        # If this is the last round, save the model + tokenizer
        if server_round == self.num_rounds:
            save_dir = os.path.join(OUTPUT_DIR, "global_model")
            os.makedirs(save_dir, exist_ok=True)
            # Save only the adapter weights (PeftModel)
            self.global_model.save_pretrained(save_dir)
            # Save tokenizer too, so you can reload easily
            self.tokenizer.save_pretrained(save_dir)
            print(f"[Server] Saved final global model to {save_dir}")

        return parameters_aggregated, metrics_aggregated

MAX_EVAL_SAMPLES = 10
def get_evaluate_fn(val_dataset, model_init: PeftModel, tokenizer):
    max_eval = min(MAX_EVAL_SAMPLES, len(val_dataset))
    eval_raw = val_dataset.shuffle(seed=SEED).select(range(max_eval))
    print(f"[Server] Evaluation will use {max_eval} examples")

    eval_tok = eval_raw.map(
        preprocess,
        batched=True,
        remove_columns=eval_raw.column_names,
    )

    def evaluate(server_round: int, parameters, config):
        first_arr = parameters[0]
        checksum = float(np.mean(first_arr))
        print(f"[Server] Round {server_round} aggregated weights checksum: {checksum:.6f}")
        
        # Fresh copy of model_init, already on DEVICE
        temp_model: PeftModel = deepcopy(model_init)
        set_lora_parameters_from_list(temp_model, parameters)
        temp_model.eval()

        total_loss = 0.0
        total_examples = len(eval_tok)

        with torch.no_grad():
            for i in range(total_examples):
                ex = eval_tok[i]

                input_ids = torch.tensor(ex["input_ids"], dtype=torch.long, device=DEVICE).unsqueeze(0)
                attention_mask = torch.tensor(ex["attention_mask"], dtype=torch.long, device=DEVICE).unsqueeze(0)
                labels = torch.tensor(ex["labels"], dtype=torch.long, device=DEVICE).unsqueeze(0)

                outputs = temp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

        avg_loss = total_loss / max(total_examples, 1)
        print(f"[Server] Round {server_round} eval loss: {avg_loss:.4f}")

        return float(avg_loss), {"loss": float(avg_loss)}

    return evaluate


def client_fn(cid: str, model_init, tokenizer_init, client_datasets_init, device):
    client_id = int(cid)
    # Reload model fresh on the assigned device
    new_model, _, _ = load_and_init_model(device)

    return LLMFlowerClient(
        cid=client_id,
        model=new_model,
        tokenizer=tokenizer_init,
        train_dataset=client_datasets_init[int(cid)],
    ).to_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated LoRA Fine-Tuning")
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client"])
    parser.add_argument("--server_address", type=str, default=SERVER_ADDRESS)
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")     # <--- new
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    print("Loading dataset...")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, trust_remote_code=True)
    client_datasets, global_val_ds = get_client_datasets(ds, num_clients=NUM_CLIENTS)

    print("Loading model & tokenizer...")
    initial_model, tokenizer, lora_config_global = load_and_init_model(args.device)

    if args.mode == "server":
        print("Starting Flower Server...")

        strategy = SaveModelStrategy(
            model=initial_model,
            tokenizer=tokenizer,
            num_rounds=NUM_ROUNDS,
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=0.0,
            min_evaluate_clients=0,
            on_fit_config_fn=lambda r: {"server_round": r},
        )

        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )

    elif args.mode == "client":
        print(f"Starting Flower Client {args.cid} on {args.device}...")
        fl.client.start_client(
            server_address=args.server_address,
            client=client_fn(str(args.cid), initial_model, tokenizer, client_datasets, args.device),
        )
