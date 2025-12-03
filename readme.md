Federated QLoRA Fine-Tuning for Medical LLMs (Fed-MedQA)

This project implements a proof-of-concept for Federated Learning (FL) applied to the fine-tuning of Large Language Models (LLMs) for medical knowledge (MedQA). The system uses the Flower framework and LoRA for efficient, distributed training.

🌟 Core Concepts & Technology

Federated Learning (FL): Decentralized training where model updates are aggregated on a central server, ensuring local data privacy.

Model: We utilize the Qwen2.5-0.5B-Instruct model (transitioned from Mistral due to GPU limitations) as our base.

Efficiency: Training uses LoRA (Low-Rank Adaptation), meaning only lightweight adapter weights are transmitted and aggregated, not the full model.

Data Handling: Clients support heterogeneous data. Each client can use a different dataset or format, which includes:

Structured JSONL data (for instruction tuning).

Unstructured PDF/TXT files (for chunk-based Causal Language Modeling).

🚀 Running the System

The project can be run in two modes: a simple Command Line Interface (CLI) for distributed deployment, or a Streamlit UI for a centralized demo environment.

1. Command Line Interface (CLI)

The CLI is designed for a truly distributed setup where the server and clients can run on separate physical or virtual machines using remote IP addresses.

Core Principle: Flower is designed to operate on different machines. The client simply connects to the server's public IP and port.

Setup Commands (assuming your script is named fed_qlora_medqa.py):

A. Start the Server (The aggregator)

python fed_qlora_medqa.py --mode server --server_address 127.0.0.1:8080 --min_clients 2


B. Start the Clients (The trainers)

Run these commands on separate terminals (or separate remote machines), ensuring the --server_address matches the machine running the server.

# Client 0 (CID 0)
python fed_qlora_medqa.py --mode client --cid 0 --server_address 127.0.0.1:8080 --device cuda:0

# Client 1 (CID 1)
python fed_qlora_medqa.py --mode client --cid 1 --server_address 127.0.0.1:8080 --device cuda:1


2. Streamlit UI (Single-Machine Demo)

The Streamlit UI abstracts the command-line arguments and is the primary tool for demos. It uses Python's subprocess feature to start the server and clients on a single machine, allowing you to assign different GPUs or even CPU/GPU combinations to different clients to simulate limited resource access.

To start the UI:

streamlit run your_ui_script_name.py
# (Assuming your Streamlit file is the same file that contains the code provided)
# streamlit run fed_qlora_medqa.py


🎯 Purpose of the Streamlit UI

The Streamlit UI serves as a powerful demonstration and testing tool:

Demo Environment: Easily showcase the end-to-end workflow, including data upload (PDF/JSONL) and real-time training metrics.

Resource Simulation: Allows assigning specific devices (cuda:0, cuda:1, cpu) to individual clients, making it possible to demonstrate GPU limitations for different models (e.g., why Qwen was chosen over Mistral).

Use Case Validation: Provides a simple dashboard to help researchers and engineers check the initial accuracy and convergence behavior of federated learning for their specific dataset and LLM use case.