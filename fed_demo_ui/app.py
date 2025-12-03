import streamlit as st
import torch
import subprocess
import sys
import time
import re
import pandas as pd
import plotly.express as px
import os
import shutil
from threading import Thread
from queue import Queue, Empty
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# -----------------------------------------------------------------------------
# Configuration & State Management
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fed-MedQA Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this matches the name of your training script
SCRIPT_NAME = os.path.join(BASE_DIR, "fed_3.py")
PYTHON_EXE = sys.executable
TEMP_DATA_DIR = os.path.join(BASE_DIR, "temp_data")

# Paths for Adapters
FINAL_ADAPTER_DIR = os.path.join(BASE_DIR, "final_adapter")
CLIENT_ADAPTER_BASE_DIR = os.path.join(BASE_DIR, "client_adapters")

# Initialize Session State
if "server_process" not in st.session_state: st.session_state.server_process = None
if "client_processes" not in st.session_state: st.session_state.client_processes = {} 
if "client_files" not in st.session_state: st.session_state.client_files = {} 
if "log_queue" not in st.session_state: st.session_state.log_queue = Queue()
if "loss_history" not in st.session_state: st.session_state.loss_history = []
if "client_metrics" not in st.session_state: st.session_state.client_metrics = []
if "server_active" not in st.session_state: st.session_state.server_active = False
if "training_complete" not in st.session_state: st.session_state.training_complete = False
if "current_round" not in st.session_state: st.session_state.current_round = 0
if "server_logs" not in st.session_state: st.session_state.server_logs = ["Ready to start."]
if "client_logs" not in st.session_state: st.session_state.client_logs = []
if "messages" not in st.session_state: st.session_state.messages = [] 

os.makedirs(TEMP_DATA_DIR, exist_ok=True)
os.makedirs(CLIENT_ADAPTER_BASE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def enqueue_output(out, queue, prefix):
    for line in iter(out.readline, b''):
        queue.put((prefix, line.decode("utf-8")))
    out.close()

def get_isolated_env(device_str):
    """
    Creates an environment dictionary that isolates the specific GPU.
    Input: "cuda:4" -> Output Env: CUDA_VISIBLE_DEVICES="4"
    Input: "cpu"    -> Output Env: CUDA_VISIBLE_DEVICES=""
    """
    env = os.environ.copy()
    
    device_str = device_str.lower().strip()
    
    if "cpu" in device_str:
        # Hide all GPUs from this process
        env["CUDA_VISIBLE_DEVICES"] = ""
        script_device_arg = "cpu"
    elif "cuda" in device_str:
        # Extract the GPU ID (e.g., from 'cuda:4' get '4')
        if ":" in device_str:
            gpu_id = device_str.split(":")[-1]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        else:
            # Default to 0 if no specific ID given but cuda requested
            env["CUDA_VISIBLE_DEVICES"] = "0"
            
        # IMPORTANT: Inside the isolated process, the GPU is always 'cuda:0' 
        # because it only sees the one we exposed.
        script_device_arg = "cuda"
    else:
        # Fallback
        script_device_arg = "cpu"
        
    return env, script_device_arg

def start_server_process(num_rounds, min_clients, model_id, device_input):
    if st.session_state.server_active: return

    # Prepare isolated environment
    env, script_device = get_isolated_env(device_input)

    st.session_state.loss_history = []
    st.session_state.current_round = 0
    st.session_state.server_logs = [f"Server starting on {device_input} (Internal: {script_device})..."]
    
    cmd = [PYTHON_EXE, "-u", SCRIPT_NAME, 
           "--mode", "server", 
           "--rounds", str(num_rounds),
           "--min_clients", str(min_clients),
           "--model_name", model_id,
           "--device", script_device] # Pass generic 'cuda' or 'cpu'
    
    try:
        # Pass the 'env' to Popen
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                bufsize=1, close_fds=sys.platform != 'win32', env=env)
        st.session_state.server_process = proc
        st.session_state.server_active = True
        
        t = Thread(target=enqueue_output, args=(proc.stdout, st.session_state.log_queue, "SERVER"))
        t.daemon = True
        t.start()
    except Exception as e:
        st.error(f"Server start failed: {e}")

def start_client_process(cid, model_id, device_input):
    file_path = st.session_state.client_files.get(cid)
    
    # Prepare isolated environment
    env, script_device = get_isolated_env(device_input)
    
    cmd = [PYTHON_EXE, "-u", SCRIPT_NAME, 
           "--mode", "client", 
           "--cid", str(cid), 
           "--model_name", model_id,
           "--device", script_device] # Pass generic 'cuda' or 'cpu'
    
    if file_path:
        cmd.extend(["--data_path", file_path])
        
    try:
        # Pass the 'env' to Popen
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                bufsize=1, close_fds=sys.platform != 'win32', env=env)
        st.session_state.client_processes[cid] = proc
        
        t = Thread(target=enqueue_output, args=(proc.stdout, st.session_state.log_queue, f"CLIENT_{cid}"))
        t.daemon = True
        t.start()
    except Exception as e:
        st.error(f"Client {cid} failed: {e}")

def stop_all():
    if st.session_state.server_process:
        st.session_state.server_process.terminate()
        st.session_state.server_process = None
    
    for cid, proc in st.session_state.client_processes.items():
        if proc: proc.terminate()
    
    st.session_state.client_processes = {}
    st.session_state.server_active = False
    st.session_state.training_complete = False

# -----------------------------------------------------------------------------
# ADAPTER LOADING & INFERENCE (MODIFIED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_global_only_model(global_adapter_path, base_model_id):
    """
    Loads the Base Model + Global Adapter. 
    Does NOT load client adapters to keep things clean for global vs base comparison.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left' 
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Load only the global adapter
        model = PeftModel.from_pretrained(base_model, global_adapter_path, adapter_name="global")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
    
def generate_chat_response(prompt, model, tokenizer, adapter_name="global"):
    """
    Generates response. 
    If adapter_name is 'base', it disables adapters to run on the pre-trained model.
    """
    system_instruction = "You are a helpful AI assistant. Answer the user's question clearly."
    formatted_prompt = f"### Instruction:\n{system_instruction}\n\n### Input:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Wrapper to handle Base vs Fine-tuned logic
    def run_gen():
        return model.generate(
            **inputs,
            max_new_tokens=200,    
            do_sample=False    
            # do_sample=True,             
            # temperature=0.7,           
            # top_p=0.9,                  
            # top_k=50,
            # repetition_penalty=1.2,     
            # pad_token_id=tokenizer.pad_token_id,
            # eos_token_id=tokenizer.eos_token_id
        )

    with torch.no_grad():
        if adapter_name == "base":
            # Context manager to disable LoRA/Adapters and use base weights
            with model.disable_adapter():
                outputs = run_gen()
        else:
            # Ensure the specific adapter is active
            model.set_adapter(adapter_name)
            outputs = run_gen()
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_output:
        response = full_output.split("### Response:")[-1].strip()
    else:
        response = full_output.replace(formatted_prompt, "").strip()
    return response

# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------
st.title("Federated Learning Control Center")

with st.sidebar:
    st.header("⚙️ Configuration")
    conf_model = st.text_input("Base Model ID", "Qwen/Qwen2.5-0.5B-Instruct")
    conf_rounds = st.number_input("Rounds", min_value=1, value=3)
    conf_clients = st.slider("Number of Clients", min_value=1, max_value=5, value=2)
    st.divider()
    if st.button("🛑 RESET SYSTEM", type="primary"):
        stop_all()
        if os.path.exists(TEMP_DATA_DIR): shutil.rmtree(TEMP_DATA_DIR)
        os.makedirs(TEMP_DATA_DIR)
        st.rerun()

col_server, col_viz = st.columns([1, 2])

with col_server:
    st.subheader("1. Server Orchestration")
    
    # User types 'cpu' or 'cuda:0', etc.
    server_device_input = st.text_input("Server Device", value="cpu", help="e.g., 'cpu' to save GPU mem", key="srv_dev")
    
    if not st.session_state.server_active:
        if st.button("🚀 Start Server"):
            start_server_process(conf_rounds, conf_clients, conf_model, server_device_input)
            st.rerun()
    else:
        st.success(f"Server Running")

    st.subheader("2. Client Management")
    for i in range(conf_clients):
        with st.container(border=True):
            c_col1, c_col2 = st.columns([2, 1])
            with c_col1:
                st.markdown(f"**Client {i}**")
                uploaded_file = st.file_uploader(f"Data for Client {i}", type=['pdf', 'txt', 'jsonl'], key=f"u_{i}")
                
                # Default suggestion logic
                default_dev = "cpu"
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    # Suggest a specific GPU index
                    default_dev = f"cuda:{i % count}"
                
                c_device_input = st.text_input(f"Device", value=default_dev, key=f"dev_{i}", label_visibility="collapsed", placeholder="e.g. cuda:4")
                
                if uploaded_file:
                    path = os.path.join(TEMP_DATA_DIR, f"client_{i}_{uploaded_file.name}")
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.client_files[i] = path
                    st.caption(f"Loaded file")

            with c_col2:
                is_running = i in st.session_state.client_processes
                btn_state = st.button(f"Start Client {i}", key=f"btn_{i}", disabled=not st.session_state.server_active or is_running)
                
                if btn_state:
                    start_client_process(i, conf_model, c_device_input)
                    st.rerun()
                
                if is_running:
                    st.info(f"Running on {c_device_input}...")

with col_viz:
    st.subheader("3. Real-time Training Metrics")
    chart_ph = st.empty()
    if st.session_state.client_metrics:
        df = pd.DataFrame(st.session_state.client_metrics)
    
        fig = px.line(
            df, 
            x="Step", 
            y="Value", 
            color="Client", 
            line_dash="Metric", 
            title="Client Loss Progression (Train vs Eval)",
            labels={"Value": "Loss", "Step": "Log Event"},
            category_orders={"Metric": ["Train Loss", "Eval Loss"]} # Force legend order
        )
        chart_ph.plotly_chart(fig, use_container_width=True)
    else:
        chart_ph.info("Waiting for client training logs...")

    st.divider()
    st.subheader("Logs")
    log_c1, log_c2 = st.columns(2)
    with log_c1: 
        st.caption("Server Logs")
        server_log_content = '\n'.join(st.session_state.server_logs)
        st.download_button("⬇️ Save Server Logs", server_log_content, "server_logs.txt", key="dl_srv")
        st.code('\n'.join(st.session_state.server_logs[-15:]))
    with log_c2:
        st.caption("Client Logs")
        client_log_content = '\n'.join(st.session_state.client_logs)
        st.download_button("⬇️ Save Client Logs", client_log_content, "client_logs.txt", key="dl_cli")
        st.code('\n'.join(st.session_state.client_logs[-15:]))

try:
    while True:
        prefix, line = st.session_state.log_queue.get_nowait()
        clean_line = line.strip()
        
        if prefix == "SERVER":
            st.session_state.server_logs.append(clean_line)
            if f"Run finished" in clean_line:
                st.session_state.training_complete = True    
        else:
            # IT IS A CLIENT LOG
            st.session_state.client_logs.append(f"[{prefix}] {clean_line}")
            
            # --- MODIFIED PARSING LOGIC START ---
            metric_data = None
            
            # 1. Check for Evaluation Loss
            if "'eval_loss':" in clean_line:
                match = re.search(r"'eval_loss':\s*([\d\.]+)", clean_line)
                if match:
                    metric_data = {
                        "Value": float(match.group(1)),
                        "Metric": "Eval Loss"
                    }

            # 2. Check for Training Loss (only if not eval, to prevent overlap)
            elif "'loss':" in clean_line:
                match = re.search(r"'loss':\s*([\d\.]+)", clean_line)
                if match:
                    metric_data = {
                        "Value": float(match.group(1)),
                        "Metric": "Train Loss"
                    }

            # If we found a metric, append it
            if metric_data:
                try:
                    # Calculate a simple sequential step for this client
                    client_data = [x for x in st.session_state.client_metrics if x['Client'] == prefix]
                    step_count = len(client_data) + 1
                    
                    st.session_state.client_metrics.append({
                        "Client": prefix,
                        "Value": metric_data["Value"],
                        "Metric": metric_data["Metric"],
                        "Step": step_count
                    })
                except ValueError:
                    pass
except Empty:
    pass

st.markdown("---")
st.subheader("🤖 Comparative Model Analysis: Base vs. Federated")
if st.session_state.training_complete:
    if os.path.exists(FINAL_ADAPTER_DIR):
        # Changed: Load only global adapter
        model, tokenizer = load_global_only_model(FINAL_ADAPTER_DIR, conf_model)
        
        if model:
            st.success(f"Global Model Loaded Successfully.")
            if prompt := st.chat_input("Ask a question to compare models..."):
                st.session_state.messages.append({"role": "user", "content": prompt, "type": "user"})

            for msg in st.session_state.messages:
                if msg["type"] == "user":
                    with st.chat_message("user"): st.write(msg["content"])
                elif msg["type"] == "comparison":
                    with st.container(border=True):
                        # Changed: Only two columns now
                        cols = st.columns(2)
                        
                        # Display Base Model
                        with cols[0]:
                            st.markdown("### Base Model (Pre-trained)")
                            st.caption("Without Federated Fine-tuning")
                            st.info(msg["responses"].get("Base Model", "Error"))

                        # Display Global Model
                        with cols[1]:
                            st.markdown("### Global Model (Federated)")
                            st.caption("Aggregated knowledge from all clients")
                            st.success(msg["responses"].get("Global Model", "Error"))

            if prompt:
                responses = {}
                with st.spinner("Generating Comparative Analysis..."):
                    # 1. Generate response from the Base Model (disable adapters)
                    responses["Base Model"] = generate_chat_response(prompt, model, tokenizer, adapter_name="base")
                    
                    # 2. Generate response from the Global Federated Model
                    responses["Global Model"] = generate_chat_response(prompt, model, tokenizer, adapter_name="global")
                
                st.session_state.messages.append({"role": "assistant", "type": "comparison", "responses": responses})
                st.rerun()
    else:
        st.warning(f"Training finished, but global model not found at {FINAL_ADAPTER_DIR}.")
else:
    st.info("Complete training rounds to unlock the comparative chatbot.")

if st.session_state.server_active:
    time.sleep(1)
    st.rerun()