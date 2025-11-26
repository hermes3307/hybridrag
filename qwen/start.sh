#!/bin/bash

# Qwen3 Chatbot Start Script
# This script allows running multiple Qwen3 chatbots that can talk to each other

# Set the virtual environment path
VENV_PATH="/home/pi/qwen/venv"
PYTHON_PATH="$VENV_PATH/bin/python"

echo "Qwen3 Chatbot Launcher"
echo "======================"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first:"
    echo "python3 -m venv $VENV_PATH"
    echo "source $VENV_PATH/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Function to display available models
list_models() {
    echo "Available Qwen models:"
    echo "Qwen2 models (original):"
    echo "  1) Qwen/Qwen2-0.5B-Instruct"
    echo "  2) Qwen/Qwen2-1.5B-Instruct"
    echo "  3) Qwen/Qwen2-7B-Instruct"
    echo "  4) Qwen/Qwen2-72B-Instruct"
    echo ""
    echo "Qwen2.5 models (improved):"
    echo "  5) Qwen/Qwen2.5-0.5B-Instruct"
    echo "  6) Qwen/Qwen2.5-1.5B-Instruct"
    echo "  7) Qwen/Qwen2.5-3B-Instruct"
    echo "  8) Qwen/Qwen2.5-7B-Instruct"
    echo "  9) Qwen/Qwen2.5-14B-Instruct"
    echo " 10) Qwen/Qwen2.5-32B-Instruct"
    echo " 11) Qwen/Qwen2.5-72B-Instruct"
    echo ""
    echo "Qwen3 models (latest, reasoning-focused):"
    echo " 12) Qwen/QwQ-32B-Preview"
    echo " 13) Qwen/Qwen2.5-Coder-32B-Instruct"
    echo ""
    echo "Vision-Language models:"
    echo " 14) Qwen/Qwen2-VL-7B-Instruct"
    echo " 15) Qwen/Qwen2-VL-72B-Instruct"
}

# Function to start a single chatbot
start_single_bot() {
    echo "Starting single chatbot..."
    $PYTHON_PATH qwen_chat.py
}

# Function to start multiple chatbots for conversation
start_multi_bots() {
    echo "Setting up multiple chatbots to talk to each other..."

    # Check if we have multiple GPUs or if we should use CPU
    if nvidia-smi > /dev/null 2>&1; then
        gpu_count=$(nvidia-smi -L | wc -l)
        echo "Found $gpu_count GPU(s)"

        if [ $gpu_count -ge 2 ]; then
            echo "Running chatbots on separate GPUs..."
            echo "Starting Chatbot 1 (GPU 0)..."
            CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH qwen_chat.py --model "Qwen/Qwen2-1.5B-Instruct" > chatbot1.log 2>&1 &
            sleep 5
            echo "Starting Chatbot 2 (GPU 1)..."
            CUDA_VISIBLE_DEVICES=1 $PYTHON_PATH qwen_chat.py --model "Qwen/Qwen2-1.5B-Instruct" > chatbot2.log 2>&1 &
        else
            echo "Only one GPU found. Running chatbots on same GPU..."
            echo "Starting Chatbot 1..."
            $PYTHON_PATH qwen_chat.py --model "Qwen/Qwen2-1.5B-Instruct" > chatbot1.log 2>&1 &
            sleep 5
            echo "Starting Chatbot 2..."
            $PYTHON_PATH qwen_chat.py --model "Qwen/Qwen2-1.5B-Instruct" > chatbot2.log 2>&1 &
        fi
    else
        echo "No GPUs found. Running chatbots on CPU..."
        echo "Starting Chatbot 1..."
        $PYTHON_PATH qwen_chat.py --model "Qwen/Qwen2-1.5B-Instruct" > chatbot1.log 2>&1 &
        sleep 5
        echo "Starting Chatbot 2..."
        $PYTHON_PATH qwen_chat.py --model "Qwen/Qwen2-1.5B-Instruct" > chatbot2.log 2>&1 &
    fi

    echo "Chatbots started in background. Logs are being saved to chatbot1.log and chatbot2.log"
    echo "Press Ctrl+C to stop both chatbots"

    # Wait for both processes
    wait
}

# Function to start a conversation simulation
start_conversation() {
    echo "Starting conversation between two chatbots..."
    echo "This feature requires implementing a conversation manager."
    echo "For now, you can manually start two instances with '$PYTHON_PATH qwen_chat.py' in separate terminals."

    # Create a simple conversation simulator
    cat << 'EOF' > conversation_simulator.py
#!/usr/bin/env python3
"""
Simple conversation simulator between two Qwen models
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_conversation():
    # Load two instances of the same or different models
    print("Loading conversation participants...")

    # Use cache directory to reuse downloaded models
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    print(f"Using cache directory: {cache_dir}")

    # Determine the device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Participant 1
    try:
        tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", cache_dir=cache_dir)
        model1 = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            torch_dtype=dtype,  # Use dtype instead of torch_dtype
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir
        )
    except:
        print("Network error detected. Attempting to load from cache only...")
        tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", cache_dir=cache_dir, local_files_only=True)
        model1 = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            torch_dtype=dtype,  # Use dtype instead of torch_dtype
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir,
            local_files_only=True
        )
    
    if device == "cpu":
        model1 = model1.to(device)

    # Participant 2
    try:
        tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", cache_dir=cache_dir)
        model2 = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            torch_dtype=dtype,  # Use dtype instead of torch_dtype
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir
        )
    except:
        print("Network error detected. Attempting to load from cache only...")
        tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", cache_dir=cache_dir, local_files_only=True)
        model2 = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            torch_dtype=dtype,  # Use dtype instead of torch_dtype
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir,
            local_files_only=True
        )
    
    if device == "cpu":
        model2 = model2.to(device)

    print("Starting conversation simulation...")
    print("Type a message to start the conversation. Type 'quit' to exit.\n")

    initial_prompt = input("Starting message: ")
    if initial_prompt.lower() in ['quit', 'exit']:
        return

    # Start conversation
    current_message = initial_prompt
    turn = 1

    for i in range(10):  # Limit to 10 exchanges
        print(f"\n--- Turn {turn} ---")

        if turn % 2 == 1:
            print(f"Bot 1: ", end="")
            participant_tokenizer = tokenizer1
            participant_model = model1
            bot_name = "Bot 1"
        else:
            print(f"Bot 2: ", end="")
            participant_tokenizer = tokenizer2
            participant_model = model2
            bot_name = "Bot 2"

        # Format the prompt for Qwen models
        formatted_prompt = f"<|system|>You are participating in a conversation.<|user|>{current_message}<|assistant|>"
        inputs = participant_tokenizer.encode(formatted_prompt, return_tensors="pt").to(participant_model.device)

        with torch.no_grad():
            outputs = participant_model.generate(
                inputs,
                max_length=len(inputs[0]) + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=participant_tokenizer.eos_token_id,
                eos_token_id=participant_tokenizer.eos_token_id
            )

        response = participant_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):].strip()

        print(response)

        current_message = response
        turn += 1

if __name__ == "__main__":
    run_conversation()
EOF

    $PYTHON_PATH conversation_simulator.py
}

# Function to start web interface
start_web_interface() {
    echo "Starting web interface..."
    echo "Access the interface at: http://localhost:5000"
    python3 web_interface.py
}

# Main menu
while true; do
    echo
    echo "Choose an option:"
    echo "1) List available models"
    echo "2) Start single chatbot"
    echo "3) Start multiple chatbots (for conversation simulation)"
    echo "4) Start conversation simulator"
    echo "5) Start web interface"
    echo "6) Quit"
    echo
    read -p "Enter your choice (1-6): " choice

    case $choice in
        1)
            list_models
            ;;
        2)
            start_single_bot
            ;;
        3)
            start_multi_bots
            ;;
        4)
            start_conversation
            ;;
        5)
            start_web_interface
            ;;
        6)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter 1-6."
            ;;
    esac
done