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

    # Use higher-level models for better conversation quality
    # You can change these to any of the available models:
    # - Qwen/QwQ-32B-Preview (Qwen3 reasoning model)
    # - Qwen/Qwen2.5-32B-Instruct
    # - Qwen/Qwen2.5-14B-Instruct
    # - Qwen/Qwen2.5-7B-Instruct

    model1_name = "Qwen/Qwen2.5-7B-Instruct"  # Upgraded from 1.5B to 7B
    model2_name = "Qwen/Qwen2.5-7B-Instruct"  # Upgraded from 1.5B to 7B

    # Participant 1
    try:
        tokenizer1 = AutoTokenizer.from_pretrained(model1_name, cache_dir=cache_dir)
        model1 = AutoModelForCausalLM.from_pretrained(
            model1_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir
        )
    except:
        print("Network error detected. Attempting to load from cache only...")
        tokenizer1 = AutoTokenizer.from_pretrained(model1_name, cache_dir=cache_dir, local_files_only=True)
        model1 = AutoModelForCausalLM.from_pretrained(
            model1_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir,
            local_files_only=True
        )
    
    if device == "cpu":
        model1 = model1.to(device)

    # Participant 2
    try:
        tokenizer2 = AutoTokenizer.from_pretrained(model2_name, cache_dir=cache_dir)
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            cache_dir=cache_dir
        )
    except:
        print("Network error detected. Attempting to load from cache only...")
        tokenizer2 = AutoTokenizer.from_pretrained(model2_name, cache_dir=cache_dir, local_files_only=True)
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_name,
            torch_dtype=dtype,
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