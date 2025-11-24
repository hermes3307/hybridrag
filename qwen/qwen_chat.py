#!/usr/bin/env python3
"""
Qwen3 Chatbot Application
This script allows running Qwen3 models locally with model selection at startup.
"""

import os
import sys
import argparse
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

class QwenChatbot:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", cache_dir: str = None):
        """
        Initialize the Qwen3 chatbot with specified model
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/transformers")

        print(f"Loading model: {model_name}...")
        print(f"Using cache directory: {self.cache_dir}")

        try:
            # Load tokenizer and model with explicit cache usage
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            # Load model with proper dtype handling and explicit cache
            # When using device_map with accelerate, avoid moving model afterward
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device if torch.cuda.is_available() else "cpu",
                cache_dir=cache_dir
            )

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load from cache only...")
            # If the first attempt fails (e.g., due to network issues), try with local files only
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device if torch.cuda.is_available() else "cpu",
                cache_dir=cache_dir,
                local_files_only=True
            )

            print("Model loaded from cache successfully!")

    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate response for the given prompt
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response to get only the generated part
        response = response[len(prompt):].strip()
        return response

    def chat(self):
        """
        Interactive chat interface
        """
        print(f"\nChatting with {self.model_name}")
        print("Type 'quit' or 'exit' to stop the conversation.")
        print("-" * 50)

        conversation_history = ""

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ["quit", "exit", "stop"]:
                print("Goodbye!")
                break

            # Format the prompt for Qwen models
            formatted_prompt = f"<|system|>You are a helpful assistant.<|user|>{user_input}<|assistant|>"

            # Generate response
            response = self.generate_response(formatted_prompt)

            print(f"Bot: {response}")

def list_available_models():
    """
    List available Qwen models that can be used
    """
    models = [
        # Qwen2 models (original)
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-72B-Instruct",
        # Qwen2.5 models (improved)
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        # Qwen3 models (latest, reasoning-focused)
        "Qwen/QwQ-32B-Preview",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        # VL (Vision-Language) models
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
    ]
    return models

def main():
    parser = argparse.ArgumentParser(description="Qwen3 Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (e.g., Qwen/Qwen2-7B-Instruct)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Qwen3 models"
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available Qwen3 models:")
        for model in list_available_models():
            print(f"  - {model}")
        return

    if args.model:
        model_name = args.model
    else:
        # Interactive model selection
        print("Available Qwen3 models:")
        models = list_available_models()
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")

        try:
            choice = input(f"\nSelect a model (1-{len(models)}): ")
            index = int(choice) - 1
            if 0 <= index < len(models):
                model_name = models[index]
            else:
                print("Invalid selection. Using default model Qwen/Qwen2-1.5B-Instruct")
                model_name = "Qwen/Qwen2-1.5B-Instruct"
        except (ValueError, IndexError):
            print("Invalid selection. Using default model Qwen/Qwen2-1.5B-Instruct")
            model_name = "Qwen/Qwen2-1.5B-Instruct"

    # Initialize and start chatbot
    try:
        chatbot = QwenChatbot(model_name, cache_dir=os.path.expanduser("~/.cache/huggingface/transformers"))
        chatbot.chat()
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure you have the required dependencies installed.")
        print("Install with: pip install torch transformers accelerate")

if __name__ == "__main__":
    main()