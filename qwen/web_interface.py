#!/usr/bin/env python3
"""
Web Interface for Qwen3 Chatbot
This script provides a web-based interface for the Qwen3 chatbot with separate pages for model selection and testing.
"""

import os
import json
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Configuration file path
CONFIG_FILE = "chatbot_config.json"

# Available models
AVAILABLE_MODELS = [
    # Qwen2 models (original)
    {"name": "Qwen/Qwen2-0.5B-Instruct", "size": "0.5B", "type": "Qwen2"},
    {"name": "Qwen/Qwen2-1.5B-Instruct", "size": "1.5B", "type": "Qwen2"},
    {"name": "Qwen/Qwen2-7B-Instruct", "size": "7B", "type": "Qwen2"},
    {"name": "Qwen/Qwen2-72B-Instruct", "size": "72B", "type": "Qwen2"},
    # Qwen2.5 models (improved)
    {"name": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B", "type": "Qwen2.5"},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B", "type": "Qwen2.5"},
    {"name": "Qwen/Qwen2.5-3B-Instruct", "size": "3B", "type": "Qwen2.5"},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "type": "Qwen2.5"},
    {"name": "Qwen/Qwen2.5-14B-Instruct", "size": "14B", "type": "Qwen2.5"},
    {"name": "Qwen/Qwen2.5-32B-Instruct", "size": "32B", "type": "Qwen2.5"},
    {"name": "Qwen/Qwen2.5-72B-Instruct", "size": "72B", "type": "Qwen2.5"},
    # Qwen3 models (latest, reasoning-focused)
    {"name": "Qwen/QwQ-32B-Preview", "size": "32B", "type": "Qwen3"},
    {"name": "Qwen/Qwen2.5-Coder-32B-Instruct", "size": "32B", "type": "Qwen2.5-Coder"},
    # VL (Vision-Language) models
    {"name": "Qwen/Qwen2-VL-7B-Instruct", "size": "7B", "type": "Qwen2-VL"},
    {"name": "Qwen/Qwen2-VL-72B-Instruct", "size": "72B", "type": "Qwen2-VL"},
]

# Default configuration
DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen2-1.5B-Instruct",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 512,
    "temperature": 0.7,
    "cache_dir": os.path.expanduser("~/.cache/huggingface/transformers")
}

class QwenWebChatbot:
    def __init__(self, config=None):
        if config is None:
            config = DEFAULT_CONFIG.copy()
        
        self.config = config
        self.model_name = config["model_name"]
        self.device = config["device"]
        self.max_length = config["max_length"]
        self.temperature = config["temperature"]
        self.cache_dir = config["cache_dir"]
        
        # Load the model and tokenizer
        self.load_model()

    def load_model(self):
        print(f"Loading model: {self.model_name}...")
        print(f"Using cache directory: {self.cache_dir}")
        
        try:
            # Load tokenizer and model with explicit cache usage
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # Load model with proper dtype handling and explicit cache
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device if torch.cuda.is_available() else "cpu",
                cache_dir=self.cache_dir
            )

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load from cache only...")
            # If the first attempt fails (e.g., due to network issues), try with local files only
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device if torch.cuda.is_available() else "cpu",
                cache_dir=self.cache_dir,
                local_files_only=True
            )

            print("Model loaded from cache successfully!")

    def generate_response(self, prompt: str) -> str:
        """
        Generate response for the given prompt
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response to get only the generated part
        response = response[len(prompt):].strip()
        return response

# Global chatbot instance
chatbot = None
chat_history = []

def load_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/api/models')
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    global chatbot
    if request.method == 'POST':
        config = request.json
        save_config(config)
        
        # Reload the chatbot with new configuration if model changed
        if chatbot is None or chatbot.model_name != config['model_name']:
            chatbot = QwenWebChatbot(config)
        
        return jsonify({"status": "success"})
    
    config = load_config()
    return jsonify(config)

@app.route('/api/chat', methods=['POST'])
def chat():
    global chatbot, chat_history
    data = request.json
    user_message = data.get('message', '')
    
    if chatbot is None:
        config = load_config()
        chatbot = QwenWebChatbot(config)
    
    # Format the prompt for Qwen models
    formatted_prompt = f"<|system|>You are a helpful assistant.<|user|>{user_message}<|assistant|>"
    
    # Generate response
    response = chatbot.generate_response(formatted_prompt)
    
    # Add to chat history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})
    
    return jsonify({
        "response": response,
        "history": chat_history
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(chat_history)

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"status": "success"})

@app.route('/api/test-model', methods=['POST'])
def test_model():
    """Test if the selected model works properly"""
    data = request.json
    test_prompt = data.get('prompt', 'Hello, how are you?')
    model_name = data.get('model_name', DEFAULT_CONFIG['model_name'])
    
    try:
        # Create a temporary instance with the specified model
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        
        # Load tokenizer and model with explicit cache usage
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Test with a simple prompt
        inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_response = response[len(test_prompt):].strip()
        
        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            "status": "success",
            "response": test_response,
            "message": f"Model {model_name} is working properly!"
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error testing model {model_name}: {str(e)}"
        })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

def create_templates_and_static():
    """Create necessary directories and files for the web interface"""
    # Create templates directory
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Create CSS file
    css_content = """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    header {
        background: #2c3e50;
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    h1 {
        margin: 0;
        font-size: 1.8em;
    }
    
    nav {
        background: #34495e;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    nav ul {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        gap: 15px;
    }
    
    nav li {
        display: inline;
    }
    
    nav a {
        color: white;
        text-decoration: none;
        padding: 8px 16px;
        border-radius: 5px;
        transition: background 0.3s;
    }
    
    nav a:hover {
        background: #2980b9;
    }
    
    nav a.active {
        background: #3498db;
    }
    
    .main-content {
        display: flex;
        gap: 20px;
        min-height: calc(100vh - 150px);
    }
    
    .config-panel {
        width: 300px;
        background: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        overflow-y: auto;
    }
    
    .chat-container {
        flex: 1;
        background: white;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .page {
        display: none;
        flex: 1;
    }
    
    .page.active {
        display: block;
    }
    
    .chat-history {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    
    .message {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 18px;
        line-height: 1.5;
    }
    
    .user-message {
        align-self: flex-end;
        background-color: #3498db;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message {
        align-self: flex-start;
        background-color: #ecf0f1;
        color: #2c3e50;
        border-bottom-left-radius: 4px;
    }
    
    .input-area {
        padding: 15px;
        border-top: 1px solid #eee;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    #message-input {
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 24px;
        font-size: 16px;
        resize: vertical;
        min-height: 90px;
        max-height: 180px;
        height: 90px;
        width: 100%;
    }
    
    #send-button {
        padding: 10px 20px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        font-size: 15px;
        align-self: flex-end;
        width: auto;
    }
    
    #send-button:hover {
        background: #2980b9;
    }
    
    .config-section {
        margin-bottom: 25px;
    }
    
    .config-section h3 {
        margin-top: 0;
        margin-bottom: 15px;
        color: #2c3e50;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
    }
    
    label {
        display: block;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    select, input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 14px;
    }
    
    button {
        background: #27ae60;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 6px;
        cursor: pointer;
        width: 100%;
        font-size: 16px;
    }
    
    button:hover {
        background: #219a52;
    }
    
    .test-btn {
        background: #2980b9;
    }
    
    .test-btn:hover {
        background: #21618c;
    }
    
    .clear-btn {
        background: #e74c3c;
        margin-top: 10px;
    }
    
    .clear-btn:hover {
        background: #c0392b;
    }
    
    .model-item {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 6px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .model-item:hover {
        background-color: #f1f8ff;
    }
    
    .model-item.selected {
        background-color: #d6eaf8;
        border-color: #3498db;
    }
    
    .model-name {
        font-weight: bold;
        color: #2c3e50;
    }
    
    .model-info {
        font-size: 0.9em;
        color: #7f8c8d;
    }
    
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background-color: #ecf0f1;
        border-radius: 18px;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
        width: auto;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #7f8c8d;
        border-radius: 50%;
        margin: 0 3px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    .result-box {
        background: #e8f4fd;
        border: 1px solid #bde0fe;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        min-height: 50px;
    }
    
    .success {
        color: #27ae60;
    }
    
    .error {
        color: #e74c3c;
    }
    
    .model-list-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 5px;
    }
    
    .model-list-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .model-list-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .model-list-container::-webkit-scrollbar-thumb {
        background: #bdc3c7;
        border-radius: 10px;
    }
    """
    
    with open('static/style.css', 'w') as f:
        f.write(css_content)
    
    # Create main index template (chat page)
    index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3 Chatbot - Chat</title>
    <link rel="stylesheet" href="{{ url_for('send_static', path='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Qwen3 Chatbot</h1>
        </header>
        
        <nav>
            <ul>
                <li><a href="/" class="active">Chat</a></li>
                <li><a href="/models">Model Selection</a></li>
                <li><a href="/test">Model Test</a></li>
            </ul>
        </nav>
        
        <div class="chat-container">
            <div class="chat-history" id="chat-history">
                <!-- Messages will be displayed here -->
                <div class="message assistant-message">
                    Hello! I'm your Qwen3 assistant. How can I help you today?
                </div>
            </div>
            
            <div class="input-area">
                <textarea id="message-input" placeholder="Type your message here..."></textarea>
                <button id="send-button">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        // DOM Elements
        const chatHistory = document.getElementById('chat-history');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        
        // Load chat history from API on page load
        async function loadHistory() {
            try {
                const response = await fetch('/api/history');
                const history = await response.json();
                
                chatHistory.innerHTML = '';
                if (history.length === 0) {
                    addMessage('assistant', 'Hello! I\'m your Qwen3 assistant. How can I help you today?');
                } else {
                    history.forEach(msg => {
                        addMessage(msg.role, msg.content);
                    });
                }
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
        
        // Add message to chat history
        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}-message`;
            messageDiv.textContent = content;
            chatHistory.appendChild(messageDiv);
            
            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
        
        // Show typing indicator
        function showTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'typing-indicator';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;
            chatHistory.appendChild(typingDiv);
            
            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
        
        // Hide typing indicator
        function hideTypingIndicator() {
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        // Send message to chatbot
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            messageInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                // Remove typing indicator
                hideTypingIndicator();
                
                // Add assistant response
                if (data.response) {
                    addMessage('assistant', data.response);
                }
            } catch (error) {
                // Remove typing indicator
                hideTypingIndicator();
                
                addMessage('assistant', 'Error: Could not get response from the chatbot.');
                console.error('Error sending message:', error);
            }
        }
        
        // Event Listeners
        sendButton.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', () => {
            loadHistory();
        });
    </script>
</body>
</html>"""
    
    with open('templates/index.html', 'w') as f:
        f.write(index_content)
    
    # Create model selection template
    models_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3 Chatbot - Model Selection</title>
    <link rel="stylesheet" href="{{ url_for('send_static', path='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Qwen3 Chatbot</h1>
        </header>
        
        <nav>
            <ul>
                <li><a href="/">Chat</a></li>
                <li><a href="/models" class="active">Model Selection</a></li>
                <li><a href="/test">Model Test</a></li>
            </ul>
        </nav>
        
        <div class="main-content">
            <div class="config-panel">
                <div class="config-section">
                    <h3>⚙️ Configuration</h3>
                    <label for="max-length">Max Response Length:</label>
                    <input type="number" id="max-length" value="512" min="100" max="2048">
                    
                    <label for="temperature">Temperature:</label>
                    <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
                    
                    <label for="cache-dir">Cache Directory:</label>
                    <input type="text" id="cache-dir" value="">
                    
                    <button id="save-config">Save Configuration</button>
                </div>
                
                <div class="config-section">
                    <button id="clear-history" class="clear-btn">Clear Chat History</button>
                </div>
            </div>
            
            <div class="chat-container">
                <div class="config-section">
                    <h3>🤖 Available Models</h3>
                    <div class="model-list-container" id="model-list">
                        <!-- Model list will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Current configuration
        let config = {};
        
        // DOM Elements
        const modelList = document.getElementById('model-list');
        const maxLengthInput = document.getElementById('max-length');
        const temperatureInput = document.getElementById('temperature');
        const cacheDirInput = document.getElementById('cache-dir');
        const saveConfigButton = document.getElementById('save-config');
        const clearHistoryButton = document.getElementById('clear-history');
        
        // Load models from API
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();
                
                modelList.innerHTML = '';
                models.forEach(model => {
                    const modelDiv = document.createElement('div');
                    modelDiv.className = 'model-item';
                    if (model.name === config.model_name) {
                        modelDiv.classList.add('selected');
                    }
                    
                    modelDiv.innerHTML = `
                        <div class="model-name">${model.name}</div>
                        <div class="model-info">Size: ${model.size} | Type: ${model.type}</div>
                    `;
                    
                    modelDiv.addEventListener('click', () => {
                        // Remove selection from other items
                        document.querySelectorAll('.model-item').forEach(item => {
                            item.classList.remove('selected');
                        });
                        
                        // Select this item
                        modelDiv.classList.add('selected');
                        
                        // Update config
                        config.model_name = model.name;
                    });
                    
                    modelList.appendChild(modelDiv);
                });
            } catch (error) {
                console.error('Error loading models:', error);
            }
        }
        
        // Load current configuration
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                config = await response.json();
                
                maxLengthInput.value = config.max_length || 512;
                temperatureInput.value = config.temperature || 0.7;
                cacheDirInput.value = config.cache_dir || '';
                
                // Reload models to show selection
                loadModels();
            } catch (error) {
                console.error('Error loading config:', error);
            }
        }
        
        // Save configuration
        async function saveConfig() {
            config.max_length = parseInt(maxLengthInput.value);
            config.temperature = parseFloat(temperatureInput.value);
            config.cache_dir = cacheDirInput.value;
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(config)
                });
                
                if (response.ok) {
                    alert('Configuration saved successfully!');
                }
            } catch (error) {
                console.error('Error saving config:', error);
                alert('Error saving configuration.');
            }
        }
        
        // Clear chat history
        async function clearHistory() {
            try {
                await fetch('/api/history', {
                    method: 'DELETE'
                });
                
                alert('Chat history cleared!');
            } catch (error) {
                console.error('Error clearing history:', error);
            }
        }
        
        // Event Listeners
        saveConfigButton.addEventListener('click', saveConfig);
        clearHistoryButton.addEventListener('click', clearHistory);
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', () => {
            loadConfig();
        });
    </script>
</body>
</html>"""
    
    with open('templates/models.html', 'w') as f:
        f.write(models_content)
    
    # Create model test template
    test_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3 Chatbot - Model Test</title>
    <link rel="stylesheet" href="{{ url_for('send_static', path='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Qwen3 Chatbot</h1>
        </header>
        
        <nav>
            <ul>
                <li><a href="/">Chat</a></li>
                <li><a href="/models">Model Selection</a></li>
                <li><a href="/test" class="active">Model Test</a></li>
            </ul>
        </nav>
        
        <div class="main-content">
            <div class="config-panel">
                <div class="config-section">
                    <h3>🤖 Select Model to Test</h3>
                    <select id="test-model-select">
                        <!-- Options will be populated by JavaScript -->
                    </select>
                </div>
                
                <div class="config-section">
                    <h3>💬 Test Prompt</h3>
                    <textarea id="test-prompt" rows="4" placeholder="Enter a test prompt...">Hello, can you introduce yourself?</textarea>
                    <button id="test-model-btn" class="test-btn">Test Model</button>
                </div>
            </div>
            
            <div class="chat-container">
                <div class="config-section">
                    <h3>📊 Test Results</h3>
                    <div id="test-result" class="result-box">
                        <p>Click "Test Model" to check if the selected model works properly.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // DOM Elements
        const testModelSelect = document.getElementById('test-model-select');
        const testPromptInput = document.getElementById('test-prompt');
        const testModelBtn = document.getElementById('test-model-btn');
        const testResult = document.getElementById('test-result');
        
        // Load models for the selection dropdown
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();
                
                testModelSelect.innerHTML = '';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = `${model.name} (${model.size})`;
                    if (model.name === 'Qwen/Qwen2-1.5B-Instruct') {
                        option.selected = true;
                    }
                    testModelSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading models:', error);
            }
        }
        
        // Test the selected model
        async function testModel() {
            const model_name = testModelSelect.value;
            const prompt = testPromptInput.value.trim();
            
            if (!prompt) {
                testResult.innerHTML = '<p class="error">Please enter a test prompt.</p>';
                return;
            }
            
            // Show loading state
            testResult.innerHTML = '<p>Testing model, please wait...</p>';
            testModelBtn.disabled = true;
            testModelBtn.textContent = 'Testing...';
            
            try {
                const response = await fetch('/api/test-model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        model_name: model_name,
                        prompt: prompt
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    testResult.innerHTML = `
                        <p class="success">${data.message}</p>
                        <p><strong>Response:</strong> ${data.response || 'No response generated'}</p>
                    `;
                } else {
                    testResult.innerHTML = `<p class="error">${data.message}</p>`;
                }
            } catch (error) {
                testResult.innerHTML = `<p class="error">Error testing model: ${error.message}</p>`;
                console.error('Error testing model:', error);
            } finally {
                testModelBtn.disabled = false;
                testModelBtn.textContent = 'Test Model';
            }
        }
        
        // Event Listeners
        testModelBtn.addEventListener('click', testModel);
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', () => {
            loadModels();
        });
    </script>
</body>
</html>"""
    
    with open('templates/test.html', 'w') as f:
        f.write(test_content)

def run_web_server():
    """Run the Flask web server"""
    create_templates_and_static()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    print("Starting Qwen3 Web Interface...")
    print("Visit http://localhost:5000 to access the interface")
    
    # Start the web server in a separate thread
    server_thread = threading.Thread(target=run_web_server)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down web server...")