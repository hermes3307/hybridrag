#!/usr/bin/env python3
"""
Multi-Provider LLM Assistant Module for Image Processing System

This module provides LLM integration supporting multiple providers:
- Anthropic Claude
- OpenAI GPT
- Ollama (Local)
- Simple pattern matching fallback
"""

import os
import json
import re
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


def create_llm_assistant(provider: str = None, api_key: str = None, model: str = None):
    """Factory function to create appropriate LLM assistant based on provider

    Args:
        provider: LLM provider ('anthropic', 'openai', 'ollama', 'simple')
        api_key: API key for the provider (if needed)
        model: Specific model to use

    Returns:
        Appropriate LLM assistant instance
    """
    if provider is None:
        provider = os.getenv('LLM_PROVIDER', 'simple')

    if provider == 'anthropic':
        return AnthropicAssistant(api_key=api_key, model=model)
    elif provider == 'openai':
        return OpenAIAssistant(api_key=api_key, model=model)
    elif provider == 'ollama':
        return OllamaAssistant(model=model)
    else:
        return SimpleLLMAssistant()


class BaseLLMAssistant:
    """Base class for LLM assistants"""

    def __init__(self):
        self.conversation_history = []
        self.available_commands = {}
        self.system_status = {}

    def register_command(self, name: str, description: str, handler: Callable):
        """Register a command that the LLM can invoke"""
        self.available_commands[name] = {
            'description': description,
            'handler': handler
        }

    def update_system_status(self, status: Dict[str, Any]):
        """Update the system status information"""
        self.system_status = status

    def _build_system_prompt(self) -> str:
        """Build the system prompt with available commands and status"""
        commands_doc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.available_commands.items()
        ])

        status_doc = json.dumps(self.system_status, indent=2) if self.system_status else "No status available"

        return f"""You are an AI assistant integrated into an Image Processing System. You help users manage image downloads, embeddings, and searches using natural language.

Available Commands:
{commands_doc}

Current System Status:
{status_doc}

Your role:
1. Understand user requests in natural language
2. Execute appropriate commands to fulfill requests
3. Provide clear, concise responses
4. Handle errors gracefully and suggest solutions

When executing commands, respond with JSON in this format:
{{
    "action": "command_name",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "explanation": "Brief explanation of what you're doing"
}}

For informational queries that don't require commands, respond normally without JSON.

Guidelines:
- For download requests, parse the count and source from the user's message
- For embedding requests, determine if they want to process all images or just new ones
- For search requests, extract the query image path and search parameters
- For setup/configuration requests, guide the user through the necessary steps
- Always confirm actions before executing destructive operations
"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response for actions and parameters"""
        json_match = re.search(r'\{[\s\S]*\}', response)

        if json_match:
            try:
                action_data = json.loads(json_match.group())
                action = action_data.get('action')
                parameters = action_data.get('parameters', {})
                explanation = action_data.get('explanation', '')

                if action and action in self.available_commands:
                    return {
                        'response': response,
                        'action': action,
                        'parameters': parameters,
                        'explanation': explanation,
                        'error': False
                    }
            except json.JSONDecodeError:
                pass

        return {
            'response': response,
            'action': None,
            'parameters': None,
            'error': False
        }

    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command action"""
        if action not in self.available_commands:
            return {'success': False, 'error': f"Unknown action: {action}"}

        try:
            handler = self.available_commands[action]['handler']
            result = handler(**parameters)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': f"Error executing {action}: {str(e)}"}

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return self.conversation_history.copy()

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and return response - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement chat()")


class AnthropicAssistant(BaseLLMAssistant):
    """Anthropic Claude LLM Assistant"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model or os.getenv('LLM_MODEL', 'claude-sonnet-4-20250514')

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process message using Claude"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self._build_system_prompt(),
                messages=self.conversation_history
            )

            assistant_message = response.content[0].text

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return self._parse_response(assistant_message)

        except Exception as e:
            return {
                'response': f"Error communicating with Claude: {str(e)}",
                'action': None,
                'parameters': None,
                'error': True
            }


class OpenAIAssistant(BaseLLMAssistant):
    """OpenAI GPT LLM Assistant"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

        import openai
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model or os.getenv('LLM_MODEL', 'gpt-4o')

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process message using GPT"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            messages.extend(self.conversation_history)

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                messages=messages
            )

            assistant_message = response.choices[0].message.content

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return self._parse_response(assistant_message)

        except Exception as e:
            return {
                'response': f"Error communicating with GPT: {str(e)}",
                'action': None,
                'parameters': None,
                'error': True
            }


class OllamaAssistant(BaseLLMAssistant):
    """Ollama Local LLM Assistant"""

    def __init__(self, model: Optional[str] = None):
        super().__init__()
        self.model = model or os.getenv('LLM_MODEL', 'llama3.2')
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process message using Ollama"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            import requests

            # Build conversation context
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            messages.extend(self.conversation_history)

            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                }
            )

            if response.status_code != 200:
                raise Exception(f"Ollama returned status {response.status_code}: {response.text}")

            result = response.json()
            assistant_message = result.get('message', {}).get('content', '')

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return self._parse_response(assistant_message)

        except Exception as e:
            return {
                'response': f"Error communicating with Ollama: {str(e)}",
                'action': None,
                'parameters': None,
                'error': True
            }


class SimpleLLMAssistant(BaseLLMAssistant):
    """Fallback assistant without API integration for testing/demo"""

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process message with simple pattern matching"""
        msg_lower = user_message.lower()

        # Simple command detection patterns
        if 'download' in msg_lower:
            count_match = re.search(r'(\d+)', user_message)
            count = int(count_match.group(1)) if count_match else 10

            return {
                'response': f"I'll download {count} images for you.",
                'action': 'download_images',
                'parameters': {'count': count, 'source': 'picsum_landscape'},
                'explanation': f"Downloading {count} images from picsum_landscape",
                'error': False
            }

        elif 'embed' in msg_lower or 'process' in msg_lower:
            process_all = 'all' in msg_lower

            return {
                'response': "I'll process and embed the images.",
                'action': 'embed_images' if process_all else 'embed_new_images',
                'parameters': {},
                'explanation': "Processing images and generating embeddings",
                'error': False
            }

        elif 'search' in msg_lower:
            return {
                'response': "Please specify the search query or image path.",
                'action': None,
                'parameters': None,
                'error': False
            }

        elif 'status' in msg_lower:
            return {
                'response': "Let me get the system status for you.",
                'action': 'get_status',
                'parameters': {},
                'explanation': "Retrieving current system status",
                'error': False
            }

        else:
            commands = "\n".join([
                f"- {name}: {info['description']}"
                for name, info in self.available_commands.items()
            ])

            return {
                'response': f"I can help you with:\n\n{commands}\n\nTry saying:\n- 'Download 20 images'\n- 'Embed all images'\n- 'Show system status'",
                'action': None,
                'parameters': None,
                'error': False
            }


# Backward compatibility - keep original class names
LLMAssistant = AnthropicAssistant
