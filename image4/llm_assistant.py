#!/usr/bin/env python3
"""
LLM Assistant Module for Image Processing System

This module provides LLM integration to allow natural language commands
for downloading, embedding, and searching images.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import anthropic


class LLMAssistant:
    """LLM-powered assistant for image processing tasks"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Assistant

        Args:
            api_key: Anthropic API key (if None, reads from environment)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.conversation_history = []
        self.available_commands = {}
        self.system_status = {}

    def register_command(self, name: str, description: str, handler: Callable):
        """Register a command that the LLM can invoke

        Args:
            name: Command name
            description: What the command does
            handler: Function to execute the command
        """
        self.available_commands[name] = {
            'description': description,
            'handler': handler
        }

    def update_system_status(self, status: Dict[str, Any]):
        """Update the system status information

        Args:
            status: Dictionary containing current system state
        """
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

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and return response with potential actions

        Args:
            user_message: The user's message

        Returns:
            Dict containing:
                - response: The LLM's text response
                - action: Command to execute (if any)
                - parameters: Parameters for the command (if any)
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Call Claude API
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=self._build_system_prompt(),
                messages=self.conversation_history
            )

            # Extract response
            assistant_message = response.content[0].text

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Parse response for actions
            result = self._parse_response(assistant_message)

            return result

        except Exception as e:
            return {
                'response': f"Error communicating with LLM: {str(e)}",
                'action': None,
                'parameters': None,
                'error': True
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response for actions and parameters

        Args:
            response: The LLM's response text

        Returns:
            Parsed response dictionary
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)

        if json_match:
            try:
                action_data = json.loads(json_match.group())

                # Extract action and parameters
                action = action_data.get('action')
                parameters = action_data.get('parameters', {})
                explanation = action_data.get('explanation', '')

                # Validate action exists
                if action and action in self.available_commands:
                    return {
                        'response': response,
                        'action': action,
                        'parameters': parameters,
                        'explanation': explanation,
                        'error': False
                    }
                else:
                    # Action not found, treat as regular response
                    return {
                        'response': response,
                        'action': None,
                        'parameters': None,
                        'error': False
                    }

            except json.JSONDecodeError:
                # Not valid JSON, treat as regular response
                pass

        # No action found, return as regular response
        return {
            'response': response,
            'action': None,
            'parameters': None,
            'error': False
        }

    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command action

        Args:
            action: Command name
            parameters: Command parameters

        Returns:
            Result of the command execution
        """
        if action not in self.available_commands:
            return {
                'success': False,
                'error': f"Unknown action: {action}"
            }

        try:
            handler = self.available_commands[action]['handler']
            result = handler(**parameters)
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error executing {action}: {str(e)}"
            }

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history

        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()


class SimpleLLMAssistant:
    """Fallback assistant without API integration for testing/demo"""

    def __init__(self):
        """Initialize simple assistant"""
        self.available_commands = {}
        self.conversation_history = []

    def register_command(self, name: str, description: str, handler: Callable):
        """Register a command"""
        self.available_commands[name] = {
            'description': description,
            'handler': handler
        }

    def update_system_status(self, status: Dict[str, Any]):
        """Update system status (no-op for simple assistant)"""
        pass

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process message with simple pattern matching

        Args:
            user_message: User's message

        Returns:
            Response dictionary
        """
        msg_lower = user_message.lower()

        # Simple command detection patterns
        if 'download' in msg_lower:
            # Extract count if present
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

        else:
            # List available commands
            commands = "\n".join([
                f"- {name}: {info['description']}"
                for name, info in self.available_commands.items()
            ])

            return {
                'response': f"I can help you with:\n\n{commands}\n\nTry saying:\n- 'Download 20 images'\n- 'Embed all images'\n- 'Search for similar images'",
                'action': None,
                'parameters': None,
                'error': False
            }

    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command"""
        if action not in self.available_commands:
            return {'success': False, 'error': f"Unknown action: {action}"}

        try:
            handler = self.available_commands[action]['handler']
            result = handler(**parameters)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
