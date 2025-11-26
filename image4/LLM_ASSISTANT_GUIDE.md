# LLM Assistant Guide

## Overview

The LLM Assistant tab provides a natural language interface to control the Image Processing System. You can use conversational commands to download images, process embeddings, search for images, and manage system settings.

## Features

### Two Operating Modes

1. **Full AI Mode** (with Anthropic API key)
   - Powered by Claude AI
   - Understands natural language commands
   - Provides intelligent responses and suggestions
   - Handles complex queries

2. **Simple Mode** (without API key)
   - Pattern-matching fallback
   - Handles basic commands
   - No API key required
   - Limited to predefined patterns

## Setup

### Getting an Anthropic API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key

### Configuration

1. Add your API key to the `.env` file:
   ```bash
   ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Restart the application to activate full AI mode

3. The LLM Assistant tab will show:
   - "Connected (Claude API)" - Full AI mode active
   - "Simple mode (no API key)" - Pattern-matching mode active

## Available Commands

The LLM Assistant can execute the following commands:

### 1. Download Images

**Command**: `download_images`

**Examples**:
- "Download 20 images"
- "Get 50 landscape images"
- "Download images from picsum_hd"

**Parameters**:
- `count`: Number of images to download (default: 10)
- `source`: Image source (default: "picsum_landscape")
  - picsum_general
  - picsum_landscape
  - picsum_square
  - picsum_portrait
  - picsum_hd

### 2. Embed All Images

**Command**: `embed_images`

**Examples**:
- "Process all images"
- "Generate embeddings for all images"
- "Embed everything"

**What it does**: Processes and generates vector embeddings for all images in the database

### 3. Embed New Images Only

**Command**: `embed_new_images`

**Examples**:
- "Process new images only"
- "Embed unprocessed images"
- "Generate embeddings for new images"

**What it does**: Processes only images that haven't been embedded yet

### 4. Search Images

**Command**: `search_images`

**Examples**:
- "Search for similar images"
- "Find images like /path/to/image.jpg"
- "Search for matching images"

**Parameters**:
- `query_image`: Path to the query image
- `limit`: Number of results (default: 10)

**Note**: Currently switches to the search tab for manual image selection

### 5. Get System Status

**Command**: `get_status`

**Examples**:
- "Show system status"
- "What's the current status?"
- "Give me system information"

**What it does**: Returns current database connection, image directory, model settings, and statistics

### 6. Setup Database

**Command**: `setup_database`

**Examples**:
- "Initialize the database"
- "Setup vector database"
- "Configure the database"

**What it does**: Initializes the PostgreSQL vector database with pgvector extension

## Using the LLM Assistant

### Chat Interface

1. **Navigate to LLM Assistant tab**
2. **Type your message** in the input box at the bottom
3. **Send message** by:
   - Clicking "Send Message" button
   - Pressing `Ctrl+Return`

### Chat Features

- **Color-coded messages**:
  - Blue: Your messages
  - Green: Assistant responses
  - Orange: System notifications
  - Red: Errors

- **Timestamps**: Each message includes time sent
- **Action execution**: Commands are automatically executed
- **Progress updates**: Real-time feedback on actions

### Example Conversations

#### Example 1: Download and Process Images

```
You: Download 30 landscape images
Assistant: I'll download 30 images for you.
System: Executing: Downloading 30 images from picsum_landscape
System: Action completed successfully: Started downloading 30 images from picsum_landscape

You: Now process all the new images
Assistant: I'll process and embed the images.
System: Executing: Processing images and generating embeddings
System: Action completed successfully: Started processing and embedding new images only
```

#### Example 2: Check System Status

```
You: What's the current system status?
Assistant: Let me check the system status for you.
System: Executing: Getting current system status
System: Action completed successfully:
Current System Status:
database_connected: True
images_directory: ./images
embedding_model: CLIP
download_source: picsum_landscape
total_images: 150
```

#### Example 3: Setup Workflow

```
You: I need to setup the database first
Assistant: I'll initialize the vector database for you.
System: Executing: Initializing vector database
System: Action completed successfully: Started database initialization
```

## Tips and Best Practices

### 1. Be Specific
- Specify exact numbers: "Download 25 images" instead of "Download some images"
- Include parameters when needed: "Download from picsum_hd source"

### 2. Check Status First
- Ask for status to understand current state
- Verify database connection before processing
- Check available images before searching

### 3. Sequential Operations
- Download images first
- Then process/embed them
- Finally search or analyze

### 4. Use Clear Language
- Full AI mode understands variations and context
- Simple mode works best with keyword matching
- Both modes support common command phrases

### 5. Monitor Progress
- Watch for system notifications
- Check error messages if something fails
- Switch to relevant tabs to see detailed progress

## Troubleshooting

### API Connection Issues

**Problem**: "Error communicating with LLM"

**Solutions**:
1. Check your API key is correctly set in `.env`
2. Verify internet connection
3. Check Anthropic API status
4. Ensure API key has sufficient credits

### Simple Mode Limitations

**Problem**: Commands not recognized in simple mode

**Solutions**:
1. Use exact command phrases from examples
2. Include keywords like "download", "embed", "search"
3. Consider adding API key for better understanding
4. Refer to example commands section

### Command Execution Failures

**Problem**: "Action failed" messages

**Solutions**:
1. Check system is properly initialized
2. Verify database connection
3. Ensure images directory exists
4. Check logs in other tabs for details

### Response Delays

**Problem**: Long wait times for responses

**Solutions**:
1. API calls may take a few seconds
2. Complex operations run in background
3. Check "Processing..." indicator
4. Wait for completion message

## Advanced Usage

### Custom Workflows

You can chain multiple commands in conversation:

```
You: I want to build an image database. First download 100 images,
     then process them all, and finally check the status.

Assistant: I'll help you build an image database. Let me start by
          downloading 100 images...
```

### Context Awareness

The assistant maintains conversation history:

```
You: Download 50 images
Assistant: I'll download 50 images for you.
[Action executes]

You: Now embed them
Assistant: I'll process and embed those images.
[Uses context to understand "them" refers to downloaded images]
```

### Error Recovery

If something fails, the assistant can suggest solutions:

```
You: Search for similar images
Assistant: I need a query image path. Could you provide the path
          to the image you want to search with?

You: /home/pi/images/sample.jpg
Assistant: I'll search for images similar to that one.
```

## Architecture

### Components

1. **llm_assistant.py**
   - `LLMAssistant`: Full Claude API integration
   - `SimpleLLMAssistant`: Pattern-matching fallback
   - Command registration system
   - Response parsing and execution

2. **image.py (LLM Tab)**
   - Chat interface UI
   - Message display with formatting
   - Command handler functions
   - System status integration

### Command Flow

1. User types message → Input captured
2. Message sent to assistant → LLM/pattern matching processes
3. Response generated → May include action
4. Action parsed → Parameters extracted
5. Command executed → Registered handler called
6. Result displayed → User sees outcome

### Integration Points

- **Download Tab**: Triggered via `llm_download_images()`
- **Process Tab**: Triggered via `llm_embed_*_images()`
- **Search Tab**: Triggered via `llm_search_images()`
- **Config Tab**: Triggered via `llm_setup_database()`
- **System Status**: Accessed via `get_current_system_status()`

## API Reference

### LLMAssistant Class

```python
class LLMAssistant:
    def __init__(self, api_key: Optional[str] = None)
    def register_command(self, name: str, description: str, handler: Callable)
    def update_system_status(self, status: Dict[str, Any])
    def chat(self, user_message: str) -> Dict[str, Any]
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]
    def clear_history(self)
    def get_history(self) -> List[Dict[str, str]]
```

### Response Format

```python
{
    'response': str,      # Assistant's text response
    'action': str,        # Command to execute (or None)
    'parameters': dict,   # Command parameters (or None)
    'explanation': str,   # Brief explanation of action
    'error': bool         # Whether an error occurred
}
```

## Future Enhancements

Potential improvements for the LLM Assistant:

1. **Direct Image Selection**
   - Browse and select images through chat
   - Upload images via chat interface

2. **Batch Operations**
   - Queue multiple operations
   - Schedule tasks

3. **Advanced Queries**
   - Natural language search queries
   - Complex filtering options

4. **Voice Integration**
   - Voice input support
   - Text-to-speech responses

5. **Multi-modal Understanding**
   - Analyze images in chat
   - Describe image contents

6. **Learning and Preferences**
   - Remember user preferences
   - Suggest workflows

## Support

For issues or questions:
- Check the main README_DOCUMENTATION.md
- Review IMPLEMENTATION_GUIDE.md
- Check system logs in Overview tab
- Verify .env configuration

## License

Same as the main Image Processing System