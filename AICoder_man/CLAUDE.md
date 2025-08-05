# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Coder is a manual-based professional AI code generation system that specializes in learning from technical documentation (PDFs, Word docs, HTML) and using that knowledge to generate accurate, compliant code. The system is particularly optimized for ALTIBASE database development but supports any manual-based development workflow.

## Key Components and Architecture

### Core System (`main.py`)
- **AICoderSystem**: Main orchestrator class that coordinates all components
- **ManualProcessor**: Handles parsing of PDF, DOCX, HTML documents into chunks
- **EnhancedVectorDBManager**: ChromaDB-based storage and retrieval for manual content
- **EnhancedClaudeClient**: Claude API integration with manual context injection
- **Data Classes**: ManualChunk, CodeGenerationResult, ValidationResult for structured data

### GUI Application (`gui.py`)
- **AICoderGUI**: Complete Tkinter-based interface with tabs for:
  - Manual Management: Upload and process documentation
  - Code Generation: Generate code with manual context
  - Code Validation: Validate against manual standards
  - Manual Search: Query the vector database
  - Settings: API keys and configuration
  - Logs: System activity monitoring

### Key Workflows
1. **Manual Upload**: Documents → Parsing → Chunking → Vector DB Storage
2. **Code Generation**: Task + Manual Search → RAG Context → Claude API → Code Output
3. **Validation**: Code + Manual Type → Compliance Checking → Suggestions

## Development Commands

Since this is a Python project with no build system, use these commands:

### Running the Application
```bash
# GUI mode (recommended)
python gui.py

# CLI/testing mode
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with:
```
CLAUDE_API_KEY=your_claude_api_key_here
GITHUB_API_KEY=your_github_api_key_here  # Optional
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

## Important Technical Details

### Manual Processing
- Supports PDF, DOCX, HTML, MD, TXT formats
- Uses pdfplumber as primary PDF parser with PyPDF2 fallback
- Intelligent chunking based on section markers and content structure
- Extracts keywords, code examples, and metadata from each chunk

### Vector Database
- ChromaDB persistent storage in `./manual_db/`
- Custom hash-based embedding function (768-dimensional)
- Metadata includes manual type, version, section, keywords, code examples
- Supports filtered search by manual type and version

### Claude API Integration
- Uses claude-3-5-sonnet-20241022 model
- Rate limiting: 2-second minimum interval between calls
- Test mode when API key not configured
- Manual context injection for accurate code generation

### Manual Types Supported
- `altibase`: ALTIBASE database-specific manuals
- `database`: General database documentation
- `api_reference`: API documentation
- `administration`: System administration guides
- `custom`: User-defined manual types

## Configuration Files

### Generated Files (Auto-created)
- `ai_coder_config.json`: GUI settings and API keys
- `code_generation_history.json`: History of generated code
- `manual_db/`: ChromaDB vector database directory
- `manuals/`: Manual file storage directory
- `generated_code/`: Output directory for saved code
- `logs/`: System log files

## Key Features for Code Generation

### Manual-Based Context
The system searches the vector database for relevant manual content and injects it into Claude prompts with instructions to:
- Follow exact syntax from manuals
- Use specific API patterns documented
- Respect version compatibility
- Include manual references in generated code

### Code Validation
Basic validation includes:
- Language-specific checks (SQL, Python, etc.)
- Manual compliance scoring
- API compatibility assessment
- Best practice suggestions based on manual content

## Error Handling and Logging

The system includes comprehensive logging at INFO level with GUI display. Common issues:
- PDF parsing failures trigger PyPDF2 fallback
- Claude API errors are handled gracefully with test mode
- Vector DB initialization creates collections automatically
- File upload supports batch processing with error reporting

## Development Notes

- The system is designed to work offline for manual processing
- Only code generation requires internet (Claude API)
- Vector database is fully local (ChromaDB)
- GUI uses threading for non-blocking operations
- All manual content stays local for privacy

## Testing

The system includes a built-in test mode when Claude API is not configured, returning template code for development and testing purposes.