#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vector Query Assistant (vectorqry.py)
-------------------------------------
Document-based query answering system using vector search and LLM.
- Uses vector search to find relevant document chunks
- Leverages Anthropic's Claude API to generate answers based on retrieved documents
- Double-checks answers through original document references for accuracy
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import traceback
from datetime import datetime
from collections import deque

# Vector store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Anthropic API
from anthropic import Anthropic

# Rich UI for terminal
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint
from rich.layout import Layout
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_query.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Vector_Query")

# Initialize console
console = Console()

class ProgressTracker:
    """Track progress with ETA calculation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.file_times = deque(maxlen=50)
        self.current_file_start = None
        self.total_files = 0
        self.completed_files = 0
        self.current_phase = ""
        
    def start_tracking(self, total_files: int, phase: str = ""):
        """Start tracking progress"""
        self.total_files = total_files
        self.completed_files = 0
        self.start_time = time.time()
        self.current_phase = phase
        self.file_times.clear()
        
    def start_file(self):
        """Mark the start of file processing"""
        self.current_file_start = time.time()
    
    def complete_file(self, file_name: str = None):
        """Mark file as completed and update statistics"""
        if self.current_file_start:
            processing_time = time.time() - self.current_file_start
            self.file_times.append(processing_time)
            self.current_file_start = None
        
        self.completed_files += 1
        return self.get_progress_info(file_name)
    
    def get_progress_info(self, current_file: str = None) -> Dict[str, Any]:
        """Get current progress information"""
        if self.total_files == 0:
            return {}
        
        percentage = (self.completed_files / self.total_files) * 100
        elapsed_time = time.time() - self.start_time
        
        # Calculate ETA
        eta_seconds = None
        avg_time_per_file = None
        
        if self.file_times and self.completed_files > 0:
            avg_time_per_file = sum(self.file_times) / len(self.file_times)
            remaining_files = self.total_files - self.completed_files
            if remaining_files > 0:
                eta_seconds = avg_time_per_file * remaining_files
        
        # Format times
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta_seconds) if eta_seconds else "calculating..."
        avg_time_str = self._format_time(avg_time_per_file) if avg_time_per_file else "calculating..."
        
        return {
            'current_file': current_file,
            'completed': self.completed_files,
            'total': self.total_files,
            'percentage': percentage,
            'elapsed_time': elapsed_str,
            'eta': eta_str,
            'avg_time_per_file': avg_time_str,
            'remaining': self.total_files - self.completed_files,
            'phase': self.current_phase
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds is None:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def log_progress(self, logger, current_file: str = None, prefix: str = ""):
        """Log progress information"""
        info = self.get_progress_info(current_file)
        if not info:
            return
        
        progress_msg = (
            f"{prefix}[{info['completed']}/{info['total']}] "
            f"{info['percentage']:.1f}% | "
            f"Elapsed: {info['elapsed_time']} | "
            f"ETA: {info['eta']} | "
            f"Avg: {info['avg_time_per_file']}/file"
        )
        
        if current_file:
            progress_msg += f" | Current: {Path(current_file).name}"
        
        logger.info(progress_msg)

class Config:
    """Configuration for the vector query system."""
    
    def __init__(self):
        # Paths
        self.vector_db_path = os.path.join(os.getcwd(), "vector")
        self.documents_path = os.path.join(os.getcwd(), "batch")
        
        # Anthropic API settings
        self.anthropic_api_key = ""
        self.model = "claude-3-opus-20240229"  # Latest model
        
        # Query settings
        self.max_chunks = 5  # Maximum chunks to retrieve
        self.similarity_threshold = 0.7  # Similarity threshold (0-1)
        
        # Embedding model
        self.embedding_model = "all-MiniLM-L6-v2"
        
        # Language preference
        self.response_language = "en"  # 'ko' for Korean, 'en' for English
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.query_history = []
        
        # Advanced
        self.temperature = 0.2
        self.max_tokens = 2000
    
    def save(self, filename="vectorqry_config.json"):
        """Save configuration to a file."""
        # Don't save API key and history in the file
        save_data = {k: v for k, v in self.__dict__.items() 
                    if k not in ['anthropic_api_key', 'query_history']}
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Configuration saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load(self, filename="vectorqry_config.json"):
        """Load configuration from a file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                logger.info(f"Configuration loaded from {filename}")
                return True
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                return False
        else:
            logger.warning(f"Configuration file {filename} not found")
            return False

class VectorQueryAssistant:
    """Vector-based query assistant using LLM."""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        
        # Try to load configuration
        self.config.load()
        
        # Initialize paths
        self.vector_db_path = Path(self.config.vector_db_path)
        self.documents_path = Path(self.config.documents_path)
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.anthropic_client = None
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize vector store and LLM."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            logger.info(f"Initialized embeddings with model: {self.config.embedding_model}")
            
            # Load vector store if exists
            vector_path = str(self.vector_db_path)
            if self.vector_db_path.exists():
                index_file = self.vector_db_path / "index.faiss"
                if index_file.exists():
                    logger.info(f"Loading vector store from {vector_path}")
                    try:
                        # Add allow_dangerous_deserialization=True to fix the pickle security warning
                        self.vector_store = FAISS.load_local(
                            vector_path, 
                            self.embeddings,
                            allow_dangerous_deserialization=True  # This is the key fix
                        )
                        logger.info("Vector store loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load vector store: {e}")
                        logger.error(traceback.format_exc())
                        
                        # Try alternative loading method if the first one fails
                        try:
                            logger.info("Attempting alternative loading method...")
                            import faiss
                            import pickle
                            
                            # Manually load the index and docstore
                            index = faiss.read_index(str(index_file))
                            with open(str(self.vector_db_path / "docstore.pkl"), "rb") as f:
                                docstore = pickle.load(f)
                                
                            # Create a new FAISS instance
                            self.vector_store = FAISS(self.embeddings.embed_query, index, docstore, {})
                            logger.info("Vector store loaded successfully using alternative method")
                        except Exception as alt_e:
                            logger.error(f"Alternative loading also failed: {alt_e}")
                            self.vector_store = None
                else:
                    logger.warning(f"Vector index file not found at {index_file}")
                    self.vector_store = None
            else:
                logger.warning(f"Vector store directory not found at {vector_path}")
                self.vector_store = None
            
            # Initialize Anthropic client if API key is available
            if self.config.anthropic_api_key:
                try:
                    self.anthropic_client = Anthropic(api_key=self.config.anthropic_api_key)
                    logger.info("Anthropic client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Anthropic client: {e}")
                    logger.error(traceback.format_exc())
                    self.anthropic_client = None
            else:
                logger.warning("No Anthropic API key provided")
                self.anthropic_client = None
        
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.error(traceback.format_exc())
            self.embeddings = None
            self.vector_store = None
            self.anthropic_client = None

    def search_vector_store(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search the vector store for relevant chunks."""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        
        if k is None:
            k = self.config.max_chunks
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by threshold if configured
            if self.config.similarity_threshold > 0:
                # Convert score to similarity (1 - distance)
                results = [(doc, score) for doc, score in results 
                          if (1 - score) >= self.config.similarity_threshold]
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_original_content(self, doc: Document) -> Tuple[str, Path]:
        """Get original content and file path from document metadata."""
        source_path = doc.metadata.get('source', '')
        if not source_path:
            return doc.page_content, None
        
        # Create Path object for source file
        source_file = Path(source_path)
        
        # Check if the file exists at the original location
        if source_file.exists():
            return doc.page_content, source_file
        
        # If not found at original location, try to find in documents directory
        if self.documents_path.exists():
            # Try to find by filename
            filename = source_file.name
            alternative_path = self.documents_path / filename
            if alternative_path.exists():
                return doc.page_content, alternative_path
        
        # If original content can't be verified, just return the chunk content
        return doc.page_content, None
    
    def format_context(self, search_results: List[Tuple[Document, float]]) -> str:
        """Format search results into context for the LLM."""
        context_parts = []
        
        for i, (doc, score) in enumerate(search_results, 1):
            content, file_path = self.get_original_content(doc)
            
            # Format metadata
            filename = file_path.name if file_path else "Unknown"
            similarity = 1.0 - score  # Convert distance to similarity
            
            # Add header with metadata
            header = f"[Document {i} - {filename} - Similarity: {similarity:.2f}]"
            
            # Format the chunk content
            context_parts.append(f"{header}\n{content.strip()}\n")
        
        # Join all parts with newlines
        return "\n\n".join(context_parts)
    

    def answer_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Answer a query using vector search and LLM."""
        if not self.vector_store:
            return "Vector store not initialized. Please index documents first.", {}
        
        if not self.anthropic_client:
            return "Anthropic API not configured. Please set up API key.", {}
        
        # Dictionary to store query metadata
        query_metadata = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'search_results': []
        }
        
        try:
            # Search vector store - use a simple message instead of Progress
            console.print("[dim]Searching for relevant documents...[/dim]")
            search_results = self.search_vector_store(query)
            
            # Check if results were found
            if not search_results:
                return "No documents found related to your query.", query_metadata
            
            # Store search results in metadata
            for doc, score in search_results:
                content, file_path = self.get_original_content(doc)
                query_metadata['search_results'].append({
                    'file': str(file_path) if file_path else "Unknown",
                    'similarity': float(1.0 - score),
                    'chunk_index': doc.metadata.get('chunk_index', 'Unknown'),
                    'content_preview': content[:100] + "..." if len(content) > 100 else content
                })
            
            # Format context for LLM
            context = self.format_context(search_results)
            
            # Determine language instruction based on config
            lang_instruction = ""
            if self.config.response_language == "ko":
                lang_instruction = "Please answer in Korean."
            elif self.config.response_language == "en":
                lang_instruction = "Please answer in English."
            
            # Create prompt for LLM
            system_prompt = f"""You are a professional assistant with expertise in document analysis.
    You must answer questions accurately based on the provided documents.

    Follow these rules strictly:
    1. Only use information from the provided documents to answer.
    2. Honestly state when you don't know something that's not in the documents.
    3. Clearly cite your sources by document number (e.g., "According to Document 1...").
    4. Structure your answers logically.
    5. {lang_instruction}
    6. If the documents don't contain a clear answer, honestly say so.
    7. Do not use external knowledge outside the provided documents.
    """

            # Prepare message for Claude - Use a simple message instead of Progress
            console.print("[dim]Generating answer with Claude...[/dim]")
            
            message = self.anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Here are the relevant documents:

    {context}

    Question: {query}

    Please answer the question using only the information in these documents. Do not include information not present in the documents."""
                    }
                ]
            )
            
            # Extract response
            response = message.content[0].text
            
            # Update token usage statistics
            query_metadata['tokens_used'] = {
                'input': message.usage.input_tokens,
                'output': message.usage.output_tokens,
                'total': message.usage.input_tokens + message.usage.output_tokens
            }
            
            # Calculate approximate cost
            # Claude 3 Opus costs $15 per million input tokens and $75 per million output tokens
            input_cost = (message.usage.input_tokens / 1_000_000) * 15
            output_cost = (message.usage.output_tokens / 1_000_000) * 75
            total_cost = input_cost + output_cost
            
            query_metadata['cost'] = {
                'input': input_cost,
                'output': output_cost,
                'total': total_cost
            }
            
            # Update global statistics
            self.config.total_tokens_used += message.usage.input_tokens + message.usage.output_tokens
            self.config.total_cost += total_cost
            
            # Append to query history
            self.config.query_history.append(query_metadata)
            
            return response, query_metadata
            
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            logger.error(traceback.format_exc())
            return f"An error occurred: {str(e)}", query_metadata
       
    def display_search_results(self, search_results: List[Tuple[Document, float]]):
        """Display search results in a table format."""
        if not search_results:
            console.print("[yellow]No search results found.[/yellow]")
            return
        
        # Create table for results
        table = Table(title="Search Results")
        table.add_column("#", style="cyan")
        table.add_column("Document", style="green")
        table.add_column("Chunk", style="blue")
        table.add_column("Similarity", style="magenta")
        table.add_column("Content Preview", style="white")
        
        for i, (doc, score) in enumerate(search_results, 1):
            content, file_path = self.get_original_content(doc)
            filename = file_path.name if file_path else "Unknown"
            similarity = 1.0 - score  # Convert distance to similarity
            
            # Create content preview (first 70 characters)
            preview = content.strip()[:70] + "..." if len(content.strip()) > 70 else content.strip()
            
            # Add row to table
            table.add_row(
                str(i),
                filename,
                str(doc.metadata.get('chunk_index', 'Unknown')),
                f"{similarity:.4f}",
                preview
            )
        
        console.print(table)
    
    def test_query(self):
        """Test a query and display results without using LLM."""
        if not self.vector_store:
            console.print("[red]Vector store not initialized. Please index documents first.[/red]")
            return
        
        # Get query from user
        query = Prompt.ask("\n[bold]Enter your query")
        
        if not query.strip():
            console.print("[yellow]No query entered.[/yellow]")
            return
        
        # Get number of results to show
        k = int(Prompt.ask("Number of results to display", default="5"))
        
        console.print(f"\n[bold]Searching for:[/bold] {query}")
        
        try:
            # Perform similarity search
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Searching..."),
                console=console
            ) as progress:
                task = progress.add_task("Searching", total=1)
                results = self.search_vector_store(query, k=k)
                progress.update(task, completed=1)
            
            # Display search results
            self.display_search_results(results)
            
            # Ask if user wants to see detailed content
            if results and Confirm.ask("Would you like to see the full content of a specific result?", default=False):
                result_num = int(Prompt.ask(
                    f"Enter result number (1-{len(results)})",
                    default="1"
                ))
                
                if 1 <= result_num <= len(results):
                    doc, score = results[result_num - 1]
                    content, file_path = self.get_original_content(doc)
                    
                    # Display full content in a panel
                    filename = file_path.name if file_path else "Unknown"
                    similarity = 1.0 - score
                    
                    # Display metadata and content
                    console.print(Panel(
                        f"[bold]File:[/bold] {filename}\n"
                        f"[bold]Chunk:[/bold] {doc.metadata.get('chunk_index', 'Unknown')}\n"
                        f"[bold]Similarity:[/bold] {similarity:.4f}\n\n"
                        f"{content}",
                        title=f"Result {result_num} Details",
                        expand=False
                    ))
                else:
                    console.print("[yellow]Invalid result number.[/yellow]")
        
        except Exception as e:
            console.print(f"[red]Error during search: {str(e)}[/red]")
            logger.error(f"Error in test query: {e}")
            logger.error(traceback.format_exc())
    
    def interactive_query(self):
        """Interactive query session with the assistant."""
        if not self.vector_store:
            console.print("[red]Vector store not initialized. Please index documents first.[/red]")
            return
        
        if not self.anthropic_client:
            console.print("[red]Anthropic API not configured. Please set up API key.[/red]")
            return
        
        console.print("[bold blue]Starting conversation with the document assistant[/bold blue]")
        console.print("Enter your questions. Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            query = Prompt.ask("\n[bold cyan]Question")
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query.strip():
                continue
            
            # CHANGE: Remove the outer console.status to prevent nested live displays
            # Instead, add a simple message before answering
            console.print("[bold green]Generating answer...[/bold green]")
            answer, metadata = self.answer_query(query)
            
            # Parse and display answer with markdown formatting
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(answer))
            
            # Display usage statistics
            if 'tokens_used' in metadata and 'cost' in metadata:
                tokens = metadata['tokens_used']
                cost = metadata['cost']
                
                console.print(f"\n[dim]Token usage: Input {tokens['input']}, Output {tokens['output']}, "
                            f"Total {tokens['total']} tokens[/dim]")
                console.print(f"[dim]Estimated cost: ${cost['total']:.6f}[/dim]")
                
    def configure_api(self):
        """Configure Anthropic API settings."""
        console.print("[bold blue]Anthropic API Configuration[/bold blue]")
        
        # Get API key
        current_key = self.config.anthropic_api_key
        masked_key = "********" + current_key[-4:] if current_key else ""
        
        new_key = Prompt.ask(
            "Anthropic API Key",
            default=masked_key,
            password=True
        )
        
        # Only update if changed (not masked)
        if new_key and not new_key.startswith("********"):
            self.config.anthropic_api_key = new_key
        
        # Select model
        console.print("\n[bold]Model Selection:[/bold]")
        console.print("1. Claude 3 Opus (highest quality, most expensive)")
        console.print("2. Claude 3 Sonnet (balanced)")
        console.print("3. Claude 3 Haiku (fastest)")
        
        model_choice = Prompt.ask(
            "Select a model",
            choices=["1", "2", "3"],
            default="1"
        )
        
        if model_choice == "1":
            self.config.model = "claude-3-opus-20240229"
        elif model_choice == "2":
            self.config.model = "claude-3-sonnet-20240229"
        elif model_choice == "3":
            self.config.model = "claude-3-haiku-20240307"
        
        # Advanced settings
        console.print("\n[bold]Advanced Settings:[/bold]")
        
        self.config.temperature = float(Prompt.ask(
            "Temperature (0.0-1.0)",
            default=str(self.config.temperature)
        ))
        
        self.config.max_tokens = int(Prompt.ask(
            "Maximum response tokens",
            default=str(self.config.max_tokens)
        ))
        
        # Apply and save settings
        self.config.save()
        
        # Reinitialize Anthropic client
        try:
            self.anthropic_client = Anthropic(api_key=self.config.anthropic_api_key)
            console.print("[green]Anthropic API settings updated.[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing API client: {str(e)}[/red]")
    
    def configure_settings(self):
        """Configure application settings."""
        console.print("[bold blue]Application Settings[/bold blue]")
        
        # Vector store settings
        console.print("\n[bold]Vector Store Settings:[/bold]")
        
        self.config.vector_db_path = Prompt.ask(
            "Vector store path",
            default=self.config.vector_db_path
        )
        
        self.config.documents_path = Prompt.ask(
            "Document directory path",
            default=self.config.documents_path
        )
        
        # Query settings
        console.print("\n[bold]Search Settings:[/bold]")
        
        self.config.max_chunks = int(Prompt.ask(
            "Maximum chunks to retrieve",
            default=str(self.config.max_chunks)
        ))
        
        self.config.similarity_threshold = float(Prompt.ask(
            "Similarity threshold (0.0-1.0)",
            default=str(self.config.similarity_threshold)
        ))
        
        # Language preference
        console.print("\n[bold]Language Settings:[/bold]")
        console.print("1. English")
        console.print("2. Korean")
        
        lang_choice = Prompt.ask(
            "Select response language",
            choices=["1", "2"],
            default="1"
        )
        
        if lang_choice == "1":
            self.config.response_language = "en"
        else:
            self.config.response_language = "ko"
        
        # Embedding model settings
        console.print("\n[bold]Embedding Model:[/bold]")
        console.print("1. all-MiniLM-L6-v2 (fast, lower quality)")
        console.print("2. msmarco-distilbert-base-v4 (balanced)")
        console.print("3. all-mpnet-base-v2 (slower, high quality)")
        
        model_choice = Prompt.ask(
            "Choose embedding model",
            choices=["1", "2", "3"],
            default="1"
        )
        
        if model_choice == "1":
            self.config.embedding_model = "all-MiniLM-L6-v2"
        elif model_choice == "2":
            self.config.embedding_model = "msmarco-distilbert-base-v4"
        elif model_choice == "3":
            self.config.embedding_model = "all-mpnet-base-v2"
        
        # Save settings
        self.config.save()
        console.print("[green]Settings saved.[/green]")
        
        # Reinitialize components with new settings
        if Confirm.ask("Apply new settings now?", default=True):
            self.vector_db_path = Path(self.config.vector_db_path)
            self.documents_path = Path(self.config.documents_path)
            self._initialize_components()
            console.print("[green]Settings applied.[/green]")
    
    def show_vector_statistics(self):
        """Show vector store statistics."""
        console.print("[bold blue]Vector Store Statistics[/bold blue]")

        # Basic information about the vector store
        console.print(f"\n[bold]General Information:[/bold]")
        console.print(f"Vector Store Path: {self.vector_db_path}")
        console.print(f"Documents Path: {self.documents_path}")
        console.print(f"Vector Store Exists: {'Yes' if self.vector_store else 'No'}")
        console.print(f"Embedding Model: {self.config.embedding_model}")
        
        console.print(f"\n[bold]Configuration:[/bold]")
        console.print(f"Max Chunks: {self.config.max_chunks}")
        console.print(f"Similarity Threshold: {self.config.similarity_threshold}")
        
        # If vector store exists, try to get some statistics from it
        if self.vector_store:
            try:
                # Try to get count of documents in vector store
                doc_count = len(self.vector_store.docstore._dict)
                console.print(f"\n[bold]Vector Store Contents:[/bold]")
                console.print(f"Document Chunks: {doc_count}")
            except Exception as e:
                console.print(f"[yellow]Could not retrieve detailed vector statistics: {str(e)}[/yellow]")
                        
    def show_statistics(self):
        """Show usage statistics."""

        self.show_vector_statistics()
        console.print("[bold blue]Usage Statistics[/bold blue]")
        
        # Check if vector store is initialized
        vector_store_status = "[green]Initialized" if self.vector_store else "[red]Not initialized"
        console.print(f"\n[bold]Vector Store:[/bold] {vector_store_status}")
        console.print(f"Vector store path: {self.config.vector_db_path}")
        
        # API statistics
        api_status = "[green]Connected" if self.anthropic_client else "[red]Not connected"
        console.print(f"\n[bold]Anthropic API:[/bold] {api_status}")
        console.print(f"Model: {self.config.model}")
        
        # Token and cost statistics
        console.print(f"\n[bold]Token Usage:[/bold] {self.config.total_tokens_used:,} tokens")
        console.print(f"[bold]Estimated Total Cost:[/bold] ${self.config.total_cost:.6f}")
        
        # Query history
        history_count = len(self.config.query_history)
        console.print(f"\n[bold]Query History:[/bold] {history_count} queries")
        
        if history_count > 0 and Confirm.ask("View query history?", default=False):
            history_table = Table(title="Recent Query History")
            history_table.add_column("#", style="cyan")
            history_table.add_column("Query", style="green")
            history_table.add_column("Tokens", style="blue")
            history_table.add_column("Cost", style="magenta")
            history_table.add_column("Time", style="dim")
            
            # Show 10 most recent queries
            for i, query_data in enumerate(self.config.query_history[-10:], 1):
                tokens = query_data.get('tokens_used', {}).get('total', 0)
                cost = query_data.get('cost', {}).get('total', 0.0)
                
                # Format timestamp
                timestamp = query_data.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                
                # Format query text (truncate if too long)
                query_text = query_data.get('query', '')
                if len(query_text) > 50:
                    query_text = query_text[:47] + "..."
                
                history_table.add_row(
                    str(i),
                    query_text,
                    f"{tokens:,}",
                    f"${cost:.6f}",
                    timestamp
                )
            
            console.print(history_table)
    
    def run(self):
        """Run the CLI application."""
        while True:
            console.clear()
            console.print("[bold blue]Vector Query Assistant[/bold blue]")
            console.print("[yellow]=====================[/yellow]")
            console.print("Document-based Q&A system using vector search and LLM")
            
            # Status check
            vector_status = "✅" if self.vector_store else "❌"
            api_status = "✅" if self.anthropic_client else "❌"
            
            console.print(f"\n[bold]System Status:[/bold]")
            console.print(f"Vector Store: {vector_status} | Anthropic API: {api_status}")
            console.print(f"Model: {self.config.model} | Language: {'Korean' if self.config.response_language == 'ko' else 'English'}")
            
            # Create menu
            console.print("\n[bold]Menu:[/bold]")
            console.print("1. Chat with Document Assistant")
            console.print("2. Test Vector Search")
            console.print("3. Configure API")
            console.print("4. Application Settings")
            console.print("5. View Statistics")
            console.print("0. Exit")
            
            choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5"], default="1")
            
            if choice == "0":
                break
            elif choice == "1":
                # Interactive query
                self.interactive_query()
                Prompt.ask("Press Enter to continue")
            elif choice == "2":
                # Test vector search
                self.test_query()
                Prompt.ask("Press Enter to continue")
            elif choice == "3":
                # Configure API
                self.configure_api()
                Prompt.ask("Press Enter to continue")
            elif choice == "4":
                # Configure application settings
                self.configure_settings()
                Prompt.ask("Press Enter to continue")
            elif choice == "5":
                # Show statistics
                self.show_statistics()
                Prompt.ask("Press Enter to continue")
        
        console.print("[green]Exiting Vector Query Assistant. Thank you![/green]")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Vector Query Assistant")
    parser.add_argument("--vector-dir", help="Vector database directory")
    parser.add_argument("--docs-dir", help="Documents directory")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--model", help="Anthropic model name")
    parser.add_argument("--language", choices=["ko", "en"], help="Response language (ko or en)")
    parser.add_argument("--query", help="Run a single query non-interactively")
    
    args = parser.parse_args()
    
    try:
        # Create assistant
        config = Config()
        
        # Apply command-line arguments if provided
        if args.vector_dir:
            config.vector_db_path = args.vector_dir
        if args.docs_dir:
            config.documents_path = args.docs_dir
        if args.api_key:
            config.anthropic_api_key = args.api_key
        if args.model:
            config.model = args.model
        if args.language:
            config.response_language = args.language
            
        assistant = VectorQueryAssistant(config)
        
        # Check if we're running in single-query mode
        if args.query:
            # Run single query
            if not assistant.vector_store:
                console.print("[red]Could not initialize vector store.[/red]")
                return 1
                
            if not assistant.anthropic_client:
                console.print("[red]Could not initialize Anthropic API client.[/red]")
                return 1
                
            console.print(f"[bold]Question:[/bold] {args.query}")
            
            with console.status("[bold green]Generating answer..."):
                answer, metadata = assistant.answer_query(args.query)
            
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(answer))
            
            # Display usage statistics
            if 'tokens_used' in metadata and 'cost' in metadata:
                tokens = metadata['tokens_used']
                cost = metadata['cost']
                
                console.print(f"\n[dim]Token usage: Input {tokens['input']}, Output {tokens['output']}, "
                            f"Total {tokens['total']} tokens[/dim]")
                console.print(f"[dim]Estimated cost: ${cost['total']:.6f}[/dim]")
        else:
            # Run interactive mode
            assistant.run()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Program terminated by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
