#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Generator
-------------------------
A CLI tool that extracts entities and relations from documents and stores them in a Neo4j graph database.
Supports .doc, .docx, .pdf, and .txt files.
"""

import os
import sys
import time
import json
import argparse
import logging
import re
import tempfile
import textract
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set, Union
from datetime import datetime
from collections import Counter

# Document processing
import docx
import PyPDF2

import chardet

# NLP libraries
import spacy
import en_core_web_sm

# Neo4j connection
from neo4j import GraphDatabase

# For local LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# For Anthropic API
import anthropic

# Terminal UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.layout import Layout
from rich.text import Text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KG_Generator")

# Initialize console
console = Console()

class Config:
    """Configuration for the application."""
    
    def __init__(self):
        self.batch_dir = os.path.join(os.getcwd(), "batch")
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.anthropic_api_key = ""
        self.extraction_method = "spacy"  # Options: "spacy", "local_llm", "anthropic"
        self.local_llm_model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        self.anthropic_calls = 0
        self.anthropic_cost = 0.0
        self.last_scan_time = None
        self.scanned_files = []
        self.processed_entities = 0
        self.processed_relations = 0
        
    def save(self, filename="config.json"):
        """Save configuration to a file."""
        save_data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
            
    def load(self, filename="config.json"):
        """Load configuration from a file."""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, value in data.items():
                    setattr(self, key, value)


class DocumentProcessor:
    """Process different document types and extract text content."""
    
    @staticmethod
    def detect_encoding(file_path):
        """Detect file encoding."""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    
    @staticmethod
    def read_text_file(file_path):
        """Read content from a text file."""
        try:
            encoding = DocumentProcessor.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return ""
            
    @staticmethod
    def read_docx_file(file_path):
        """Read content from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def read_doc_file(file_path):
        """Read content from a DOC file using textract."""
        try:
            text = textract.process(file_path).decode('utf-8')
            return text
        except Exception as e:
            logger.error(f"Error reading DOC file {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def read_pdf_file(file_path):
        """Read content from a PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text(file_path):
        """Extract text from a document based on its extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                return DocumentProcessor.read_text_file(file_path)
            elif file_extension == '.docx':
                return DocumentProcessor.read_docx_file(file_path)
            elif file_extension == '.doc':
                return DocumentProcessor.read_doc_file(file_path)
            elif file_extension == '.pdf':
                return DocumentProcessor.read_pdf_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""


class EntityRelationExtractor:
    """Extract entities and relations from text using different methods."""
    
    def __init__(self, config):
        self.config = config
        self.nlp = None
        self.local_llm = None
        self.anthropic_client = None
        
        # Initialize based on extraction method
        if self.config.extraction_method == "spacy":
            self._init_spacy()
        elif self.config.extraction_method == "local_llm":
            self._init_local_llm()
        elif self.config.extraction_method == "anthropic":
            self._init_anthropic()
    
    def _init_spacy(self):
        """Initialize spaCy NLP model."""
        try:
            self.nlp = en_core_web_sm.load()
            logger.info("Initialized spaCy NLP model")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            self.nlp = None
    
    def _init_local_llm(self):
        """Initialize local LLM for entity-relation extraction."""
        try:
            logger.info(f"Loading local LLM model: {self.config.local_llm_model}")
            
            # Initialize model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.local_llm_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.local_llm_model,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Create a pipeline for text generation
            self.local_llm = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2048,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            logger.info("Local LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {str(e)}")
            self.local_llm = None
    
    def _init_anthropic(self):
        """Initialize Anthropic API client."""
        if not self.config.anthropic_api_key:
            logger.error("Anthropic API key not provided")
            self.anthropic_client = None
            return
            
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            logger.info("Initialized Anthropic API client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            self.anthropic_client = None
    
    def extract_with_spacy(self, text, file_path):
        """Extract entities and relations using spaCy."""
        if not self.nlp:
            logger.error("spaCy model not initialized")
            return [], []
            
        entities = []
        relations = []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                entity = {
                    "name": ent.text,
                    "entityType": ent.label_,
                    "source": os.path.basename(file_path)
                }
                entities.append(entity)
            
            # Try to extract relations from dependency parse
            for sent in doc.sents:
                for token in sent:
                    # Look for verbs that might indicate relationships
                    if token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp", "ccomp"]:
                        subj = None
                        obj = None
                        
                        # Find subject and object
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"] and child.ent_type_:
                                subj = child.text
                            elif child.dep_ in ["dobj", "pobj"] and child.ent_type_:
                                obj = child.text
                        
                        # If we have both subject and object, create a relation
                        if subj and obj:
                            relation = {
                                "from": subj,
                                "to": obj,
                                "relationType": token.lemma_,
                                "source": os.path.basename(file_path)
                            }
                            relations.append(relation)
            
            return entities, relations
        except Exception as e:
            logger.error(f"Error in spaCy extraction: {str(e)}")
            return [], []
    
    def extract_with_local_llm(self, text, file_path):
        """Extract entities and relations using local LLM."""
        if not self.local_llm:
            logger.error("Local LLM not initialized")
            return [], []
            
        entities = []
        relations = []
        
        # Create prompt for entity and relation extraction
        prompt = f"""
        Extract entities and relationships from the following text. 
        Format the output as JSON with two arrays: "entities" and "relations".
        
        For entities:
        - "name": The entity name
        - "entityType": The type of entity (person, organization, date, etc.)
        
        For relations:
        - "from": The source entity name
        - "to": The target entity name
        - "relationType": Type of relationship between entities
        
        Text to analyze:
        {text[:4000]}  # Limit text length to avoid token limit issues
        
        JSON Output:
        """
        
        try:
            # Generate response from LLM
            response = self.local_llm(prompt, max_length=4096)[0]['generated_text']
            
            # Extract the JSON part from the response
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'{.*}', response, re.DOTALL)
                
            if json_match:
                json_str = json_match.group(1) if '```json' in response else json_match.group(0)
                result = json.loads(json_str)
                
                # Process entities
                for entity in result.get('entities', []):
                    entities.append({
                        "name": entity.get('name', ''),
                        "entityType": entity.get('entityType', 'Unknown'),
                        "source": os.path.basename(file_path)
                    })
                
                # Process relations
                for relation in result.get('relations', []):
                    relations.append({
                        "from": relation.get('from', ''),
                        "to": relation.get('to', ''),
                        "relationType": relation.get('relationType', 'Unknown'),
                        "source": os.path.basename(file_path)
                    })
            
            return entities, relations
        except Exception as e:
            logger.error(f"Error in local LLM extraction: {str(e)}")
            return [], []
    
    def extract_with_anthropic(self, text, file_path):
        """Extract entities and relations using Anthropic API."""
        if not self.anthropic_client:
            logger.error("Anthropic client not initialized")
            return [], []
            
        entities = []
        relations = []
        
        try:
            # Update call statistics
            self.config.anthropic_calls += 1
            
            # Limit text length to avoid token limit issues
            truncated_text = text[:8000]
            
            # Create message for Claude
            message = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Extract entities and their relationships from the following text. 
                        
                        Format the output as JSON with two arrays: "entities" and "relations".
                        
                        For entities:
                        - "name": The entity name
                        - "entityType": The type of entity (person, organization, date, etc.)
                        
                        For relations:
                        - "from": The source entity name
                        - "to": The target entity name
                        - "relationType": Type of relationship between entities (use verb form)
                        
                        Only include clearly defined entities and relationships that are explicitly mentioned in the text.
                        Be specific with entity types and relation types.
                        
                        Text to analyze:
                        ```
                        {truncated_text}
                        ```
                        
                        Return only the JSON, nothing else, in this format:
                        {{
                            "entities": [
                                {{"name": "Entity1", "entityType": "Type1"}},
                                {{"name": "Entity2", "entityType": "Type2"}}
                            ],
                            "relations": [
                                {{"from": "Entity1", "to": "Entity2", "relationType": "RelationType"}}
                            ]
                        }}
                        """
                    }
                ]
            )
            
            # Estimate cost: assuming $15 per million input tokens and $75 per million output tokens
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            cost = (input_tokens * 0.000015) + (output_tokens * 0.000075)
            self.config.anthropic_cost += cost
            
            # Extract JSON from the response
            response_content = message.content[0].text
            json_match = re.search(r'{.*}', response_content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                
                # Process entities
                for entity in result.get('entities', []):
                    entities.append({
                        "name": entity.get('name', ''),
                        "entityType": entity.get('entityType', 'Unknown'),
                        "source": os.path.basename(file_path)
                    })
                
                # Process relations
                for relation in result.get('relations', []):
                    relations.append({
                        "from": relation.get('from', ''),
                        "to": relation.get('to', ''),
                        "relationType": relation.get('relationType', 'Unknown'),
                        "source": os.path.basename(file_path)
                    })
            
            return entities, relations
        except Exception as e:
            logger.error(f"Error in Anthropic API extraction: {str(e)}")
            return [], []
    
    def extract(self, text, file_path):
        """Extract entities and relations using the configured method."""
        if not text.strip():
            return [], []
            
        # Choose extraction method based on configuration
        if self.config.extraction_method == "spacy":
            return self.extract_with_spacy(text, file_path)
        elif self.config.extraction_method == "local_llm":
            return self.extract_with_local_llm(text, file_path)
        elif self.config.extraction_method == "anthropic":
            return self.extract_with_anthropic(text, file_path)
        else:
            logger.error(f"Unknown extraction method: {self.config.extraction_method}")
            return [], []


class Neo4jManager:
    """Manage Neo4j database operations."""
    
    def __init__(self, config):
        self.config = config
        self.driver = None
        
    def connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def create_entity(self, entity):
        """Create an entity node in Neo4j."""
        if not self.driver:
            logger.error("Neo4j connection not established")
            return False
            
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.entityType = $entityType, e.source = $source
                    ON MATCH SET e.entityType = $entityType, e.source = $source
                    RETURN e
                    """,
                    name=entity["name"],
                    entityType=entity["entityType"],
                    source=entity.get("source", "unknown")
                )
                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to create entity: {str(e)}")
            return False
    
    def create_relationship(self, relation):
        """Create a relationship between entities in Neo4j."""
        if not self.driver:
            logger.error("Neo4j connection not established")
            return False
            
        try:
            with self.driver.session() as session:
                # Create a relationship between two entities
                result = session.run(
                    """
                    MATCH (from:Entity {name: $from_entity})
                    MATCH (to:Entity {name: $to_entity})
                    MERGE (from)-[r:RELATED {type: $relation_type}]->(to)
                    ON CREATE SET r.source = $source
                    ON MATCH SET r.source = $source
                    RETURN r
                    """,
                    from_entity=relation["from"],
                    to_entity=relation["to"],
                    relation_type=relation["relationType"],
                    source=relation.get("source", "unknown")
                )
                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to create relationship: {str(e)}")
            return False
    
    def get_entities(self, limit=100):
        """Get entities from Neo4j."""
        if not self.driver:
            logger.error("Neo4j connection not established")
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.entityType AS entityType, e.source AS source
                    LIMIT $limit
                    """,
                    limit=limit
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to get entities: {str(e)}")
            return []
    
    def get_relationships(self, limit=100):
        """Get relationships from Neo4j."""
        if not self.driver:
            logger.error("Neo4j connection not established")
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (from:Entity)-[r:RELATED]->(to:Entity)
                    RETURN from.name AS from, to.name AS to, r.type AS relationType, r.source AS source
                    LIMIT $limit
                    """,
                    limit=limit
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to get relationships: {str(e)}")
            return []
    
    def get_stats(self):
        """Get database statistics."""
        if not self.driver:
            logger.error("Neo4j connection not established")
            return {}
            
        try:
            with self.driver.session() as session:
                # Get entity count
                entity_result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
                entity_count = entity_result.single()["count"]
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS count")
                rel_count = rel_result.single()["count"]
                
                # Get entity types - modified query without explicit GROUP BY
                entity_types_result = session.run(
                    """
                    MATCH (e:Entity)
                    WITH e.entityType AS type, count(*) AS count
                    RETURN type, count
                    ORDER BY count DESC
                    """
                )
                entity_types = {record["type"]: record["count"] for record in entity_types_result}
                
                # Get relation types - modified query without explicit GROUP BY
                relation_types_result = session.run(
                    """
                    MATCH ()-[r:RELATED]->()
                    WITH r.type AS type, count(*) AS count
                    RETURN type, count
                    ORDER BY count DESC
                    """
                )
                relation_types = {record["type"]: record["count"] for record in relation_types_result}
                
                return {
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                    "entity_types": entity_types,
                    "relationship_types": relation_types
                }
        except Exception as e:
            logger.error(f"Failed to get database statistics: {str(e)}")
            return {}
        

class KnowledgeGraphApp:
    """Main application class."""
    
    def __init__(self):
        self.config = Config()
        
        # Try to load configuration
        try:
            self.config.load()
        except:
            # If loading fails, use default configuration
            pass
        
        # Ensure batch directory exists
        os.makedirs(self.config.batch_dir, exist_ok=True)
        
        # Initialize components
        self.extractor = None
        self.neo4j_manager = Neo4jManager(self.config)
    
    def initialize_extractor(self):
        """Initialize the entity-relation extractor."""
        self.extractor = EntityRelationExtractor(self.config)
    
    def scan_directory(self, directory_path=None):
        """Scan directory for documents and extract entities and relations."""
        if directory_path is None:
            directory_path = self.config.batch_dir
        
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            console.print(f"[red]Directory not found: {directory_path}[/red]")
            return
        
        # Connect to Neo4j
        if not self.neo4j_manager.connect():
            console.print("[red]Failed to connect to Neo4j database. Please check your configuration.[/red]")
            return
        
        
        # Initialize extractor if not already initialized
        if self.extractor is None:
            self.initialize_extractor()
        
        # Get list of supported files
        supported_extensions = ['.txt', '.doc', '.docx', '.pdf']
        files = []
        
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(root, filename))
        
        if not files:
            console.print(f"[yellow]No supported documents found in {directory_path}[/yellow]")
            return
        
        # Process files with progress bar
        console.print(f"[green]Found {len(files)} documents to process[/green]")
        self.config.scanned_files = files
        self.config.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_entities = 0
        total_relations = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            console=console
        ) as progress:
            process_task = progress.add_task("[green]Processing documents...", total=len(files))
            
            for file_path in files:
                progress.update(process_task, description=f"[cyan]Processing {os.path.basename(file_path)}[/cyan]")
                
                # Extract text from file
                text = DocumentProcessor.extract_text(file_path)
                
                if not text:
                    progress.advance(process_task)
                    continue
                
                # Extract entities and relations
                entities, relations = self.extractor.extract(text, file_path)
                
                # Store entities in Neo4j
                for entity in entities:
                    if self.neo4j_manager.create_entity(entity):
                        total_entities += 1
                
                # Store relations in Neo4j
                for relation in relations:
                    if self.neo4j_manager.create_relationship(relation):
                        total_relations += 1
                
                progress.advance(process_task)
        
        # Update statistics
        self.config.processed_entities += total_entities
        self.config.processed_relations += total_relations
        
        # Save configuration
        self.config.save()
        
        # Display summary
        console.print(f"[green]Processing complete![/green]")
        console.print(f"[blue]Extracted {total_entities} entities and {total_relations} relations from {len(files)} documents[/blue]")
        
        # Close Neo4j connection
        self.neo4j_manager.close()
    
    def show_results(self):
        """Show entities and relations extracted from documents."""
        # Connect to Neo4j
        if not self.neo4j_manager.connect():
            console.print("[red]Failed to connect to Neo4j database. Please check your configuration.[/red]")
            return
        
        # Get entities
        entities = self.neo4j_manager.get_entities(limit=100)
        
        if entities:
            # Create entities table
            entity_table = Table(title="Extracted Entities")
            entity_table.add_column("Name", style="cyan")
            entity_table.add_column("Type", style="green")
            entity_table.add_column("Source", style="magenta")
            
            for entity in entities:
                entity_table.add_row(
                    entity.get("name", ""),
                    entity.get("entityType", ""),
                    entity.get("source", "")
                )
            
            console.print(entity_table)
        else:
            console.print("[yellow]No entities found in the database.[/yellow]")
        
        # Get relationships
        relationships = self.neo4j_manager.get_relationships(limit=100)
        
        if relationships:
            # Create relationships table
            relation_table = Table(title="Extracted Relationships")
            relation_table.add_column("From", style="cyan")
            relation_table.add_column("Relation", style="red")
            relation_table.add_column("To", style="green")
            relation_table.add_column("Source", style="magenta")
            
            for relation in relationships:
                relation_table.add_row(
                    relation.get("from", ""),
                    relation.get("relationType", ""),
                    relation.get("to", ""),
                    relation.get("source", "")
                )
            
            console.print(relation_table)
        else:
            console.print("[yellow]No relationships found in the database.[/yellow]")
        
        # Get database statistics
        stats = self.neo4j_manager.get_stats()
        
        if stats:
            console.print("\n[bold]Database Statistics:[/bold]")
            console.print(f"Total Entities: {stats.get('entity_count', 0)}")
            console.print(f"Total Relationships: {stats.get('relationship_count', 0)}")
            
            if 'entity_types' in stats and stats['entity_types']:
                console.print("\n[bold]Entity Types:[/bold]")
                for entity_type, count in list(stats['entity_types'].items())[:10]:  # Show top 10
                    console.print(f"- {entity_type}: {count}")
            
            if 'relationship_types' in stats and stats['relationship_types']:
                console.print("\n[bold]Relationship Types:[/bold]")
                for relation_type, count in list(stats['relationship_types'].items())[:10]:  # Show top 10
                    console.print(f"- {relation_type}: {count}")
        
        # Close connection
        self.neo4j_manager.close()
    
    def show_statistics(self):
        """Show application statistics."""
        console.print("[bold blue]Application Statistics[/bold blue]")
        
        # General statistics
        console.print("\n[bold]General Statistics:[/bold]")
        console.print(f"Extraction Method: {self.config.extraction_method}")
        console.print(f"Last Scan Time: {self.config.last_scan_time or 'Never'}")
        console.print(f"Processed Files: {len(self.config.scanned_files)}")
        console.print(f"Processed Entities: {self.config.processed_entities}")
        console.print(f"Processed Relations: {self.config.processed_relations}")
        
        # Anthropic API statistics
        if self.config.extraction_method == "anthropic":
            console.print("\n[bold]Anthropic API Statistics:[/bold]")
            console.print(f"Total API Calls: {self.config.anthropic_calls}")
            console.print(f"Estimated Cost: ${self.config.anthropic_cost:.4f}")
        
        # Connect to Neo4j for additional statistics
        if self.neo4j_manager.connect():
            # Get database statistics
            stats = self.neo4j_manager.get_stats()
            
            if stats:
                console.print("\n[bold]Database Statistics:[/bold]")
                console.print(f"Total Entities: {stats.get('entity_count', 0)}")
                console.print(f"Total Relationships: {stats.get('relationship_count', 0)}")
            
            # Close connection
            self.neo4j_manager.close()
    
    def configure_app(self):
        """Configure application settings."""
        console.print("[bold blue]Application Configuration[/bold blue]")
        
        # Batch directory
        current_batch_dir = self.config.batch_dir
        new_batch_dir = Prompt.ask(
            "Batch directory path", 
            default=current_batch_dir
        )
        self.config.batch_dir = new_batch_dir
        
        # Neo4j connection
        console.print("\n[bold]Neo4j Connection:[/bold]")
        self.config.neo4j_uri = Prompt.ask(
            "Neo4j URI", 
            default=self.config.neo4j_uri
        )
        self.config.neo4j_user = Prompt.ask(
            "Neo4j Username", 
            default=self.config.neo4j_user
        )
        self.config.neo4j_password = Prompt.ask(
            "Neo4j Password", 
            default=self.config.neo4j_password,
            password=True
        )
        
        # Extraction method
        console.print("\n[bold]Entity-Relation Extraction Method:[/bold]")
        extraction_options = {
            "1": "spaCy NLP (Default)",
            "2": "Local LLM",
            "3": "Anthropic API (Claude)"
        }
        
        for key, value in extraction_options.items():
            console.print(f"{key}. {value}")
        
        extraction_choice = Prompt.ask(
            "Select extraction method",
            choices=["1", "2", "3"],
            default="1"
        )
        
        if extraction_choice == "1":
            self.config.extraction_method = "spacy"
        elif extraction_choice == "2":
            self.config.extraction_method = "local_llm"
            self.config.local_llm_model = Prompt.ask(
                "Local LLM model",
                default=self.config.local_llm_model
            )
        elif extraction_choice == "3":
            self.config.extraction_method = "anthropic"
            self.config.anthropic_api_key = Prompt.ask(
                "Anthropic API Key",
                default=self.config.anthropic_api_key,
                password=True
            )
        
        # Save configuration
        self.config.save()
        console.print("[green]Configuration saved successfully![/green]")
        
        # Re-initialize extractor with new configuration
        self.initialize_extractor()
    
    def test_connection(self):
        """Test Neo4j connection."""
        console.print("[bold blue]Testing Neo4j Connection...[/bold blue]")
        
        if self.neo4j_manager.connect():
            console.print("[green]Connection successful![/green]")
            self.neo4j_manager.close()
        else:
            console.print("[red]Connection failed. Please check your Neo4j configuration.[/red]")
    
    def run(self):
        """Run the CLI application."""
        while True:
            console.clear()
            console.print("[bold blue]Knowledge Graph Generator[/bold blue]")
            console.print("[yellow]============================[/yellow]")
            console.print("A tool to extract entities and relations from documents and store them in Neo4j")
            console.print("\n[bold]Current Configuration:[/bold]")
            console.print(f"Batch Directory: {self.config.batch_dir}")
            console.print(f"Extraction Method: {self.config.extraction_method}")
            console.print(f"Neo4j URI: {self.config.neo4j_uri}")
            
            console.print("\n[bold]Menu:[/bold]")
            console.print("1. Scan Directory")
            console.print("2. ...")
            console.print("3. Show Results")
            console.print("4. Show Statistics")
            console.print("5. Configure Application")
            console.print("6. Test Neo4j Connection")
            console.print("0. Exit")
            
            choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5", "6"], default="1")
            
            if choice == "0":
                break
            elif choice == "1":
                custom_dir = Confirm.ask("Use custom directory?", default=False)
                if custom_dir:
                    dir_path = Prompt.ask("Enter directory path", default=self.config.batch_dir)
                    self.scan_directory(dir_path)
                else:
                    self.scan_directory()
                Prompt.ask("Press Enter to continue")
            elif choice == "2":
                Prompt.ask("Press Enter to continue")
            elif choice == "3":
                self.show_results()
                Prompt.ask("Press Enter to continue")
            elif choice == "4":
                self.show_statistics()
                Prompt.ask("Press Enter to continue")
            elif choice == "5":
                self.configure_app()
                Prompt.ask("Press Enter to continue")
            elif choice == "6":
                self.test_connection()
                Prompt.ask("Press Enter to continue")
        
        console.print("[green]Thank you for using Knowledge Graph Generator![/green]")


def main():
    """Main entry point."""
    try:
        # Display startup banner
        console.print("[bold blue]Knowledge Graph Generator[/bold blue]")
        console.print("[yellow]============================[/yellow]")
        console.print("Starting application...")
        
        # Run the application
        app = KnowledgeGraphApp()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Application terminated by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        console.print("[red]Please check the log file for details.[/red]")
    finally:
        console.print("[green]Exiting application. Goodbye![/green]")


if __name__ == "__main__":
    main()