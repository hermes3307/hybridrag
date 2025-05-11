#!/usr/bin/env python3
"""
Vector Database + LLM Query System
Combines vector search with LLM to answer complex queries and generate reports
Updated to use latest LangChain packages
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Updated imports for LangChain
try:
    # Try new imports first
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_anthropic import ChatAnthropic
except ImportError:
    # Fallback to alternative imports
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        # Last resort - old imports
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.chat_models import ChatAnthropic

# Core LangChain imports
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

class VectorLLMQuerySystem:
    """System that combines vector database with LLM for intelligent querying"""
    
    def __init__(
        self,
        vector_db_path: str = "./vector_database",
        llm_provider: str = "claude",
        api_key: Optional[str] = None,
        model_name: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7
    ):
        """
        Initialize the query system
        
        Args:
            vector_db_path: Path to vector database
            llm_provider: LLM provider ("claude", "openai", etc.)
            api_key: API key for LLM provider
            model_name: Specific model to use
            temperature: LLM temperature setting
        """
        self.vector_db_path = Path(vector_db_path)
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        
        # Set API key
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Please provide Anthropic API key or set ANTHROPIC_API_KEY environment variable")
        
        # Initialize components
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._load_vector_store()
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_embeddings(self):
        """Initialize embeddings with fallback options"""
        try:
            # Try to use HuggingFaceEmbeddings from the correct package
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            self.logger.warning(f"Error initializing embeddings: {e}")
            # Fallback to basic configuration
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def _load_vector_store(self) -> FAISS:
        """Load the vector store"""
        if not (self.vector_db_path / "index.faiss").exists():
            raise FileNotFoundError(f"Vector database not found at {self.vector_db_path}")
        
        return FAISS.load_local(
            str(self.vector_db_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def _initialize_llm(self):
        """Initialize the LLM with proper configuration"""
        if self.llm_provider == "claude":
            try:
                # Try the new import structure
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=4096,
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            except ImportError:
                # Fallback to older structure (not recommended)
                try:
                    from langchain_community.chat_models import ChatAnthropic
                    return ChatAnthropic(
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=4096
                    )
                except Exception as e:
                    self.logger.error(f"Failed to initialize LLM: {e}")
                    raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def create_qa_chain(self):
        """Create a basic question-answering chain"""
        template = """You are a helpful AI assistant with access to a knowledge base of documents. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite your sources when using information from the context.

        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def create_conversation_chain(self):
        """Create a conversational chain with memory"""
        template = """You are a helpful AI assistant with access to a knowledge base of documents.
        Use the following pieces of context and chat history to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite your sources when using information from the context.

        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        
        Assistant:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """
        Query the system with a question
        
        Args:
            question: The question to ask
            use_conversation: Whether to use conversational chain with memory
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            if use_conversation:
                chain = self.create_conversation_chain()
                result = chain({"question": question})
            else:
                chain = self.create_qa_chain()
                result = chain({"query": question})
            
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=5)
            
            return {
                "answer": result.get("result", result.get("answer", "")),
                "relevant_documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    } for doc in relevant_docs
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during query: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_report(self, topic: str, report_type: str = "summary") -> str:
        """
        Generate a report on a specific topic
        
        Args:
            topic: The topic to generate a report about
            report_type: Type of report ("summary", "detailed", "analysis")
            
        Returns:
            Generated report as string
        """
        templates = {
            "summary": """Based on the provided context, create a concise summary report about {topic}.
            Include key facts, important details, and main findings.
            
            Context: {context}
            
            Summary Report on {topic}:""",
            
            "detailed": """Based on the provided context, create a detailed report about {topic}.
            Include background information, comprehensive analysis, specific examples, and detailed findings.
            Structure the report with clear sections and subsections.
            
            Context: {context}
            
            Detailed Report on {topic}:""",
            
            "analysis": """Based on the provided context, create an analytical report about {topic}.
            Include data analysis, trends, implications, and recommendations.
            Provide insights and draw conclusions from the available information.
            
            Context: {context}
            
            Analytical Report on {topic}:"""
        }
        
        # Get relevant documents
        relevant_docs = self.vector_store.similarity_search(topic, k=10)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate report
        template = templates.get(report_type, templates["summary"])
        prompt = PromptTemplate(template=template, input_variables=["topic", "context"])
        
        chain = prompt | self.llm
        
        try:
            report = chain.invoke({"topic": topic, "context": context})
            
            # Extract the content from the response
            if hasattr(report, 'content'):
                report_text = report.content
            else:
                report_text = str(report)
            
            # Add source references
            sources = set()
            for doc in relevant_docs:
                if "filename" in doc.metadata:
                    sources.add(doc.metadata["filename"])
            
            report_with_sources = f"{report_text}\n\n---\nSources: {', '.join(sources)}"
            
            return report_with_sources
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    def comparative_analysis(self, items: List[str]) -> str:
        """
        Perform comparative analysis between multiple items
        
        Args:
            items: List of items to compare
            
        Returns:
            Comparative analysis report
        """
        template = """Based on the provided context, create a comparative analysis of the following items: {items}.
        
        Compare and contrast their features, advantages, disadvantages, and use cases.
        Provide a structured comparison with clear sections.
        
        Context: {context}
        
        Comparative Analysis:"""
        
        # Get relevant documents for each item
        all_docs = []
        for item in items:
            docs = self.vector_store.similarity_search(item, k=5)
            all_docs.extend(docs)
        
        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        context = "\n\n".join([doc.page_content for doc in unique_docs])
        
        prompt = PromptTemplate(template=template, input_variables=["items", "context"])
        chain = prompt | self.llm
        
        try:
            analysis = chain.invoke({"items": ", ".join(items), "context": context})
            
            # Extract content from response
            if hasattr(analysis, 'content'):
                return analysis.content
            else:
                return str(analysis)
                
        except Exception as e:
            self.logger.error(f"Error in comparative analysis: {e}")
            return f"Error generating comparative analysis: {str(e)}"
    
    def extract_specific_info(self, entity: str, info_type: str) -> Dict[str, Any]:
        """
        Extract specific information about an entity
        
        Args:
            entity: The entity to extract information about
            info_type: Type of information to extract
            
        Returns:
            Dictionary containing extracted information
        """
        info_templates = {
            "technical_specs": "Extract all technical specifications for {entity}",
            "pricing": "Extract all pricing information for {entity}",
            "features": "List all features and capabilities of {entity}",
            "history": "Provide historical information and timeline for {entity}",
            "comparison": "Compare {entity} with similar products/services",
            "use_cases": "List all use cases and applications for {entity}"
        }
        
        template = info_templates.get(info_type, "Extract information about {entity}")
        
        # Get relevant documents
        relevant_docs = self.vector_store.similarity_search(f"{entity} {info_type}", k=5)
        
        if not relevant_docs:
            return {
                "entity": entity,
                "info_type": info_type,
                "result": "No relevant information found in the database.",
                "sources": []
            }
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = PromptTemplate(
            template=f"{template}\n\nContext: {{context}}\n\nExtracted Information:",
            input_variables=["entity", "context"]
        )
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({"entity": entity, "context": context})
            
            # Extract content from response
            if hasattr(result, 'content'):
                result_text = result.content
            else:
                result_text = str(result)
            
            sources = [doc.metadata.get("filename", "Unknown") for doc in relevant_docs]
            
            return {
                "entity": entity,
                "info_type": info_type,
                "result": result_text,
                "sources": list(set(sources))
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting information: {e}")
            return {
                "entity": entity,
                "info_type": info_type,
                "error": str(e)
            }
    
    def save_conversation(self, filename: Optional[str] = None):
        """Save conversation history"""
        if not filename:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else [],
            "model": self.model_name
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filename


# Example usage and main interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector DB + LLM Query System")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--vector-db", default="./vector_database", help="Path to vector database")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = VectorLLMQuerySystem(
            vector_db_path=args.vector_db,
            api_key=args.api_key
        )
        
        if args.interactive:
            print("🤖 Vector DB + LLM Query System")
            print("=" * 40)
            print("Commands:")
            print("  /report <topic> - Generate a report")
            print("  /compare <item1> <item2> ... - Compare items")
            print("  /extract <entity> <info_type> - Extract specific info")
            print("  /save - Save conversation")
            print("  /exit - Exit")
            print()
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() == "/exit":
                        break
                    elif user_input.startswith("/report"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) > 1:
                            report = system.generate_report(parts[1])
                            print(f"\nReport:\n{report}\n")
                        else:
                            print("Usage: /report <topic>")
                    elif user_input.startswith("/compare"):
                        parts = user_input.split()[1:]
                        if len(parts) >= 2:
                            analysis = system.comparative_analysis(parts)
                            print(f"\nAnalysis:\n{analysis}\n")
                        else:
                            print("Usage: /compare <item1> <item2> ...")
                    elif user_input.startswith("/extract"):
                        parts = user_input.split()
                        if len(parts) >= 3:
                            result = system.extract_specific_info(parts[1], parts[2])
                            print(f"\nExtracted Information:\n{json.dumps(result, indent=2)}\n")
                        else:
                            print("Usage: /extract <entity> <info_type>")
                    elif user_input == "/save":
                        filename = system.save_conversation()
                        print(f"Conversation saved to: {filename}")
                    else:
                        # Regular query
                        result = system.query(user_input, use_conversation=True)
                        print(f"\nAssistant: {result['answer']}\n")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            print("\nGoodbye!")
        else:
            # Single query mode
            query = input("Enter your query: ")
            result = system.query(query)
            print(f"\nAnswer: {result['answer']}")
            
            if "relevant_documents" in result:
                print("\nSources:")
                for doc in result["relevant_documents"]:
                    print(f"- {doc['metadata'].get('filename', 'Unknown')}")
                    
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease make sure you have:")
        print("1. Installed required packages:")
        print("   pip install langchain-anthropic langchain-community langchain-huggingface")
        print("2. Set your Anthropic API key:")
        print("   export ANTHROPIC_API_KEY='your-api-key'")