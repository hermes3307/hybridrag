#!/usr/bin/env python3
"""
ChromaDB Installation and Management Script
Installs ChromaDB, initializes collections, and provides detailed database information
"""

import subprocess
import sys
import os
import json
from typing import Dict, List, Any, Optional
import time

def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    return result

def install_chromadb():
    """Install ChromaDB and required dependencies"""
    print("🔧 Installing ChromaDB and dependencies...")

    # Install ChromaDB
    run_command("pip install chromadb")

    # Install additional useful packages
    run_command("pip install numpy sentence-transformers")

    print("✅ ChromaDB installation completed!")

def check_chromadb_installation():
    """Verify ChromaDB is properly installed"""
    print("🔍 Checking ChromaDB installation...")

    try:
        import chromadb
        print(f"✅ ChromaDB version: {chromadb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ ChromaDB not found: {e}")
        return False

def initialize_chromadb():
    """Initialize ChromaDB client and create sample collections"""
    print("🚀 Initializing ChromaDB...")

    try:
        import chromadb
        from chromadb.config import Settings

        # Create persistent client
        client = chromadb.PersistentClient(path="./chroma_db")

        print("✅ ChromaDB client initialized successfully!")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize ChromaDB: {e}")
        return None

def get_collection_info(client, collection_name: str) -> Dict[str, Any]:
    """Get detailed information about a collection"""
    try:
        collection = client.get_collection(collection_name)

        # Get collection metadata
        metadata = collection.metadata or {}

        # Get collection count
        count = collection.count()

        # Try to peek at some data to determine dimensions
        dimensions = None
        sample_data = None

        if count > 0:
            try:
                # Get a small sample to inspect
                results = collection.peek(limit=1)
                if results['embeddings'] and len(results['embeddings']) > 0:
                    dimensions = len(results['embeddings'][0])
                    sample_data = {
                        'ids': results['ids'][:3] if results['ids'] else [],
                        'metadata_sample': results['metadatas'][:3] if results['metadatas'] else [],
                        'documents_sample': results['documents'][:3] if results['documents'] else []
                    }
            except Exception as e:
                print(f"Warning: Could not peek into collection {collection_name}: {e}")

        return {
            'name': collection_name,
            'count': count,
            'metadata': metadata,
            'dimensions': dimensions,
            'sample_data': sample_data
        }
    except Exception as e:
        return {
            'name': collection_name,
            'error': str(e)
        }

def create_sample_collection(client):
    """Create a sample collection with some test data"""
    print("📊 Creating sample collection...")

    try:
        import numpy as np

        # Create or get collection
        collection = client.get_or_create_collection(
            name="sample_collection",
            metadata={"description": "Sample collection for testing"}
        )

        # Add some sample data
        sample_embeddings = np.random.rand(5, 384).tolist()  # 384-dimensional embeddings
        sample_documents = [
            "This is a sample document about machine learning",
            "ChromaDB is a vector database for AI applications",
            "Vector embeddings represent text as numerical vectors",
            "Similarity search finds related documents",
            "Semantic search uses meaning rather than keywords"
        ]
        sample_ids = [f"doc_{i}" for i in range(5)]
        sample_metadata = [
            {"topic": "ml", "length": len(doc)} for doc in sample_documents
        ]

        collection.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadata,
            ids=sample_ids
        )

        print("✅ Sample collection created with test data!")
        return True

    except Exception as e:
        print(f"❌ Failed to create sample collection: {e}")
        return False

def display_database_info(client):
    """Display comprehensive database information"""
    print("\n" + "="*80)
    print("📊 CHROMADB DATABASE INFORMATION")
    print("="*80)

    try:
        # Get all collections
        collections = client.list_collections()

        print(f"\n📁 Total Collections: {len(collections)}")

        if not collections:
            print("   No collections found. Database is empty.")
            return

        total_documents = 0

        for collection in collections:
            print(f"\n🗂️  Collection: {collection.name}")
            print("-" * 50)

            info = get_collection_info(client, collection.name)

            if 'error' in info:
                print(f"   ❌ Error: {info['error']}")
                continue

            print(f"   📄 Document Count: {info['count']:,}")
            total_documents += info['count']

            if info['dimensions']:
                print(f"   📐 Vector Dimensions: {info['dimensions']}")

            if info['metadata']:
                print(f"   🏷️  Collection Metadata:")
                for key, value in info['metadata'].items():
                    print(f"      • {key}: {value}")

            if info['sample_data']:
                print(f"   🔍 Sample Data:")
                sample = info['sample_data']

                if sample['ids']:
                    print(f"      • Sample IDs: {sample['ids']}")

                if sample['documents_sample']:
                    print(f"      • Sample Documents:")
                    for doc in sample['documents_sample']:
                        preview = doc[:100] + "..." if len(doc) > 100 else doc
                        print(f"        - {preview}")

                if sample['metadata_sample']:
                    print(f"      • Sample Metadata: {sample['metadata_sample']}")

        print(f"\n📊 SUMMARY")
        print("-" * 30)
        print(f"Total Collections: {len(collections)}")
        print(f"Total Documents: {total_documents:,}")

        # Database size estimation
        db_path = "./chroma_db"
        if os.path.exists(db_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)

            size_mb = total_size / (1024 * 1024)
            print(f"Database Size: {size_mb:.2f} MB ({total_size:,} bytes)")

    except Exception as e:
        print(f"❌ Error displaying database info: {e}")

def interactive_menu(client):
    """Interactive menu for database operations"""
    while True:
        print("\n" + "="*50)
        print("🔧 CHROMADB INTERACTIVE MENU")
        print("="*50)
        print("1. 📊 Show Database Information")
        print("2. 🔍 List All Collections")
        print("3. 📁 Create New Collection")
        print("4. 🗑️  Delete Collection")
        print("5. 🔄 Refresh Database Info")
        print("6. 🚪 Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            display_database_info(client)
        elif choice == "2":
            collections = client.list_collections()
            print(f"\n📁 Collections ({len(collections)}):")
            for col in collections:
                print(f"   • {col.name}")
        elif choice == "3":
            name = input("Enter collection name: ").strip()
            if name:
                try:
                    collection = client.create_collection(name)
                    print(f"✅ Collection '{name}' created successfully!")
                except Exception as e:
                    print(f"❌ Error creating collection: {e}")
        elif choice == "4":
            collections = client.list_collections()
            if not collections:
                print("No collections to delete.")
                continue

            print("Available collections:")
            for i, col in enumerate(collections, 1):
                print(f"   {i}. {col.name}")

            try:
                idx = int(input("Enter collection number to delete: ")) - 1
                if 0 <= idx < len(collections):
                    col_name = collections[idx].name
                    confirm = input(f"Delete '{col_name}'? (y/N): ").strip().lower()
                    if confirm == 'y':
                        client.delete_collection(col_name)
                        print(f"✅ Collection '{col_name}' deleted!")
            except (ValueError, IndexError):
                print("Invalid selection.")
        elif choice == "5":
            print("🔄 Refreshing...")
            display_database_info(client)
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main function to run the ChromaDB setup and management"""
    print("🚀 ChromaDB Setup and Management Script")
    print("="*80)

    # Check if ChromaDB is installed
    if not check_chromadb_installation():
        install_chromadb()
        if not check_chromadb_installation():
            print("❌ Installation failed. Exiting.")
            sys.exit(1)

    # Initialize ChromaDB
    client = initialize_chromadb()
    if not client:
        sys.exit(1)

    # Create sample collection if none exist
    collections = client.list_collections()
    if not collections:
        print("📝 No collections found. Creating sample collection...")
        create_sample_collection(client)

    # Display initial database info
    display_database_info(client)

    # Start interactive menu
    try:
        interactive_menu(client)
    except KeyboardInterrupt:
        print("\n\n👋 Script interrupted. Goodbye!")

if __name__ == "__main__":
    main()