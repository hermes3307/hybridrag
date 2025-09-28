# ChromaDB Setup and Management

This directory contains scripts to install, setup, and manage ChromaDB vector database.

## Files

- `setup_chroma.py` - Main installation and interactive management script
- `run_chroma_info.py` - Simple script to display database information

## Quick Start

### 1. Install and Setup ChromaDB

```bash
python3 setup_chroma.py
```

This script will:
- Install ChromaDB and dependencies (chromadb, numpy, sentence-transformers)
- Initialize a persistent ChromaDB instance in `./chroma_db/`
- Create a sample collection with test data
- Show detailed database information
- Launch an interactive menu for database management

### 2. View Database Information

```bash
python3 run_chroma_info.py
```

This will display:
- Collection count and names
- Document counts per collection
- Vector dimensions
- Sample data and metadata
- Database size information

## Features

### Database Information Displayed
- **Collections**: Names and metadata
- **Document Counts**: Total documents per collection
- **Vector Dimensions**: Embedding vector dimensions
- **Sample Data**: Preview of documents and metadata
- **Database Size**: Physical storage size

### Interactive Menu Options
1. Show Database Information
2. List All Collections
3. Create New Collection
4. Delete Collection
5. Refresh Database Info
6. Exit

## Database Location

ChromaDB data is stored in `./chroma_db/` directory (persistent storage).

## Dependencies

The scripts automatically install:
- `chromadb` - Vector database
- `numpy` - Numerical operations
- `sentence-transformers` - For embedding generation (optional)

## Example Output

```
📊 CHROMADB DATABASE INFORMATION
================================================================================

📁 Total Collections: 1

🗂️  Collection: sample_collection
--------------------------------------------------
   📄 Document Count: 5
   📐 Vector Dimensions: 384
   🏷️  Collection Metadata:
      • description: Sample collection for testing
   🔍 Sample Data:
      • Sample IDs: ['doc_0', 'doc_1', 'doc_2']
      • Sample Documents:
        - This is a sample document about machine learning
        - ChromaDB is a vector database for AI applications

📊 SUMMARY
------------------------------
Total Collections: 1
Total Documents: 5
Database Size: 16.19 MB (16,976,228 bytes)
```