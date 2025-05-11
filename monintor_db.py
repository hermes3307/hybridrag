#!/usr/bin/env python3
"""
RAG Database Monitor
Monitors and displays statistics for FAISS vector database and Neo4j graph database
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase


class RAGDatabaseMonitor:
    """Monitor RAG databases and display statistics"""
    
    def __init__(self,
                 vector_db_path: str = "./vector_database",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        
        self.vector_db_path = Path(vector_db_path)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize connections
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.driver = None
        self.vector_store = None
    
    def connect_neo4j(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            return False


    def load_vector_store(self):
        """Load FAISS vector store"""
        try:
            if (self.vector_db_path / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.vector_db_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Add this parameter
                )
                return True
            else:
                print(f"Vector store not found at {self.vector_db_path}")
                return False
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def get_processing_stats(self):
        """Get processing statistics from the processed_files.json"""
        processed_files_path = self.vector_db_path / "processed_files.json"
        
        if not processed_files_path.exists():
            return None
        
        try:
            with open(processed_files_path, 'r', encoding='utf-8') as f:
                processed_files = json.load(f)
            
            stats = {
                'total_processed': len(processed_files),
                'total_chunks': sum(file_info.get('chunks', 0) for file_info in processed_files.values()),
                'last_processed': None,
                'first_processed': None
            }
            
            # Find first and last processed times
            if processed_files:
                dates = [file_info.get('processed_date') for file_info in processed_files.values() if file_info.get('processed_date')]
                if dates:
                    dates.sort()
                    stats['first_processed'] = dates[0]
                    stats['last_processed'] = dates[-1]
            
            return stats
        except Exception as e:
            print(f"Error reading processed files: {e}")
            return None
    
    def get_vector_store_stats(self):
        """Get vector store statistics"""
        if not self.vector_store:
            return None
        
        try:
            # Get number of vectors
            num_vectors = self.vector_store.index.ntotal
            
            return {
                'num_vectors': num_vectors,
                'vector_dimension': self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else None
            }
        except Exception as e:
            print(f"Error getting vector store stats: {e}")
            return None
    
    def get_neo4j_stats(self):
        """Get Neo4j database statistics"""
        if not self.driver:
            return None
        
        stats = {}
        
        try:
            with self.driver.session() as session:
                # Document counts
                result = session.run("""
                    MATCH (d:Document) 
                    RETURN count(d) as total_docs
                """)
                stats['total_documents'] = result.single()['total_docs']
                
                # Documents by label
                result = session.run("""
                    MATCH (d:Document)
                    RETURN labels(d) as labels, count(d) as count
                    ORDER BY count DESC
                """)
                stats['documents_by_label'] = [
                    {'labels': record['labels'], 'count': record['count']}
                    for record in result
                ]
                
                # Entity counts
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN count(e) as total_entities
                """)
                stats['total_entities'] = result.single()['total_entities']
                
                # Entities by type
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type as type, count(e) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                stats['top_entity_types'] = [
                    {'type': record['type'], 'count': record['count']}
                    for record in result
                ]
                
                # Relationship counts
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as total_relationships
                """)
                stats['total_relationships'] = result.single()['total_relationships']
                
                # Relationships by type
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """)
                stats['relationships_by_type'] = [
                    {'type': record['type'], 'count': record['count']}
                    for record in result
                ]
                
                # Document-Entity relationships
                result = session.run("""
                    MATCH (d:Document)-[r:CONTAINS_ENTITY]->(e:Entity)
                    RETURN count(r) as doc_entity_rels
                """)
                stats['document_entity_relationships'] = result.single()['doc_entity_rels']
                
                # Average entities per document
                result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
                    WITH d, count(e) as entity_count
                    RETURN avg(entity_count) as avg_entities_per_doc
                """)
                stats['avg_entities_per_document'] = result.single()['avg_entities_per_doc']
                
                return stats
                
        except Exception as e:
            print(f"Error getting Neo4j stats: {e}")
            return None
    
    def display_stats(self):
        """Display all statistics"""
        print("\n" + "="*50)
        print("RAG DATABASE MONITORING REPORT")
        print("="*50)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Processing stats
        proc_stats = self.get_processing_stats()
        if proc_stats:
            print("\n📁 PROCESSING STATISTICS:")
            print(f"  Total files processed: {proc_stats['total_processed']}")
            print(f"  Total chunks created: {proc_stats['total_chunks']}")
            if proc_stats['first_processed']:
                print(f"  First processed: {proc_stats['first_processed']}")
            if proc_stats['last_processed']:
                print(f"  Last processed: {proc_stats['last_processed']}")
        else:
            print("\n❌ No processing statistics available")
        
        # Vector store stats
        print("\n🔍 VECTOR STORE (FAISS):")
        if self.load_vector_store():
            vector_stats = self.get_vector_store_stats()
            if vector_stats:
                print(f"  Total vectors: {vector_stats['num_vectors']}")
                if vector_stats['vector_dimension']:
                    print(f"  Vector dimension: {vector_stats['vector_dimension']}")
            else:
                print("  Unable to retrieve vector store statistics")
        else:
            print("  Vector store not available")
        
        # Neo4j stats
        print("\n🌐 GRAPH DATABASE (Neo4j):")
        if self.connect_neo4j():
            neo4j_stats = self.get_neo4j_stats()
            if neo4j_stats:
                print(f"  Total documents: {neo4j_stats['total_documents']}")
                print(f"  Total entities: {neo4j_stats['total_entities']}")
                print(f"  Total relationships: {neo4j_stats['total_relationships']}")
                
                if neo4j_stats['avg_entities_per_document']:
                    print(f"  Avg entities per document: {neo4j_stats['avg_entities_per_document']:.2f}")
                
                if neo4j_stats['documents_by_label']:
                    print("\n  Documents by Label:")
                    for item in neo4j_stats['documents_by_label']:
                        print(f"    {item['labels']}: {item['count']}")
                
                if neo4j_stats['top_entity_types']:
                    print("\n  Top Entity Types:")
                    for item in neo4j_stats['top_entity_types']:
                        print(f"    {item['type']}: {item['count']}")
                
                if neo4j_stats['relationships_by_type']:
                    print("\n  Relationships by Type:")
                    for item in neo4j_stats['relationships_by_type']:
                        print(f"    {item['type']}: {item['count']}")
            else:
                print("  Unable to retrieve Neo4j statistics")
        else:
            print("  Neo4j not available")
        
        print("\n" + "="*50)
    
    def check_health(self):
        """Check if databases are healthy"""
        health = {
            'vector_store': False,
            'neo4j': False,
            'overall': False
        }
        
        # Check vector store
        if self.load_vector_store():
            vector_stats = self.get_vector_store_stats()
            health['vector_store'] = vector_stats is not None and vector_stats.get('num_vectors', 0) > 0
        
        # Check Neo4j
        if self.connect_neo4j():
            neo4j_stats = self.get_neo4j_stats()
            health['neo4j'] = neo4j_stats is not None
        
        health['overall'] = health['vector_store'] and health['neo4j']
        
        return health
    
    def close(self):
        """Close connections"""
        if self.driver:
            self.driver.close()


def main():
    parser = argparse.ArgumentParser(description="Monitor RAG Database Status")
    parser.add_argument("--vector-db", default="./vector_database", help="Vector database path")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--health-check", action="store_true", help="Perform health check only")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor (updates every 30 seconds)")
    
    args = parser.parse_args()
    
    monitor = RAGDatabaseMonitor(
        vector_db_path=args.vector_db,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    try:
        if args.health_check:
            health = monitor.check_health()
            print("\nHEALTH CHECK:")
            print(f"  Vector Store: {'✓' if health['vector_store'] else '✗'}")
            print(f"  Neo4j: {'✓' if health['neo4j'] else '✗'}")
            print(f"  Overall: {'✓' if health['overall'] else '✗'}")
        else:
            if args.watch:
                import time
                print("Starting continuous monitoring (Ctrl+C to stop)...")
                while True:
                    monitor.display_stats()
                    time.sleep(30)
            else:
                monitor.display_stats()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()