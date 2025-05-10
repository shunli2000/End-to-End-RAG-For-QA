"""
Search documents in the vector database
Provides semantic similarity-based query functionality
"""

import os
import argparse
import time
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from param import EMBEDDING, VECTOR_DB, SEARCH

def initialize_vector_db(persist_directory: str = VECTOR_DB["persist_directory"]):
    """
    Initialize vector database connection
    
    Args:
        persist_directory: Directory where the vector database is stored
        
    Returns:
        Initialized Chroma vector database instance
    """
    print(f"Connecting to vector database: {persist_directory}")
    
    # Check if vector database directory exists
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector database directory does not exist: {persist_directory}")
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING["model_name"],
        model_kwargs={'device': EMBEDDING["device"]},
        encode_kwargs={'normalize_embeddings': EMBEDDING["normalize_embeddings"]}
    )
    
    # Connect to existing vector database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=VECTOR_DB["collection_name"]
    )
    
    # Get database info
    collection = db._collection
    count = collection.count()
    print(f"Successfully connected to vector database with {count} documents")
    
    return db

def search_documents(
    query: str,
    db: Chroma = None,
    top_k: int = SEARCH["top_k"],
    score_threshold: float = SEARCH["score_threshold"]
) -> List[Dict[str, Any]]:
    """
    Search for documents semantically related to the query in the vector database
    
    Args:
        query: User query
        db: Vector database instance, initialized automatically if None
        top_k: Maximum number of results to return
        score_threshold: Similarity score threshold, results below this will be filtered out
        
    Returns:
        List of results containing relevant documents and metadata
    """
    # Initialize database if not provided
    if db is None:
        db = initialize_vector_db()
    
    # print(f"Searching for: '{query}'")
    start_time = time.time()
    
    # Perform similarity search
    results = db.similarity_search_with_relevance_scores(
        query,
        k=top_k
    )
    
    # Process results
    processed_results = []
    for doc, score in results:
        # Skip if score is below threshold
        if score < score_threshold:
            continue
            
        # Extract document content and metadata
        processed_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    
    search_time = time.time() - start_time
    # print(f"Search completed in {search_time:.2f} seconds, found {len(processed_results)} relevant results")
    
    return processed_results

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for display
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string representation of results
    """
    if not results:
        return "No relevant results found."
    
    formatted_output = []
    for i, result in enumerate(results, 1):
        # Format each result
        formatted_result = f"Result {i} (Score: {result['score']:.4f})\n"
        formatted_result += f"Source: {result['metadata'].get('filename', 'Unknown')}\n"
        formatted_result += f"Content: {result['content'][:500]}...\n"
        formatted_result += "-" * 50
        
        formatted_output.append(formatted_result)
    
    return "\n".join(formatted_output)

def main():
    """
    Main function to run the search functionality from command line
    """
    
    # Initialize database
    db = initialize_vector_db()
    
    # Perform search
    query = "What is Pittsburgh's nickname that refers to its bridges?"
    results = search_documents(
        query=query,
        db=db,
        top_k=SEARCH["top_k"],
        score_threshold=SEARCH["score_threshold"]
    )
    # # Display results
    print("\nSearch Results:")
    print(format_search_results(results))

if __name__ == "__main__":
    main()
