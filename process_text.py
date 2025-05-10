"""
This file is the process_text for the assignment 2.
You can provide the text_dir and the chunk_size and the chunk_overlap.

The process_text will load the text data from the text_dir and split it into chunks.
Then it will use the embedding model to get the embeddings of the chunks.
It will save the embeddings to the embeddings.pkl file.
And it will create a vector database with the chunks and the embeddings.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict
import time

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from param import *
import json
from langchain_community.vectorstores import Chroma
import shutil
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

def load_texts(text_dir: str = CRAWLER_CONFIG["text_dir"]) -> List[Document]:
    """
    Load text documents from a directory
    
    Args:
        data_dir: Directory containing text files
        
    Returns:
        List of Document objects that contain the text and metadata from the text files
    """
    print(f"Loading documents from {text_dir}...")
    
    # Get all text files
    files = list(Path(text_dir).glob("*.txt"))
    
    # 如果测试模式启用，限制文件数量
    if TEST_MODE["enabled"]:
        print(f"TEST MODE: Limiting to {TEST_MODE['max_embedding_files']} files")
        files = files[:TEST_MODE["max_embedding_files"]]
        # I want to save this files to a new folder
        os.makedirs("test_files", exist_ok=True)
        for file in files:
            shutil.copy(file, "test_files")
    
    print(f"Found {len(files)} text files")

    documents = []
    # Load documents
    for file_path in files:
        try:
            # Load text content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create document object
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name
                }
            )
            
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents: List[Document], chunk_size: int = TEXT_SPLITTING["chunk_size"], chunk_overlap: int = TEXT_SPLITTING["chunk_overlap"]) -> List[Document]:
    """
    Split documents into smaller chunks
    
    Args:
        documents: List of documents to split   
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        List of document chunks
    """
    print("Splitting documents into chunks...")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=TEXT_SPLITTING["separators"]   
    )

    chunks = text_splitter.split_documents(documents)
    
    # Add chunk ID and position information to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i)
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def get_embeddings(chunks, embedding_model):
    """
    Generate embeddings for all chunks
    
    Args:
        chunks: List of document chunks
        embedding_model: Embedding model to use
        
    Returns:
        Tuple of (unique_chunks, embeddings_dict) where embeddings_dict maps chunk_id to embedding
    """
    
    # Generate embeddings for all chunks
    texts = [chunk.page_content for chunk in chunks]
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    total_batches = (len(texts) + batch_size - 1) // batch_size
    all_embeddings = []
    
    with tqdm(total=total_batches, desc="Generating embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            pbar.update(1)
    
    embeddings = np.array(all_embeddings)
    # store embeddings
    with open(EMBEDDING["output_file"], "wb") as f:
        pickle.dump(embeddings, f)

    
    return chunks, embeddings


def create_db_with_embeddings(documents, embeddings, embedding_model):
    """
    Create a vector database using pre-computed embeddings
    
    Args:
        documents: List of Document objects to store
        embeddings_dict: Dictionary mapping chunk_id to embedding
        embedding_model: Embedding model (used for configuration only)
        
    Returns:
        Chroma: The created vector database
    """

    if len(documents) != len(embeddings):
        raise ValueError(f"Number of documents ({len(documents)}) does not match number of embeddings ({len(embeddings)})")
    
    total_chunks = len(documents)
    print(f"Creating vector database with {total_chunks} chunks using pre-computed embeddings...")
    
    # Create directory if it doesn't exist          
    if not os.path.exists(VECTOR_DB["persist_directory"]):
        os.makedirs(VECTOR_DB["persist_directory"])
    
    # Record start time
    start_time = time.time()
    
    # Initialize Chroma client
    import chromadb
    client = chromadb.PersistentClient(path=VECTOR_DB["persist_directory"])
    
    # Delete collection if it exists
    try:
        client.delete_collection(VECTOR_DB["collection_name"])
    except:
        pass
    
    # Create collection
    collection = client.create_collection(
        name=VECTOR_DB["collection_name"],
        metadata={"hnsw:space": VECTOR_DB.get("distance_metric", "cosine")}
    )
    
    # Process in batches
    batch_size = 1000
    
    # Prepare data for insertion
    ids = [doc.metadata['chunk_id'] for doc in documents]
    embeddings = embeddings
    metadatas = [doc.metadata for doc in documents]
    texts = [doc.page_content for doc in documents]
    
    # Add in batches 
    with tqdm(total=len(ids), desc="Adding to database", unit="chunk") as pbar:
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_texts = texts[i:end_idx]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )
            
            pbar.update(end_idx - i)
    
    # Create LangChain wrapper
    db = Chroma(
        client=client,
        collection_name=VECTOR_DB["collection_name"],
        embedding_function=embedding_model
    )
    
    # Calculate time taken
    store_time = time.time() - start_time
    print(f"Vector database created in {store_time:.2f} seconds")
    
    # Save metadata
    metadata = {
        "chunk_count": total_chunks,
        "embedding_model": EMBEDDING["model_name"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_seconds": round(store_time, 2),
        "deduplication": True
    }
    
    metadata_path = os.path.join(VECTOR_DB["persist_directory"], "db_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    
    return db


def main(): 
    # Load documents
    documents = load_texts()

    # Split documents
    chunks = split_documents(documents)
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_TRAINING["model_save_path"],
        model_kwargs={'device': EMBEDDING["device"]},
        encode_kwargs={'normalize_embeddings': EMBEDDING["normalize_embeddings"]}
    )
    
    # Remove similar chunks and get embeddings
    # embeddings = get_embeddings(chunks, embedding_model)

    with open("document_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    # Create vector database with pre-computed embeddings
    db = create_db_with_embeddings(chunks, embeddings, embedding_model)

    print("Vector database created successfully!")

if __name__ == "__main__":
    main()