"""
Configuration parameters for the RAG system.
This file centralizes all parameters used across different components.
"""

# Crawler parameters
CRAWLER_CONFIG = {
    "max_depth": 4,
    "max_pages_per_url": 2000,
    "allowed_domains": [
        'wikipedia.org', 
        'cmu.edu', 
        'pittsburghpa.gov',
        'britannica.com',
        'visitpittsburgh.com'
    ],
    "min_word_count": 100,
    "max_newline_ratio": 0.5,
    "text_dir": "text_data"
}


# Text splitting parameters
TEXT_SPLITTING = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", " ", ""]
}

# Embedding parameters
EMBEDDING = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",    # sentence-transformers/multi-qa-mpnet-base-dot-v1  require too much computing resource
    "device": "cpu",
    "normalize_embeddings": True,
    "batch_size": 32,
    "output_file": "document_embeddings.pkl",
    "metadata_file": "embedding_metadata.json"
}

# Vector database parameters
VECTOR_DB = {
    "persist_directory": "vector_db",
    "distance_metric": "cosine",  # Options: cosine, l2, ip
    "collection_name": "pittsburgh_data"  # Default collection name
}

# Search parameters
SEARCH = {
    "top_k": 3,
    "score_threshold": 0.5  # Minimum similarity score to consider
}

# LLM parameters
LLM = {
    "model_name": "TinyLlama-1.1B-Chat-v1.0",
    "model_path": "/Users/berniec/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "temperature": 0.7,
    "n_ctx": 2048,
    "max_tokens": 256,
    "top_p": 0.9
}

# RAG parameters
RAG = {
    "prompt_template": """Answer the question based on the context provided. 
Context: {context}

Question: {query}

Answer:"""
}

# File paths
PATHS = {
    "text_data": "text_data",
    "embeddings": "document_embeddings.pkl",
    "vector_db": "vector_db",
    "metadata": "vector_db/metadata.json",
    "question": "data/test/questions.txt",
    "answer": "data/test/reference_answers.json",
    "generated_answer": "data/test/generated_answers.json"
}

# Training parameters for embedding model
EMBEDDING_TRAINING = {
    "train_batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_seq_length": 512,
    "model_save_path": "trained_embeddings",
    "save_best_model": True,
}

'''
for the small training size, we can try the following parameters
EMBEDDING_TRAINING = {
    "train_batch_size": 16,     # 减小batch size
    "num_epochs": 20,           # 可以增加epoch数
    "learning_rate": 1e-5,      # 降低学习率
    "warmup_ratio": 0.2,        # 增加warmup比例
    "max_seq_length": 512,
    "model_save_path": "trained_embeddings",
    "save_best_model": True,
}
'''


# 测试模式参数
TEST_MODE = {
    "enabled": False,  # 设置为True启用测试模式，False禁用
    "max_embedding_files": 10,   # 测试模式下最多处理的文件数
} 





"""
embedding model:
    - sentence-transformers/multi-qa-mpnet-base-dot-v1:
        Parameters: 110M
        embedding size: 768
        max sequence length: 512

            




"""