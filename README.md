# RAG-based Question Answering System

This project [11711-assignment-2](https://github.com/Bernie-cc/11711-assignment-2) implements a Retrieval-Augmented Generation (RAG) system for question answering. The system combines document retrieval with language model generation to provide accurate answers based on a specific knowledge base for CMU and Pittsburgh.

## Features

- finetune embedding model
- Efficient document retrieval using vector database
- Advanced language model for answer generation
- Context-aware question answering
- Customizable prompt

## System Architecture

### Core Components

1. **Document crawler** ([crawler.py](crawler.py))
   - Crawl the text data from the internet
   - Save the text data to the text_data directory
   - Save the metadata to the text_metadata.json file in the current directory
   - Save the visited urls to the visited_urls.json file

2. **Document processor** ([process_text.py](process_text.py))
   - Load the text data from the text_data directory
   - Split the text data into chunks
   - Save the chunks to the chunks.pkl file
   - Save the metadata to the chunks_metadata.json file

3. **Embedding model and vector database**
   - Load the chunks from the chunks.pkl file
   - Save the embeddings to the embeddings.pkl file
   - Save the metadata to the embedding_metadata.json file
   - Create a vector database with the chunks and the embeddings

4. **Retrieval Engine**
   - Vector similarity search
   - Context relevance ranking
   - Dynamic retrieval optimization

5. **Language Model Interface** ([RAG_QA_model.py](RAG_QA_model.py))
   - Query processing
   - Context integration
   - Answer generation

## Getting Started

### Prerequisites
- Python 3.8+
- See [requirements.txt](requirements.txt) for all dependencies

### Installation

1. Clone the repository

```bash
git clone https://github.com/Bernie-cc/11711-assignment-2.git
```

2. Install the required packages

```bash
pip install -r requirements.txt 
```

3. Run the crawler

You can run the crawler by providing the initial urls and the max depth and the max pages per url. It will save the text data to the text_data directory and the metadata to the text_metadata.json file in the current directory.

```bash
cd web_scrape
python crawler.py
python events.py
python sports.py
cd ..
```

4. Run the document processor

You can run the document processor by providing the text_dir and the chunk_size and the chunk_overlap. It will save the chunks to the chunks.pkl file and the metadata to the chunks_metadata.json file. It will embed the chunks and create a vector database with the chunks and the embeddings.

```bash
python process_text.py
```

5. Run the RAG_QA_model

You can run the RAG_QA_model by providing the question and the context. It will answer the question based on the context.

```bash
python RAG_QA_model.py
``` 

## Evaluation

You can evaluate the RAG_QA_model by providing the ground truth and the predictions. It will save the evaluation results to the [evaluation_results.json](evaluation_results.json) file.

```bash
python evaluate.py
``` 

## Results

You can find the results in the [system_output_1.json](system_output_1.json) file.

## Contributors


- [Shun Li](https://github.com/shunli2000) 
- [Youyou Huang](https://github.com/youyouh511) 
- [Zijin Cui](https://github.com/Bernie-cc) 

## Related Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)

## Configuration

All configuration parameters can be found in [param.py](param.py).


