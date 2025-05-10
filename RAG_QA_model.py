
"""
This file is the RAG_QA_model for the assignment 2.
You can provide the question and the context and the model.

The RAG_QA_model will answer the question based on the context.

"""

from llama_cpp import Llama
from typing import List, Dict, Any
from search import *
from tqdm import tqdm
import json
from param import *

def setup_llama_model() -> Llama:
    """
    Initialize the quantized TinyLlama model
    
    Returns:
        Llama: Initialized model instance
    """
    model = Llama(
        model_path=LLM["model_path"],  # Download this file
        n_ctx=LLM["n_ctx"],        # Context window size
        n_threads=4,       # Adjust based on CPU
        n_gpu_layers=0,    # Set to use GPU if available
        verbose=False     
    )
    return model

def get_model_response(
    model: Llama, 
    query: str, 
    context: str, 
    temperature: float = LLM["temperature"], 
    max_tokens: int = LLM["max_tokens"],
    top_p: float = LLM["top_p"]
) -> str:
    """
    Get response from the model
    
    Args:
        model: Initialized Llama model
        query: User question
        context: Retrieved context from vector database
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        str: Model's response
    """
    prompt = f"""You are an intelligent assistant specializing in answering questions about Pittsburgh and Carnegie Mellon University (CMU).
    You use a retrieval system to access highly relevant documents related to Pittsburgh and CMU. Based on the information retrieved, generate a clear, concise, and factual response.

### Instructions:
1. Carefully read the retrieved context provided below.
2. Use the information in the retrieved context and your own knowledge to answer the question accurately. Do not add any information that is not present in the retrieved context.
3. Your response should be concise, factual, and directly address the question.

---
### Example Input:
**User Question:** Where is Carnegie Mellon University located?

**Retrieved Context:**
1. Carnegie Mellon University is located in Pittsburgh, Pennsylvania.
2. The campus is situated in the Oakland neighborhood of Pittsburgh.

**Your Answer:**
Carnegie Mellon University is located in Pittsburgh, Pennsylvania, specifically in the Oakland neighborhood.

---
### Now, answer the following question:**
**User Question:** {query}

**Retrieved Context:**
{context}

**Your Answer:**"""
    
    response = model.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["Question:", "\n\n"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

def qa_with_context(model: Llama, query: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Perform question answering using search results
    
    Args:
        query: User question
        search_results: Results from vector database search
        
    Returns:
        str: Model's answer
    """
    # Initialize model
    
    # Prepare context from search results
    context = "\n".join([result['content'] for result in search_results])
    
    try:
        answer = get_model_response(model, query, context)
        return answer
    except Exception as e:
        print(f"Error during QA: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

# Integration with your search function
def main():
    """
    Main function demonstrating the QA pipeline
    """
    # Initialize vector database
    db = initialize_vector_db()
    model = setup_llama_model()
    
    # Example query
    # read question from file
    with open(PATHS["question"], "r") as f:
        querys = f.readlines()
    
    # querys = querys[:10]
    # remove change line character
    querys = [query.strip() for query in querys if query.strip()]

    answers = {}

    # remove empty lines
    
    index = 0
    for query in tqdm(querys):
        index += 1
    
        # Get search results
        search_results = search_documents(
            query=query,
            db=db,
            top_k=2,  # Adjust based on needs
            score_threshold=0.5
        )
    
        # Get answer
        answer = qa_with_context(model, query, search_results)
        answers[str(index)] = answer
        # Print results
        # print(f"Question: {query}")
        # print(f"Answer: {answer}")
        # print("\nSearch Results Used:")
        # print(format_search_results(search_results))
    
    # write answers to json format with question index and answer
    with open(PATHS["generated_answer"], "w") as f:  
        json.dump(answers, f, indent=2, separators=(',\n', ': '), )


if __name__ == "__main__":
    main()