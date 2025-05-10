from typing import Dict, Any, List
import re
import json
from param import *
def evaluate_qa(
    ground_truth: Dict[str, str],
    predictions: Dict[str, str],
    case_sensitive: bool = False,
    clean_spaces: bool = True
) -> Dict[str, Any]:
    """
    Evaluate predictions using precision, recall, accuracy, exact match, and F1 score
    
    Args:
        ground_truth: Dictionary of reference answers
        predictions: Dictionary of model-generated answers
        case_sensitive: Whether to consider case in token matching
        clean_spaces: Whether to normalize spaces
        
    Returns:
        Dict containing evaluation metrics including precision, recall, accuracy, exact match, and F1 score
    """
    if set(ground_truth.keys()) != set(predictions.keys()):
        raise ValueError("Ground truth and predictions must have the same question IDs")
    
    details = {
        "per_question_scores": {},
        "total": len(ground_truth)
    }
    
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    total_exact_match = 0
    total_f1 = 0
    
    for qid in ground_truth.keys():
        # Get answers
        truth = ground_truth[qid]
        pred = predictions[qid]
        
        # Preprocess
        if not case_sensitive:
            truth = truth.lower()
            pred = pred.lower()
        if clean_spaces:
            truth = " ".join(truth.split())
            pred = " ".join(pred.split())
        
        # Remove articles
        truth = re.sub(r'\b(a|an|the)\b', '', truth)
        pred = re.sub(r'\b(a|an|the)\b', '', pred)

        # Remove punctuations
        truth = re.sub(r'[^\w\s]', '', truth)
        pred = re.sub(r'[^\w\s]', '', pred)
        
        # Tokenize (split into words and remove punctuation)
        truth_tokens = set(re.findall(r'\w+', truth))
        pred_tokens = set(re.findall(r'\w+', pred))
        
        # Calculate metrics
        true_positives = len(truth_tokens & pred_tokens)
        false_positives = len(pred_tokens - truth_tokens)
        false_negatives = len(truth_tokens - pred_tokens)
        
        # Calculate precision, recall, and accuracy for this question
        precision = true_positives / len(pred_tokens) if pred_tokens else 0
        recall = true_positives / len(truth_tokens) if truth_tokens else 0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        
        # Calculate exact match
        exact_match = 1 if truth == pred else 0
        
        # Calculate F1 score
        if true_positives == 0:
            f1 = 0
        else:
            precision_f1 = true_positives / (true_positives + false_positives)
            recall_f1 = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision_f1 * recall_f1) / (precision_f1 + recall_f1)
        
        # Store individual scores
        details["per_question_scores"][qid] = {
            "ground_truth": truth,
            "prediction": pred,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "exact_match": exact_match,
            "f1": f1
        }
        
        # Accumulate totals
        total_precision += precision
        total_recall += recall
        total_accuracy += accuracy
        total_exact_match += exact_match
        total_f1 += f1
    
    # Calculate averages
    avg_precision = total_precision / len(ground_truth)
    avg_recall = total_recall / len(ground_truth)
    avg_accuracy = total_accuracy / len(ground_truth)
    avg_exact_match = total_exact_match / len(ground_truth)
    avg_f1 = total_f1 / len(ground_truth)
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "accuracy": avg_accuracy,
        "exact_match": avg_exact_match,
        "f1": avg_f1,
        "details": details
    }

def save_evaluation_results(
    evaluation_results: Dict[str, Any],
    output_file: str = "evaluation_results.json"
) -> None:
    """
    Save evaluation results along with model and parameter information
    
    Args:
        evaluation_results: Dictionary containing evaluation metrics
        output_file: Path to save the results
    """
    from param import EMBEDDING, LLM, VECTOR_DB, SEARCH, RAG
    import json
    import time
    
    # Get current timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Combine results with model information
    # do not store the details
    full_results = {
        "timestamp": timestamp,
        "metrics": {
            "precision": evaluation_results["precision"],
            "recall": evaluation_results["recall"],
            "accuracy": evaluation_results["accuracy"],
            "F1":evaluation_results["f1"],
            "exact_match":evaluation_results["exact_match"]
        },
        "embedding": {
            "model_name": EMBEDDING["model_name"],
            "device": EMBEDDING["device"],
            "normalize_embeddings": EMBEDDING["normalize_embeddings"]
        },
        "embedding_training": {
            "train_batch_size": EMBEDDING_TRAINING["train_batch_size"],
            "num_epochs": EMBEDDING_TRAINING["num_epochs"],
            "learning_rate": EMBEDDING_TRAINING["learning_rate"],
            "warmup_ratio": EMBEDDING_TRAINING["warmup_ratio"]
        },
        "models": {
            "embedding": {
                "model_name": EMBEDDING["model_name"],
                "device": EMBEDDING["device"],
                "normalize_embeddings": EMBEDDING["normalize_embeddings"]
            },
            "llm": {
                "model_name": LLM["model_name"],
                "temperature": LLM["temperature"],
                "max_tokens": LLM["max_tokens"]
            }
        },
        "parameters": {
            "vector_db": {
                "distance_metric": VECTOR_DB["distance_metric"],
                "collection_name": VECTOR_DB["collection_name"]
            },
            "search": {
                "top_k": SEARCH["top_k"],
                "score_threshold": SEARCH["score_threshold"]
            },
            "rag": {
                "prompt_template": RAG["prompt_template"]
            }
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"Evaluation results saved to {output_file}")
    
    return full_results

if __name__ == "__main__":
    # Load data
    with open(PATHS["answer"], "r") as f:
        ground_truth = json.load(f)
    with open(PATHS["generated_answer"], "r") as f:
        generated_answer = json.load(f)
    
    # Evaluate
    result = evaluate_qa(ground_truth, generated_answer)
    
    # Print overall results
    print("\nOverall Results:")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Exact Match: {result['exact_match']:.4f}")
    print(f"F1: {result['f1']:.4f}")
    
    # Save results with model information
    save_evaluation_results(result)