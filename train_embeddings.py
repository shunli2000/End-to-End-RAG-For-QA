from typing import List
import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
import os
from tqdm import tqdm
from param import EMBEDDING, EMBEDDING_TRAINING
from torch.utils.data import DataLoader

def prepare_training_data(
    question_file: str = "data/train/questions.txt",
    answer_file: str = "data/train/reference_answers.json"
) -> List[InputExample]:
    """
    Prepare training data from questions and reference answers
    
    Args:
        question_file: File containing questions
        answer_file: JSON file containing reference answers
        
    Returns:
        List of InputExample for training
    """
    # Load questions
    with open(question_file, 'r') as f:
        questions = f.read().splitlines()
    
    questions = [question.strip() for question in questions if question.strip()]

    
    # Load reference answers
    with open(answer_file, 'r') as f:
        answers = json.load(f)
    
    print(f"Loaded {len(questions)} questions and {len(answers)} answers")
    
    training_examples = []
    
    # Create training pairs
    for i, query in enumerate(questions, 1):
        # Get positive example (correct answer)
        positive_doc = answers[str(i)]
        
        # Create positive pair
        training_examples.append(
            InputExample(
                texts=[query, positive_doc],
                label=1.0
            )
        )
        
        # Create negative pairs (using other answers as negative examples)
        negative_indices = random.sample(
            [j for j in range(1, len(questions) + 1) if j != i],
            k=1  # Number of negative examples per query
        )
        
        for neg_idx in negative_indices:
            training_examples.append(
                InputExample(
                    texts=[query, answers[str(neg_idx)]],
                    label=0.0
                )
            )
    
    print(f"Created {len(training_examples)} training examples")
    print(f"- Positive pairs: {len(questions)}")
    print(f"- Negative pairs: {len(training_examples) - len(questions)}")
    
    return training_examples

def train_embedding_model():
    """Train the embedding model using prepared training data"""
    # Load base model
    model = SentenceTransformer(EMBEDDING["model_name"])
    
    # Prepare training data
    train_examples = prepare_training_data()
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=EMBEDDING_TRAINING["train_batch_size"]
    )
    
    # Use cosine similarity loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Create output directory
    os.makedirs(EMBEDDING_TRAINING["model_save_path"], exist_ok=True)
    
    print("Starting model training...")

    # Initialize lists to store losses (移到这里)
    


    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EMBEDDING_TRAINING["num_epochs"],
        warmup_steps=int(len(train_dataloader) * EMBEDDING_TRAINING["warmup_ratio"]),
        output_path=EMBEDDING_TRAINING["model_save_path"],
        save_best_model=EMBEDDING_TRAINING["save_best_model"],
        show_progress_bar=True,    
    )
    
    print(f"Model trained and saved to {EMBEDDING_TRAINING['model_save_path']}")
    return model



if __name__ == "__main__":
    train_embedding_model() 
