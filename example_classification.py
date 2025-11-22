"""
Example script to demonstrate classification model training
"""
from train_classification import ClassificationTrainer

def main():
    """Example usage of classification trainer"""
    
    # Initialize trainer
    # You can classify: 'subject', 'difficulty', or 'question_type'
    trainer = ClassificationTrainer(
        csv_path='quiz_data.csv',
        target_column='subject',  # Change to 'difficulty' or 'question_type' as needed
        output_dir='./results/classification'
    )
    
    # Train all models
    # This will train:
    # - 2 ML models: Random Forest, SVM (using TF-IDF)
    # - 2 DL models: LSTM, CNN (using Word2Vec)
    # - 1 Transformer: BERT (using DistilBERT)
    results = trainer.train_all(
        dl_epochs=20,              # Epochs for LSTM and CNN
        dl_batch_size=32,          # Batch size for LSTM and CNN
        dl_lr=0.001,               # Learning rate for LSTM and CNN
        transformer_epochs=10,     # Epochs for BERT
        transformer_batch_size=16, # Batch size for BERT
        transformer_lr=2e-5        # Learning rate for BERT
    )
    
    print("\nTraining completed!")
    print(f"Results saved to: {trainer.output_dir}")
    print("\nYou can find:")
    print("- Training/validation plots for each model")
    print("- Confusion matrices")
    print("- Evaluation metrics (JSON files)")
    print("- All metrics: Accuracy, Precision, Recall, F1-Score, AUC, EM, Top-k Accuracy")

if __name__ == "__main__":
    main()

