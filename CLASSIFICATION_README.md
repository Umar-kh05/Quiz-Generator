# Classification Models Implementation

This module implements comprehensive classification models for quiz data classification tasks.

## Models Implemented

### Machine Learning Models (2)
1. **Random Forest** - Uses TF-IDF embeddings
2. **SVM (Support Vector Machine)** - Uses TF-IDF embeddings

### Deep Learning Models (2)
1. **LSTM (Long Short-Term Memory)** - Uses Word2Vec embeddings
2. **CNN (Convolutional Neural Network)** - Uses Word2Vec embeddings

### Transformer-based Model (1)
1. **BERT (DistilBERT)** - Uses BERT tokenization and embeddings

## Embedding Techniques

- **TF-IDF**: Term Frequency-Inverse Document Frequency for ML models
- **Word2Vec**: Word embeddings for LSTM and CNN models
- **BERT**: Pre-trained DistilBERT tokenizer and embeddings for transformer model

## Classification Targets

You can classify any of the following columns:
- `subject` - Classify quiz questions by subject
- `difficulty` - Classify by difficulty level (easy, medium, hard)
- `question_type` - Classify by question type (MCQ, short, long, etc.)

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**
- **Precision** (weighted and per-class)
- **Recall** (weighted and per-class)
- **F1-Score** (weighted and per-class)
- **AUC** (Area Under the ROC Curve)
- **Exact Match (EM)**
- **Top-k Accuracy** (Top-1, Top-3, Top-5)
- **Confusion Matrix**

## Usage

### Basic Usage

```python
from train_classification import ClassificationTrainer

# Initialize trainer
trainer = ClassificationTrainer(
    csv_path='quiz_data.csv',
    target_column='subject',  # or 'difficulty' or 'question_type'
    output_dir='./results/classification'
)

# Train all models
results = trainer.train_all(
    dl_epochs=20,
    transformer_epochs=10
)
```

### Command Line Usage

```bash
# Classify by subject
python train_classification.py --csv_path quiz_data.csv --target_column subject

# Classify by difficulty
python train_classification.py --csv_path quiz_data.csv --target_column difficulty

# Classify by question type
python train_classification.py --csv_path quiz_data.csv --target_column question_type

# Custom epochs
python train_classification.py --csv_path quiz_data.csv --target_column subject --dl_epochs 30 --transformer_epochs 15
```

#### What the Training Does:

When you run the training command, it performs the following steps:

1. **Data Loading & Preprocessing**:
   - Loads quiz data from the CSV file
   - Extracts the `question` column as input text
   - Extracts the specified `target_column` (subject/difficulty/question_type) as labels
   - Splits data into Train (80%), Validation (10%), and Test (10%) sets

2. **Embedding Preparation**:
   - **TF-IDF**: Creates TF-IDF vectors for ML models (Random Forest, SVM)
     - **What it learns from dataset**: 
       - Analyzes ALL question texts in the dataset
       - Learns which words/terms are most important (vocabulary of top 5000 words)
       - Calculates Term Frequency (TF): How often each word appears in each question
       - Calculates Inverse Document Frequency (IDF): How rare/common each word is across all questions
       - Creates a vocabulary dictionary and weight matrix
       - **Example**: If "derivative" appears often in Calculus questions but rarely in NLP questions, it gets a high weight for Calculus classification
     - **Output**: Each question becomes a 5000-dimensional vector of TF-IDF scores
   
   - **Word2Vec**: Trains Word2Vec embeddings on the text corpus for DL models (LSTM, CNN)
     - **What it learns from dataset**:
       - Trains a neural network on ALL question texts
       - Learns word meanings by predicting words from their context (surrounding words)
       - Creates 300-dimensional dense vectors for each word
       - Words with similar meanings get similar vectors (e.g., "function" and "equation" might be close)
       - **Training process**: For each word, it looks at surrounding words (window of 5 words) and learns to predict context
       - **Example**: After training, "tokenization" and "parsing" might have similar vectors because they appear in similar contexts in NLP questions
     - **Output**: Each word gets a 300-dimensional embedding vector; questions are represented as average of word vectors
   
   - **BERT**: Prepares BERT tokenization for the Transformer model
     - Uses pre-trained tokenizer (no training needed here)
     - Converts text to token IDs that BERT model understands

3. **Model Training** (in sequence):
   - **Random Forest**: Trains on TF-IDF features (fast, ~seconds to minutes)
   - **SVM**: Trains on TF-IDF features (fast, ~seconds to minutes)
   - **LSTM**: Trains neural network with Word2Vec embeddings (slower, ~minutes to hours)
   - **CNN**: Trains neural network with Word2Vec embeddings (slower, ~minutes to hours)
   - **BERT**: Fine-tunes pre-trained DistilBERT model (slowest, ~hours, GPU recommended)

4. **Evaluation**:
   - Evaluates each model on Train, Validation, and Test sets
   - Calculates all metrics: Accuracy, Precision, Recall, F1-Score, AUC, EM, Top-k Accuracy
   - Generates confusion matrices
   - Creates training history plots (for DL and Transformer models)

5. **Output Generation**:
   - Saves all metrics as JSON files
   - Saves confusion matrix images
   - Saves training history plots (accuracy & loss curves)
   - All results saved to `./results/classification/` directory

**Example**: When you run `--target_column subject`, the models learn to classify quiz questions into different subjects (e.g., NLP, Calculus, Physics) based on the question text.

### Example Script

```bash
python example_classification.py
```

## Output

After training, you'll find in the output directory:

1. **Training History Plots** (`*_training_history.png`)
   - Separate plots for accuracy and loss
   - Training and validation curves for each model

2. **Confusion Matrices** (`*_confusion_matrix.png`)
   - Visual confusion matrix for each model on test set

3. **Evaluation Metrics** (`*_metrics.json`)
   - JSON files with all metrics for train, validation, and test sets
   - Includes per-class metrics

## Model Details

### Random Forest
- Uses TF-IDF features (max_features=5000, ngram_range=(1,2))
- n_estimators=100, max_depth=20

### SVM
- Uses TF-IDF features
- Kernel: RBF, C=1.0

### LSTM
- Embedding dimension: 300
- Hidden dimension: 128
- Bidirectional, 2 layers
- Dropout: 0.3

### CNN
- Embedding dimension: 300
- Filter sizes: [3, 4, 5]
- Number of filters: 100 per size
- Dropout: 0.3

### BERT (DistilBERT)
- Model: distilbert-base-uncased
- Max sequence length: 512
- Dropout: 0.3
- Learning rate: 2e-5

## Requirements

All dependencies are in `requirements.txt`. Key packages:
- torch
- transformers
- scikit-learn
- gensim (for Word2Vec)
- matplotlib, seaborn (for plotting)

## Notes

- ML models train quickly but may have lower accuracy
- DL models (LSTM/CNN) require more training time
- Transformer model (BERT) provides best accuracy but slowest training
- All models use the same train/val/test split (80/10/10 by default)
- Training plots show separate curves for training and validation

## Performance Tips

1. For faster training, reduce epochs for DL models
2. For better accuracy, increase epochs and use GPU
3. BERT model benefits most from GPU acceleration
4. Adjust batch sizes based on available memory

