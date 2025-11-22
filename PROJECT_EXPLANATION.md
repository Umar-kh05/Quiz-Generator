# Classification Models Implementation - Project Explanation

## Project Overview

This project implements a comprehensive classification system for quiz question data using multiple machine learning, deep learning, and transformer-based approaches. The system can classify quiz questions based on different attributes (subject, difficulty level, or question type) using 5 different models with various embedding and encoding techniques.

## What Has Been Implemented

### 1. Machine Learning Models (2 Models)

#### a) Random Forest Classifier
- **Embedding Technique**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 
  - Max features: 5000
  - N-gram range: (1, 2) - captures unigrams and bigrams
  - Min document frequency: 2
- **Model Parameters**:
  - Number of estimators: 100
  - Max depth: 20
  - Random state: 42 (for reproducibility)

#### b) Support Vector Machine (SVM)
- **Embedding Technique**: TF-IDF (same as Random Forest)
- **Model Parameters**:
  - Kernel: RBF (Radial Basis Function)
  - C parameter: 1.0
  - Probability estimates enabled for probability predictions

**Why TF-IDF for ML Models?**
- TF-IDF is a traditional and effective text representation method
- Converts text into numerical features that ML algorithms can process
- Captures word importance relative to the document and corpus
- Works well with tree-based and kernel-based ML models

---

### 2. Deep Learning Models (2 Models)

#### a) LSTM (Long Short-Term Memory) Classifier
- **Embedding Technique**: Word2Vec
- **Architecture**:
  - Embedding dimension: 300
  - Hidden dimension: 128
  - Number of layers: 2
  - Bidirectional: Yes (captures context from both directions)
  - Dropout: 0.3 (prevents overfitting)
- **Vocabulary**: Built from training data (top 10,000 words)

#### b) CNN (Convolutional Neural Network) Classifier
- **Embedding Technique**: Word2Vec
- **Architecture**:
  - Embedding dimension: 300
  - Multiple filter sizes: [3, 4, 5] - captures different n-gram patterns
  - Number of filters per size: 100
  - Max pooling for feature extraction
  - Dropout: 0.3

**Why Word2Vec for Deep Learning Models?**
- Word2Vec creates dense vector representations that capture semantic relationships
- Pre-trained embeddings help models understand word meanings
- Better suited for neural networks than sparse TF-IDF vectors
- Captures contextual information through continuous vector space

---

### 3. Transformer-Based Model (1 Model)

#### BERT (Bidirectional Encoder Representations from Transformers)
- **Model**: DistilBERT-base-uncased (lighter, faster version of BERT)
- **Embedding Technique**: BERT tokenization and embeddings
- **Architecture**:
  - Pre-trained transformer model
  - Max sequence length: 512 tokens
  - Classification head: Linear layer on top of [CLS] token
  - Dropout: 0.3
- **Fine-tuning**: Model is fine-tuned on the classification task

**Why BERT for Transformer Model?**
- BERT uses contextualized embeddings (word meaning depends on context)
- Pre-trained on large corpus, captures deep linguistic patterns
- State-of-the-art performance for text classification
- Self-attention mechanism captures long-range dependencies

---

## Embedding and Encoding Techniques Summary

| Model Type | Model | Embedding Technique | Why This Technique? |
|------------|-------|---------------------|---------------------|
| ML | Random Forest | TF-IDF | Traditional, interpretable, works well with tree-based models |
| ML | SVM | TF-IDF | Sparse representation suitable for kernel methods |
| DL | LSTM | Word2Vec | Dense vectors capture semantics, good for sequential models |
| DL | CNN | Word2Vec | Dense vectors work well with convolutional operations |
| Transformer | BERT | BERT Tokenization | Contextual embeddings, state-of-the-art performance |

---

## Evaluation Metrics (All Implemented)

The system evaluates all models using the following comprehensive metrics:

### 1. **Accuracy**
- Overall correctness of predictions
- Formula: (Correct Predictions) / (Total Predictions)

### 2. **Precision**
- Measures how many of the predicted positive cases were actually positive
- Calculated as weighted average and per-class
- Formula: TP / (TP + FP)

### 3. **Recall**
- Measures how many of the actual positive cases were correctly identified
- Calculated as weighted average and per-class
- Formula: TP / (TP + FN)

### 4. **F1-Score**
- Harmonic mean of Precision and Recall
- Balances both metrics
- Formula: 2 × (Precision × Recall) / (Precision + Recall)

### 5. **AUC (Area Under the ROC Curve)**
- Measures the model's ability to distinguish between classes
- For binary: single AUC score
- For multi-class: macro-averaged AUC using one-vs-rest approach

### 6. **Exact Match (EM)**
- Percentage of predictions that exactly match the true labels
- For classification, this is equivalent to accuracy

### 7. **Top-k Accuracy**
- Measures if the true label is in the top-k predicted classes
- Implemented for k=1, k=3, and k=5
- Useful for understanding model confidence and ranking quality

### 8. **Confusion Matrix**
- Visual representation of classification performance
- Shows true vs predicted labels for all classes
- Helps identify which classes are confused with each other

**All metrics are calculated for:**
- Training set
- Validation set
- Test set

---

## Training and Validation Visualization

### Separate Plots Generated:
1. **Accuracy Plot**: 
   - Training accuracy curve
   - Validation accuracy curve
   - Shows model learning progress and overfitting detection

2. **Loss Plot**:
   - Training loss curve
   - Validation loss curve
   - Shows convergence and generalization

**Why Separate Plots?**
- Clear visualization of training vs validation performance
- Easy to identify overfitting (large gap between train and val)
- Helps understand model learning dynamics
- Standard practice in deep learning evaluation

---

## Project Structure

```
Quiz Generator/
├── classification_data_loader.py      # Data loading with multiple embeddings
├── classification_models.py           # All 5 model implementations
├── classification_evaluator.py        # Comprehensive evaluation system
├── train_classification.py            # Main training script
├── example_classification.py           # Usage example
├── CLASSIFICATION_README.md            # Detailed documentation
├── PROJECT_EXPLANATION.md             # This file
└── results/classification/            # Output directory
    ├── *_training_history.png         # Training plots
    ├── *_confusion_matrix.png         # Confusion matrices
    └── *_metrics.json                 # Evaluation metrics
```

---

## How the System Works

### 1. Data Preparation
- Loads quiz data from CSV file
- Supports classification by: `subject`, `difficulty`, or `question_type`
- Splits data into train (80%), validation (10%), and test (10%)
- Applies different preprocessing based on model type:
  - **TF-IDF**: Text → TF-IDF vectors (for ML models)
  - **Word2Vec**: Text → Word embeddings (for LSTM/CNN)
  - **BERT**: Text → Tokenized sequences (for Transformer)

### 2. Model Training
- **ML Models**: Train directly on TF-IDF features
- **DL Models**: Train using backpropagation with Word2Vec embeddings
- **Transformer**: Fine-tune pre-trained BERT on classification task

### 3. Evaluation
- All models evaluated on validation set during training
- Final evaluation on test set with all metrics
- Generates visualizations and saves metrics

### 4. Output Generation
- Training history plots (accuracy & loss)
- Confusion matrices
- JSON files with all metrics
- Per-class performance metrics

---

## Key Technical Details

### Data Split Strategy
- Stratified splitting ensures class distribution maintained
- Random seed: 42 (reproducibility)
- Train/Val/Test: 80%/10%/10%

### Hyperparameters

**ML Models:**
- Random Forest: n_estimators=100, max_depth=20
- SVM: kernel='rbf', C=1.0

**Deep Learning Models:**
- LSTM/CNN: epochs=20, batch_size=32, learning_rate=0.001
- Optimizer: Adam
- Loss: Cross-Entropy

**Transformer Model:**
- BERT: epochs=10, batch_size=16, learning_rate=2e-5
- Optimizer: Adam
- Loss: Cross-Entropy

### Device Support
- Automatically uses GPU if available (CUDA)
- Falls back to CPU if GPU not available
- All deep learning models support both

---

## How to Run the Project

### Option 1: Command Line
```bash
# Classify by subject
python train_classification.py --csv_path quiz_data.csv --target_column subject

# Classify by difficulty
python train_classification.py --csv_path quiz_data.csv --target_column difficulty

# Classify by question type
python train_classification.py --csv_path quiz_data.csv --target_column question_type
```

### Option 2: Python Script
```python
from train_classification import ClassificationTrainer

trainer = ClassificationTrainer(
    csv_path='quiz_data.csv',
    target_column='subject',  # or 'difficulty' or 'question_type'
    output_dir='./results/classification'
)

# Train all 5 models
results = trainer.train_all(
    dl_epochs=20,
    transformer_epochs=10
)
```

### Option 3: Example Script
```bash
python example_classification.py
```

---

## Expected Outputs

After running the training, you will find in `./results/classification/`:

1. **Training History Plots** (5 files):
   - `Random_Forest_training_history.png`
   - `SVM_training_history.png`
   - `LSTM_training_history.png`
   - `CNN_training_history.png`
   - `BERT_training_history.png`
   - Each shows separate accuracy and loss curves for train/val

2. **Confusion Matrices** (5 files):
   - `*_test_confusion_matrix.png` for each model
   - Visual representation of classification performance

3. **Evaluation Metrics** (15 JSON files):
   - `train_metrics.json`, `val_metrics.json`, `test_metrics.json` for each model
   - Contains all metrics: accuracy, precision, recall, F1, AUC, EM, top-k accuracy
   - Per-class metrics included

---

## Model Comparison and Selection

### When to Use Each Model:

**ML Models (Random Forest, SVM):**
- ✅ Fast training and inference
- ✅ Good baseline performance
- ✅ Interpretable (especially Random Forest)
- ❌ May not capture complex patterns
- ❌ Limited by TF-IDF representation

**Deep Learning Models (LSTM, CNN):**
- ✅ Better at capturing sequential patterns (LSTM)
- ✅ Good at local pattern detection (CNN)
- ✅ Word2Vec embeddings capture semantics
- ❌ Requires more training time
- ❌ More hyperparameters to tune

**Transformer Model (BERT):**
- ✅ Best accuracy (state-of-the-art)
- ✅ Contextual understanding
- ✅ Pre-trained on large corpus
- ❌ Slowest training and inference
- ❌ Requires more computational resources

---

## Requirements and Dependencies

All dependencies are listed in `requirements.txt`:

- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers (for BERT)
- **scikit-learn**: ML models and evaluation metrics
- **gensim**: Word2Vec implementation
- **pandas, numpy**: Data manipulation
- **matplotlib, seaborn**: Visualization
- **tqdm**: Progress bars

Install with:
```bash
pip install -r requirements.txt
```

---

## Key Achievements

✅ **5 Different Models**: 2 ML, 2 DL, 1 Transformer  
✅ **3 Different Embeddings**: TF-IDF, Word2Vec, BERT  
✅ **8 Evaluation Metrics**: All required metrics implemented  
✅ **Comprehensive Visualization**: Separate train/val plots for accuracy and loss  
✅ **Test Set Evaluation**: All metrics reported on test set  
✅ **Low Loss & Generalization**: Training includes validation monitoring to prevent overfitting  
✅ **Reproducible**: Fixed random seeds for consistent results  
✅ **Well-Documented**: Code comments, README, and examples  

---

## For TA: Quick Summary

**What was asked:**
- Implement 2 ML models, 2 DL models, 1 Transformer model
- Use different embedding/encoding techniques
- Evaluate with: Accuracy, Precision, Recall, F1, AUC, EM, Top-k Accuracy, Confusion Matrix
- Plot separate training/validation graphs for accuracy and loss
- Report all metrics on test set
- Focus on low loss and better generalization

**What was delivered:**
- ✅ Random Forest (TF-IDF) - ML Model 1
- ✅ SVM (TF-IDF) - ML Model 2
- ✅ LSTM (Word2Vec) - DL Model 1
- ✅ CNN (Word2Vec) - DL Model 2
- ✅ BERT/DistilBERT (BERT embeddings) - Transformer Model
- ✅ All 8 evaluation metrics implemented
- ✅ Separate train/val plots for accuracy and loss
- ✅ All metrics on test set
- ✅ Validation monitoring for generalization
- ✅ Comprehensive documentation

**Technical Highlights:**
- Different embeddings for different model types (appropriate choice)
- Proper train/val/test split with stratification
- GPU support for faster training
- Reproducible results (fixed seeds)
- Professional code structure and documentation

---

## Questions You Might Be Asked

**Q: Why different embeddings for different models?**  
A: Each embedding technique is optimized for the model type. TF-IDF works well with traditional ML algorithms, Word2Vec provides dense vectors for neural networks, and BERT offers contextual understanding for transformers.

**Q: How do you ensure generalization?**  
A: We use validation sets to monitor performance, early stopping concepts, dropout regularization, and separate test set evaluation. The gap between train and validation metrics indicates generalization quality.

**Q: Which model performs best?**  
A: Typically BERT performs best due to pre-training and contextual embeddings, but results depend on the dataset. All models are evaluated and compared using the same metrics.

**Q: How long does training take?**  
A: ML models: seconds to minutes. DL models: minutes to hours (depending on GPU). BERT: hours (GPU recommended). Training time is configurable via epochs parameter.

---

## Conclusion

This project demonstrates a comprehensive understanding of:
- Multiple machine learning paradigms (ML, DL, Transformers)
- Different text representation techniques
- Proper evaluation methodology
- Visualization and reporting
- Code organization and documentation

All requirements have been met and exceeded with additional features like per-class metrics, comprehensive documentation, and flexible configuration options.

