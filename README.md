# Quiz Generator Model - T5 Fine-tuning

A quiz generation system that fine-tunes T5-small model to generate quiz questions based on subject, topic, and difficulty level.

## Project Overview

This project implements a quiz generation chatbot that creates quizzes from university past papers, midterms, and finals. The system uses a fine-tuned T5 transformer model to generate questions in different formats (MCQs, short answers, long questions) based on user-specified subject and difficulty level.

## Features

- Fine-tuned T5-small model for question generation
- Support for multiple question types (MCQ, Short Answer, Long Answer)
- Difficulty level control (easy, medium, hard)
- Comprehensive evaluation metrics (ROUGE, BLEU, Exact Match, Top-k Accuracy)
- Training and validation plots
- Easy-to-use inference interface

## Dataset Format

The dataset should be a CSV file with the following columns:

- `id`: Unique identifier
- `subject`: Subject name (e.g., "Computer Science")
- `topic`: Topic name (e.g., "Machine Learning")
- `year`: Year (e.g., 2023)
- `exam_type`: Type of exam (e.g., "Midterm", "Final")
- `question_type`: Type of question (e.g., "MCQ", "Short Answer", "Long Answer")
- `difficulty`: Difficulty level (e.g., "easy", "medium", "hard")
- `question`: The actual question text

Example:
```csv
id,subject,topic,year,exam_type,question_type,difficulty,question
1,Computer Science,Machine Learning,2023,Final,MCQ,medium,"What is supervised learning? A) Option A B) Option B C) Option C D) Option D"
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Quiz Generator"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for evaluation):
```python
import nltk
nltk.download('punkt')
```

## Quick Start

### 1. Create Sample Dataset (Optional)

If you don't have a dataset yet, you can create a sample one:

```bash
python create_sample_dataset.py --output_path ./data/quiz_dataset.csv --num_samples 500
```

### 2. Train the Model

Train the T5 model on your dataset:

```bash
python train.py
```

The training script will:
- Load and preprocess the dataset
- Fine-tune the T5-small model
- Save the model to `./results/t5-quiz-generator/`
- Generate training plots and metrics

### 3. Use the Web Interface (Recommended) 🎉

Launch the Streamlit web app:

```bash
streamlit run app.py
```

Or use the helper script:

```bash
python run_app.py
```

The web interface will open in your browser where you can:
- Select subject, topic, difficulty, and question type
- Generate single or multiple questions
- Adjust advanced generation parameters
- View formatted quiz questions

### 4. Generate Quiz Questions (Command Line)

Alternatively, generate questions using the command line:

```bash
python inference.py --model_path ./results/t5-quiz-generator --subject "Computer Science" --topic "Machine Learning" --difficulty medium --question_type MCQ --num_questions 5
```

Or use the Python API:

```python
from inference import QuizGenerator

# Load the model
generator = QuizGenerator("./results/t5-quiz-generator")

# Generate a single question
question = generator.generate_question(
    subject="Computer Science",
    topic="Machine Learning",
    difficulty="medium",
    question_type="MCQ"
)
print(question)

# Generate multiple questions
questions = generator.generate_multiple_questions(
    subject="Mathematics",
    topic="Calculus",
    difficulty="hard",
    question_type="Short Answer",
    num_questions=5
)
for q in questions:
    print(q)
```

### 5. Evaluate the Model

Evaluate the model on a test set:

```bash
python evaluate.py --model_path ./results/t5-quiz-generator --test_data ./data/test_dataset.csv --output_dir ./evaluation_results
```

## Configuration

You can modify training parameters in `config.py`:

```python
@dataclass
class ModelConfig:
    model_name: str = "t5-small"
    max_length: int = 512
    max_target_length: int = 128
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    learning_rate: float = 3e-4
    # ... more parameters
```

## Project Structure

```
Quiz Generator/
├── config.py                  # Configuration settings
├── data_loader.py             # Dataset loading and preprocessing
├── train.py                   # Training script
├── inference.py               # Inference/generation script
├── evaluate.py                # Evaluation script
├── create_sample_dataset.py   # Script to create sample dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Dataset directory
│   └── quiz_dataset.csv       # Training dataset
└── results/                   # Training outputs
    └── t5-quiz-generator/     # Fine-tuned model
```

## Evaluation Metrics

The model is evaluated using:

- **Exact Match (EM)**: Percentage of exactly matching predictions
- **Top-k Accuracy**: Word overlap-based accuracy
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L for n-gram overlap
- **BLEU Score**: Bilingual Evaluation Understudy score

## Training Outputs

After training, you'll find:

- `train_metrics.json`: Training metrics
- `test_metrics.json`: Test set metrics
- `training_history.png`: Training and validation loss plots
- `evaluation_metrics.png`: Evaluation metrics visualization (after evaluation)
- Fine-tuned model and tokenizer in the output directory

## Usage Examples

### Basic Question Generation

```python
from inference import QuizGenerator

generator = QuizGenerator("./results/t5-quiz-generator")

# Generate an easy MCQ
question = generator.generate_question(
    subject="Physics",
    topic="Mechanics",
    difficulty="easy",
    question_type="MCQ"
)
```

### Advanced Generation with Custom Parameters

```python
# Generate with custom sampling parameters
question = generator.generate_question(
    subject="Chemistry",
    topic="Organic Chemistry",
    difficulty="hard",
    question_type="Long Answer",
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_length=256
)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (recommended for training)

## Notes

- Training time depends on dataset size and hardware
- For best results, use a dataset with at least 500+ examples
- The model is fine-tuned from `t5-small` - you can change to `t5-base` or `t5-large` in config.py for better quality (requires more memory)
- Make sure your dataset follows the exact column format specified above

## License

[Your License Here]

## Contact

[Your Contact Information]


