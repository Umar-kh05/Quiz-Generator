"""
Example usage script for Quiz Generator
This demonstrates how to use the fine-tuned model for generating quiz questions
"""
from inference import QuizGenerator
import os

def main():
    """Example usage of QuizGenerator"""
    
    # Path to fine-tuned model
    model_path = "./results/t5-quiz-generator"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using:")
        print("  python train.py")
        return
    
    # Initialize generator
    print("Loading model...")
    generator = QuizGenerator(model_path)
    
    print("\n" + "="*60)
    print("Quiz Generator - Example Usage")
    print("="*60 + "\n")
    
    # Example 1: Generate a single MCQ question
    print("Example 1: Generate a single MCQ question")
    print("-"*60)
    question = generator.generate_question(
        subject="Computer Science",
        topic="Machine Learning",
        difficulty="medium",
        question_type="MCQ"
    )
    print(f"Generated Question:\n{question}\n")
    
    # Example 2: Generate multiple questions
    print("\nExample 2: Generate multiple short answer questions")
    print("-"*60)
    questions = generator.generate_multiple_questions(
        subject="Mathematics",
        topic="Calculus",
        difficulty="hard",
        question_type="Short Answer",
        num_questions=3
    )
    for i, q in enumerate(questions, 1):
        print(f"Question {i}:\n{q}\n")
    
    # Example 3: Generate with custom parameters
    print("\nExample 3: Generate with custom temperature for more diversity")
    print("-"*60)
    question = generator.generate_question(
        subject="Physics",
        topic="Quantum Mechanics",
        difficulty="hard",
        question_type="Long Answer",
        temperature=0.9,
        top_p=0.95,
        max_length=256
    )
    print(f"Generated Question:\n{question}\n")
    
    # Example 4: Different subjects and difficulties
    print("\nExample 4: Different subjects and difficulties")
    print("-"*60)
    examples = [
        {"subject": "Biology", "topic": "Cell Biology", "difficulty": "easy", "question_type": "MCQ"},
        {"subject": "Chemistry", "topic": "Organic Chemistry", "difficulty": "medium", "question_type": "Short Answer"},
        {"subject": "Computer Science", "topic": "Data Structures", "difficulty": "hard", "question_type": "Long Answer"},
    ]
    
    for i, ex in enumerate(examples, 1):
        question = generator.generate_question(**ex)
        print(f"Example {i} ({ex['subject']} - {ex['topic']} - {ex['difficulty']}):")
        print(f"{question}\n")
    
    print("="*60)
    print("Examples completed!")

if __name__ == "__main__":
    main()


