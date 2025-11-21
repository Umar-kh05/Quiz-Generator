"""
Script to create a sample dataset for Quiz Generator
This creates a sample CSV file with the required format
"""
import pandas as pd
import os

def create_sample_dataset(output_path: str = "./data/quiz_dataset.csv", num_samples: int = 500):
    """Create a sample dataset for training"""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample data
    subjects = ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology"]
    topics = {
        "Computer Science": ["Machine Learning", "Data Structures", "Algorithms", "Database Systems", "Operating Systems"],
        "Mathematics": ["Calculus", "Linear Algebra", "Probability", "Statistics", "Discrete Mathematics"],
        "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Quantum Physics", "Optics"],
        "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Biochemistry", "Analytical Chemistry"],
        "Biology": ["Cell Biology", "Genetics", "Ecology", "Anatomy", "Biochemistry"]
    }
    
    difficulties = ["easy", "medium", "hard"]
    question_types = ["MCQ", "Short Answer", "Long Answer"]
    exam_types = ["Midterm", "Final", "Quiz"]
    years = [2020, 2021, 2022, 2023, 2024]
    
    # Generate sample questions
    data = []
    
    for i in range(num_samples):
        subject = subjects[i % len(subjects)]
        topic_list = topics[subject]
        topic = topic_list[i % len(topic_list)]
        difficulty = difficulties[i % len(difficulties)]
        question_type = question_types[i % len(question_types)]
        exam_type = exam_types[i % len(exam_types)]
        year = years[i % len(years)]
        
        # Create sample question based on parameters
        if question_type == "MCQ":
            if difficulty == "easy":
                question = f"What is the basic concept of {topic} in {subject}? A) Option A B) Option B C) Option C D) Option D"
            elif difficulty == "medium":
                question = f"Explain the relationship between {topic} and its applications in {subject}. A) Option A B) Option B C) Option C D) Option D"
            else:  # hard
                question = f"Analyze the complex interactions in {topic} and their implications for {subject} systems. A) Option A B) Option B C) Option C D) Option D"
        elif question_type == "Short Answer":
            if difficulty == "easy":
                question = f"Define {topic} in the context of {subject}."
            elif difficulty == "medium":
                question = f"Describe the key principles of {topic} and how they apply to {subject}."
            else:  # hard
                question = f"Critically evaluate the importance of {topic} in advancing {subject} research and applications."
        else:  # Long Answer
            if difficulty == "easy":
                question = f"Explain in detail what {topic} is and why it is important in {subject}. Provide examples to support your answer."
            elif difficulty == "medium":
                question = f"Discuss the theoretical foundations of {topic} in {subject}. Include the main concepts, applications, and limitations."
            else:  # hard
                question = f"Provide a comprehensive analysis of {topic} including its theoretical background, practical applications, current research trends, and future directions in the field of {subject}."
        
        data.append({
            'id': i + 1,
            'subject': subject,
            'topic': topic,
            'year': year,
            'exam_type': exam_type,
            'question_type': question_type,
            'difficulty': difficulty,
            'question': question
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created with {num_samples} samples")
    print(f"Saved to: {output_path}")
    print(f"\nDataset preview:")
    print(df.head(10))
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nDataset statistics:")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Topics: {df['topic'].nunique()}")
    print(f"Difficulties: {df['difficulty'].value_counts().to_dict()}")
    print(f"Question Types: {df['question_type'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample dataset for Quiz Generator")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/quiz_dataset.csv",
        help="Output path for the dataset CSV file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output_path, args.num_samples)


