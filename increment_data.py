import csv
import random

# Define course patterns
courses = {
    'pf': {'subject': 'Programming Fundamentals', 'last_id': 36, 'topics': ['Code - Arrays', 'Code - Functions', 'Code - Recursion', 'Code - Pointers', 'Code - Structures', 'Code - File Handling', 'Code - Dynamic Memory', 'Code - Classes', 'Code - Inheritance', 'Code - Polymorphism'], 'years': ['2020 -Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']},
    'nlp': {'subject': 'NLP', 'last_id': 1199, 'topics': ['Tokenization', 'Stemming', 'Lemmatization', 'POS Tagging', 'Named Entity Recognition', 'Sentiment Analysis', 'Word Embeddings', 'Language Models', 'Text Classification', 'Machine Translation'], 'years': ['2020-Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'MCQ', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']},
    'cal': {'subject': 'Calculus', 'last_id': 1399, 'topics': ['Limits', 'Continuity', 'Derivatives', 'Chain Rule', 'Product Rule', 'Quotient Rule', 'Implicit Differentiation', 'Related Rates', 'Optimization', 'Integration'], 'years': ['2020-Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'MCQ', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']},
    'app': {'subject': 'Applied Physics', 'last_id': 1599, 'topics': ['Mechanics', 'Kinematics', 'Dynamics', 'Work and Energy', 'Conservation of Momentum', 'Rotational Motion', 'Simple Harmonic Motion', 'Waves', 'Sound Waves', 'Electromagnetic Waves'], 'years': ['2020-Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'MCQ', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']},
    'ict': {'subject': 'ICT', 'last_id': 1799, 'topics': ['Computer Networks', 'OSI Model', 'TCP/IP', 'Routing', 'Switching', 'Network Security', 'Cryptography', 'Firewalls', 'VPN', 'Database Management'], 'years': ['2020-Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'MCQ', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']},
    'aut': {'subject': 'Automata', 'last_id': 1999, 'topics': ['Finite Automata', 'DFA', 'NFA', 'Regular Expressions', 'Regular Languages', 'Context-Free Grammars', 'CFG', 'Pushdown Automata', 'Context-Free Languages', 'Turing Machines'], 'years': ['2020-Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'MCQ', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']},
    'coa': {'subject': 'COAL', 'last_id': 2199, 'topics': ['Instruction Set Architecture', 'Assembly Language', 'MIPS Architecture', 'Data Representation', 'Number Systems', 'Binary Arithmetic', 'Processor Design', 'ALU', 'Control Unit', 'CPU Organization'], 'years': ['2020-Fall', '2021-Spring', '2022-Fall', '2023-Spring', '2024-Fall'], 'exam_types': ['Mid1', 'Mid2', 'Final', 'Assignment', 'Quiz'], 'question_types': ['short', 'long', 'MCQ', 'programming', 'output'], 'difficulties': ['easy', 'medium', 'hard']}
}

# Question templates based on type
question_templates = {
    'short': {
        'easy': 'Define {topic} in the context of {subject}. Provide a brief explanation.',
        'medium': 'Explain the key principles of {topic} and how they apply to {subject}. [5 Points]',
        'hard': 'Discuss the importance of {topic} in {subject} and provide examples of its practical applications. [10 Points]'
    },
    'long': {
        'easy': 'Write a detailed explanation of {topic} in {subject}. Include basic concepts and simple examples.',
        'medium': 'Provide a comprehensive explanation of {topic} in {subject}. Discuss its theoretical foundations, practical applications, and limitations. [15 Points]',
        'hard': 'Analyze {topic} in {subject} comprehensively. Include theoretical background, mathematical formulations (if applicable), real-world applications, current research trends, and future directions. Provide detailed examples. [20 Points]'
    },
    'MCQ': {
        'easy': 'Which technique/law/component is most suitable for {topic} in {subject}? A) Option A B) Option B C) Option C D) Option D',
        'medium': 'Which technique/law/component is most suitable for {topic} in {subject}? A) Option A B) Option B C) Option C D) Option D',
        'hard': 'Which technique/law/component is most suitable for {topic} in {subject}? A) Option A B) Option B C) Option C D) Option D'
    },
    'programming': {
        'easy': 'Write a simple {subject} program/code to demonstrate basic concepts of {topic}. Include comments.',
        'medium': 'Implement a {subject} solution for {topic} problem. Your code should handle edge cases and include proper error handling. [15 Points]',
        'hard': 'Design and implement an efficient {subject} algorithm for {topic}. Optimize for time and space complexity. Include test cases and documentation. [25 Points]'
    },
    'output': {
        'easy': 'What will be the output of the following {subject} code related to {topic}? Trace the execution step by step.',
        'medium': 'Trace the execution of the given {subject} code for {topic} and determine the output. Explain each step. [10 Points]',
        'hard': 'Analyze the following {subject} code related to {topic}. Predict the output, explain the logic, and identify any potential issues or optimizations. [15 Points]'
    }
}

# Generate new entries for each course
new_entries = []
for course_code, course_info in courses.items():
    start_id = course_info['last_id'] + 1
    for i in range(50):  # Add 50 entries per course
        new_id = start_id + i
        topic = random.choice(course_info['topics'])
        year = random.choice(course_info['years'])
        exam_type = random.choice(course_info['exam_types'])
        question_type = random.choice(course_info['question_types'])
        difficulty = random.choice(course_info['difficulties'])
        
        # Generate question based on template
        template = question_templates[question_type][difficulty]
        question = template.format(topic=topic, subject=course_info['subject'])
        
        new_entry = {
            'id': f'{course_code}{new_id}',
            'subject': course_info['subject'],
            'topic': topic,
            'year': year,
            'exam_type': exam_type,
            'question_type': question_type,
            'difficulty': difficulty,
            'question': question
        }
        new_entries.append(new_entry)

# Append new entries to the CSV file
with open('quiz_data.csv', 'a', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'subject', 'topic', 'year', 'exam_type', 'question_type', 'difficulty', 'question'])
    writer.writerows(new_entries)

print(f'Successfully added {len(new_entries)} new entries!')
print(f'New entries added per course: 50')
