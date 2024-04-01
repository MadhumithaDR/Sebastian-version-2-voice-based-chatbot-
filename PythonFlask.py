from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import csv
import os

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Function to process natural language query and return 1-D embedding
def process_query(query):
    inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()  # Ensure the output is numpy array

# Function to find the most similar question based on 1-D embeddings
def find_most_similar_question(query_embedding, question_embeddings, questions, answers):
    min_distance = float('inf')
    most_similar_index = -1
    for index, q_embedding in enumerate(question_embeddings):
        # Ensure both embeddings are 1-D before comparing
        distance = cosine(query_embedding, q_embedding)
        if distance < min_distance:
            min_distance = distance
            most_similar_index = index
    if most_similar_index != -1:
        return answers[most_similar_index]
    else:
        return None

# Route to handle query processing
@app.route('/query', methods=['POST'])
def query():
    file_name = 'C:\\Users\DELL\OneDrive\Desktop\datasets\Questions_answers.csv'  # Adjust to your CSV file path
    if not os.path.exists(file_name):
        return jsonify({'message': f"File '{file_name}' not found."}), 404
    
    # Load your dataset and generate 1-D embeddings for questions
    questions = []
    answers = []
    question_embeddings = []
    with open(file_name, 'r', encoding='ISO-8859-1') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            questions.append(row[0])
            answers.append(row[1])
            # Process each question to get 1-D embedding
            embeddings = process_query(row[0])
            question_embeddings.append(embeddings)

    query = request.json.get('query')
    if query:
        query_embedding = process_query(query)
        found_answer = find_most_similar_question(query_embedding, question_embeddings, questions, answers)
        if found_answer:
            return jsonify({'answer': found_answer}), 200
        else:
            return jsonify({'message': 'No answer found.'}), 404
    else:
        return jsonify({'message': 'Query not provided.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
