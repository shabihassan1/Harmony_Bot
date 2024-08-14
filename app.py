import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load DistilBERT model and tokenizer for classification
distilbert_model_name = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = AutoModelForSequenceClassification.from_pretrained("./medical_conversational_distilbert")

# Load a lighter question-answering model and tokenizer
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = DistilBertTokenizer.from_pretrained(qa_model_name)
qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)
qa_model.to(device)

# Load passages for retrieval
passages = []
with open("ai_medical_chatbot_passages.tsv", "r", encoding="utf-8") as file:
    for line in file:
        passages.append(line.strip())

# Fit the vectorizer on the passages
vectorizer = TfidfVectorizer().fit(passages)

# Function to retrieve the most relevant passage
def retrieve_passage(query, vectorizer, passages):
    query_vec = vectorizer.transform([query])
    passages_vec = vectorizer.transform(passages)
    results = cosine_similarity(query_vec, passages_vec).flatten()
    top_index = results.argmax()
    return passages[top_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    # Process user input using the classification model
    inputs = distilbert_tokenizer(user_input, return_tensors="pt").to(device)
    outputs = distilbert_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    
    # Depending on the predicted class, choose the appropriate response mechanism
    if predicted_class == 0:  # Assume class 0 is for general conversation
        response_text = "I'm here to help with your general queries."
    else:  # Assume class 1 is for question answering
        passage = retrieve_passage(user_input, vectorizer, passages)
        qa_inputs = qa_tokenizer(user_input, passage, return_tensors="pt").to(device)
        qa_outputs = qa_model(**qa_inputs)
        answer_start = qa_outputs.start_logits.argmax()
        answer_end = qa_outputs.end_logits.argmax() + 1
        response_text = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(qa_inputs["input_ids"][0][answer_start:answer_end]))
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
