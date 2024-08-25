import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from huggingface_hub import InferenceClient
from datasets import load_dataset
import markdown2
import signal

app = Flask(__name__, static_folder='static', template_folder='templates')

hf_token = "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"  

chat_doctor_dataset = load_dataset("avaliev/chat_doctor")
mental_health_dataset = load_dataset("Amod/mental_health_counseling_conversations")

client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

def select_relevant_context(user_input):
    mental_health_keywords = [
        "anxious", "depressed", "stress", "mental health", "counseling", 
        "therapy", "feelings", "worthless", "suicidal", "panic", "anxiety"
    ]
    medical_keywords = [
        "symptoms", "diagnosis", "treatment", "doctor", "prescription", "medication",
        "pain", "illness", "disease", "infection", "surgery"
    ]

    # Check if the input contains any mental health-related keywords
    if any(keyword in user_input.lower() for keyword in mental_health_keywords):
        example = mental_health_dataset['train'][0]
        context = f"Counselor: {example['Response']}\nUser: {example['Context']}"
    # Check if the input contains any medical-related keywords
    elif any(keyword in user_input.lower() for keyword in medical_keywords):
        example = chat_doctor_dataset['train'][0]
        context = f"Doctor: {example['input']}\nPatient: {example['output']}"
    else:
        # If no specific keywords are found, provide a general response
        context = "You are a general assistant. Respond to the user's query in a helpful manner."

    return context

def create_prompt(context, user_input):
    prompt = (
        f"{context}\n\n"
        f"User: {user_input}\nAssistant:"
    )
    return prompt

# Function to render Markdown into HTML
def render_markdown(text):
    return markdown2.markdown(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    context = select_relevant_context(user_input)
    
    prompt = create_prompt(context, user_input)
    
    response = ""
    for message in client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=True,
    ):
        response += message.choices[0].delta.content
    
    formatted_response = render_markdown(response)
    
    return jsonify({"response": formatted_response})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    if request.environ.get('werkzeug.server.shutdown'):
        shutdown_server()
    else:
        os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"message": "Server is shutting down..."})

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        os.kill(os.getpid(), signal.SIGINT)  # Kill the process if Werkzeug is not available
    else:
        func()

if __name__ == '__main__':
    app.run(debug=True)
