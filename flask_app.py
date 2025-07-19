from flask import Flask, request, render_template, redirect, url_for, session, flash
import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import re
from database import init_db, add_user, check_user

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a random secret key for session management

# Initialize the database
init_db()

# Load the trained Random Forest model
model = joblib.load('resume_screen.pkl')

# Load BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Clean the text by removing punctuation and converting to lowercase
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Get BERT embeddings for the text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Keyword-based relevance score calculation
def calculate_keyword_score(cleaned_resume, job_description_keywords):
    score = 0
    for keyword in job_description_keywords:
        if keyword.lower() in cleaned_resume:
            score += 1
    return score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            add_user(username, password)
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))  # Redirect to login page
        except:
            flash('Username already exists.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = check_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['username'] = username  # Store the username in the session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Redirect to home page after login
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/screen', methods=['POST'])
def screen_resumes():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    job_description = request.form['job_description']
    resumes = request.files.getlist('resumes')

    # Clean the job description text
    cleaned_job_description = clean_text(job_description)
    job_embedding = get_bert_embedding(cleaned_job_description)

    # Define job-specific keywords for keyword-based relevance scoring
    job_description_keywords = ["react", "node.js", "aws", "docker", "kubernetes", "python", "flask", "ci/cd"]

    results = []

    for resume in resumes:
        # Read and clean the resume text
        resume_text = resume.read().decode('utf-8', errors='ignore')
        cleaned_resume = clean_text(resume_text)
        resume_embedding = get_bert_embedding(cleaned_resume)

        # Get the relevance score (probability of being relevant)
        relevance_score = model.predict_proba(resume_embedding.flatten().reshape(1, -1))[0][1]  # Score for class 1 (Relevant)

        # Calculate a similarity score between resume and job description
        similarity_score = np.dot(resume_embedding.flatten(), job_embedding.flatten()) / (
            np.linalg.norm(resume_embedding.flatten()) * np.linalg.norm(job_embedding.flatten())
        )

        # Add keyword matching score
        keyword_score = calculate_keyword_score(cleaned_resume, job_description_keywords)

        # Append the results with scores
        results.append({
            'resume_name': resume.filename,
            'similarity_score': similarity_score,
            'relevance_score': relevance_score + keyword_score  # Augment relevance with keyword matching
        })

    # Sort the results based on relevance score, descending
    results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    return render_template('results.html', results=results)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)  # Remove username from session
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


from flask_cors import CORS
CORS(app)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)  # You can use a specific port
