import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify, render_template
import imaplib
import email
from email.policy import default

# Configuration for email integration
EMAIL = 'your_email@example.com'
PASSWORD = 'your_password'
SERVER = 'imap.example.com'

# Load dataset
def load_data():
    df = pd.read_csv('spam.csv', encoding='utf-8')
    df = df[['label', 'text']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Train and save models
def train_models():
    df = load_data()
    df['text'] = df['text'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    models = {
        'naive_bayes': MultinomialNB(),
        'svm': SVC(kernel='linear')
    }
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} model accuracy: {accuracy}')
        print(classification_report(y_test, y_pred))
        joblib.dump(pipeline, f'{name}_model.pkl')

# Fetch emails from the server
def fetch_emails():
    mail = imaplib.IMAP4_SSL(SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')

    status, data = mail.search(None, 'ALL')
    mail_ids = data[0].split()

    emails = []
    for mail_id in mail_ids:
        status, data = mail.fetch(mail_id, '(RFC822)')
        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email, policy=default)
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                emails.append(part.get_payload(decode=True).decode('utf-8'))
    mail.logout()
    return emails

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get('model', 'naive_bayes')
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided.'}), 400
    
    model = joblib.load(f'{model_name}_model.pkl')
    prediction = model.predict([message])[0]
    return jsonify({'spam': bool(prediction)})

@app.route('/fetch_emails', methods=['GET'])
def fetch_and_classify_emails():
    emails = fetch_emails()
    model = joblib.load('naive_bayes_model.pkl')
    classified_emails = [{'email': email, 'spam': bool(model.predict([preprocess_text(email)])[0])} for email in emails]
    return jsonify(classified_emails)

if __name__ == '__main__':
    train_models()
    app.run(debug=True)
