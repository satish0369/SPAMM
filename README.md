# Spam-Classification
# Spam Classification Through NLP Techniques

Spam messages pose a significant challenge in digital communication, leading to security risks and inefficiencies. This project focuses on developing a spam classification system using Natural Language Processing (NLP) techniques. The system preprocesses text data, extracts features, and applies machine learning models to classify messages as spam or non-spam. It provides a REST API for seamless integration with messaging platforms and a web-based user interface for result monitoring. Designed for real-time processing and scalability, the system ensures high accuracy and security, contributing to a more reliable communication environment.

## Installation

1. Clone the repository:
    ```
    git clone <repository_url>
    ```
2. Navigate to the project directory:
    ```
    cd spam_classification
    ```
3. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Download and prepare dataset
   ```
   python download_and_prepare_dataset.py
   ```
5. Run the classifier script
   ```
   python spam_classifier.py
   ```

## API

-Use the form to classify messages:

Enter a message in the text area.
Select a model (Naive Bayes or SVM).
Click the "Classify" button to see the result (Spam or Not Spam).
Fetch and classify emails:

Send a GET request to http://127.0.0.1:5000/fetch_emails to fetch and classify emails from the configured email account.
