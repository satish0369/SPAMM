import requests
import pandas as pd
import zipfile

# URL of the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

# Download the dataset
response = requests.get(url)
with open('smsspamcollection.zip', 'wb') as f:
    f.write(response.content)

# Unzip the dataset
with zipfile.ZipFile('smsspamcollection.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

# Save to CSV
df.to_csv('spam.csv', index=False, encoding='utf-8')
print("Dataset downloaded and saved as spam.csv")
