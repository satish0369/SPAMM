import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "message": "Congratulations! You have won a $1000 gift card.",
    "model": "naive_bayes"
}

response = requests.post(url, json=data)
print(response.json())
