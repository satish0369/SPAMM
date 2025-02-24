document.getElementById('spam-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const message = document.getElementById('message').value;
    const model = document.getElementById('model').value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, model }),
    });

    const result = await response.json();
    document.getElementById('result').innerText = result.spam ? 'Spam' : 'Not Spam';
});
