import requests

# URL of the REST endpoint
url = "http://127.0.0.1:5000/post-example"

# Data to be sent in the POST request
payload = {"name": "Ganga Raju"}

# Optional: custom headers
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your_token_here"
}

# Send the POST request
response = requests.post(url, json=payload, headers=headers)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
