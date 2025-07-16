import requests
import json

# URL of the endpoint
url = "https://example.com/api/endpoint"

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_ACCESS_TOKEN",  # optional
    "Custom-Header": "CustomValue"                # optional
}

# Request body (payload)
payload = {
    "name": "Ganga Raju Battem",
    "message": "Calling REST API",
    "userId": 12345
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Print response
print("Status Code:", response.status_code)
print("Response Body:", response.text)
