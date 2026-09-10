import requests

# TensorFlow Serving prediction URL
url = "http://localhost:8501/v1/models/iris:predict"

# Flower measurements
data = {
    "instances": [
        [5.1, 3.5, 1.4, 0.2]
    ]
}

# Send data to the model
response = requests.post(url, json=data)

# Display result
print("Status code:", response.status_code)

print("\nPrediction response:")
print(response.json())