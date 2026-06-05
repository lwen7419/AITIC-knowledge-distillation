import requests
import os

os.environ['no_proxy'] = 'localhost,127.0.0.1'

response = requests.post("http://localhost:11434/api/generate",
    json={"model": "qwen2.5:7b", "prompt": "Hello!", "stream": False})

print(response.json()["response"])