import requests
import json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

def test_cortex():
    print(f"Testing Cortex at {URL}...")
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Hello! Who are you?"}
        ],
        "max_tokens": 50
    }

    try:
        response = requests.post(URL, json=payload)
        if response.status_code == 200:
            print("✅ Cortex Online!")
            data = response.json()
            print("Response:", data['choices'][0]['message']['content'])
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_cortex()
