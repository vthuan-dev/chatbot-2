import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from .env file
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    print("Error: Please set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in .env file")
    print("Copy .env.example to .env and update with your values")
    exit(1)

def test_ctu_chatbot(prompt):
    """Test the CTU chatbot endpoint"""
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7
        }
    }
    
    print(f"Sending: {prompt}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'COMPLETED':
                output = result.get('output', {})
                print(f"\nResponse: {output.get('response', 'No response')}")
                print(f"Tokens: {output.get('tokens_generated', 0)}")
                print(f"Time: {output.get('response_time', 0)}s")
            else:
                print(f"\nStatus: {result.get('status')}")
                if 'error' in result:
                    print(f"Error: {result.get('error')}")
        else:
            print(f"\nError: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"\nException: {e}")

# Test it
if __name__ == "__main__":
    print("CTU Chatbot Test")
    print("=" * 50)
    test_ctu_chatbot("Xin chào, cho tôi biết về Đại học Cần Thơ?")
