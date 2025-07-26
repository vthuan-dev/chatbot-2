from huggingface_hub import HfApi
import requests

# Test if model exists and is accessible
model_id = "thuanhero1/llama3-8b-finetuned-ctu"

print(f"Testing access to model: {model_id}")
print("-" * 50)

# Method 1: Check via API
try:
    api = HfApi()
    model_info = api.model_info(model_id)
    print(f"✓ Model found: {model_info.modelId}")
    print(f"✓ Private: {model_info.private}")
    print(f"✓ Files: {len(model_info.siblings)} files")
except Exception as e:
    print(f"✗ Error accessing model: {e}")

# Method 2: Try to access a file directly
print("\nTesting direct file access...")
try:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    response = requests.head(url)
    if response.status_code == 200:
        print(f"✓ Can access model files (status: {response.status_code})")
    else:
        print(f"✗ Cannot access model files (status: {response.status_code})")
except Exception as e:
    print(f"✗ Error: {e}")

# Method 3: Check what files are in the model
print("\nChecking model files...")
try:
    files_url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(files_url)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Model type: {data.get('pipeline_tag', 'unknown')}")
        print(f"✓ Model files:")
        for file in data.get('siblings', [])[:5]:  # Show first 5 files
            print(f"  - {file['rfilename']}")
    else:
        print(f"✗ Cannot get model info (status: {response.status_code})")
except Exception as e:
    print(f"✗ Error: {e}")
