import requests
import time
import pytest


class OllamaTestClient:

    def __init__(self, model="tinyllama:latest", host="localhost", port=11434):
        # constructor to initialize the client with model, host and port
        self.model = model
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}/api/generate"

    def check_model_available(self):
        """Check if the model is available in Ollama"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m["name"] for m in models_data.get("models", [])]
                return self.model in available_models
            else:
                print(f"Failed to fetch models: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False

    def generate_result(self, prompt, max_tokens=100, temperature=0.7):
        """Generate text using the Ollama model"""
        start_time = time.time()
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {                   # FIX: max_tokens sent via options
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }, timeout=30)
            end_time = time.time()
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result["response"],
                    "prompt_token": len(prompt.split()),
                    "completion_token": len(result["response"].split()),
                    "latency": end_time - start_time,
                }
            else:
                print(f"Failed to generate text: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error generating text: {e}")
            return None

    def generate_result_with_param(self, prompt, max_tokens=100, temperature=0.7, assert_func=None):
        """Generate text using the Ollama model, then optionally validate via assert_func."""
        start_time = time.time()
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {                   # FIX: max_tokens sent via options;
                    "num_predict": max_tokens, #      assert_func removed from payload
                    "temperature": temperature,
                },
            }, timeout=30)
            end_time = time.time()
            if response.status_code == 200:
                result = response.json()
                output = {
                    "text": result["response"],
                    "prompt_token": len(prompt.split()),
                    "completion_token": len(result["response"].split()),
                    "latency": end_time - start_time,
                }
                if assert_func is not None:    # FIX: actually call assert_func locally
                    assert_func(output)
                return output
            else:
                print(f"Failed to generate text: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error generating text: {e}")
            return None


@pytest.fixture
def llm_client():
    client = OllamaTestClient(model="tinyllama:latest")  # FIX: match default model name
    return client
