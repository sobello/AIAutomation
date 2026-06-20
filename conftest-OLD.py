import requests
import time
import pytest
 
class OllamaTestClient:
 
    def __init__(self, model="tinyllama:latest", host="localhost", port=11434): 
        #constructor to initialize the client with model, host and port
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
        start_time=time.time()
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,    
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }, timeout=30)
            end_time=time.time()
            if response.status_code == 200:
                result= response.json()
                return{
                    "text": result["response"],
                    "prompt_token":len(prompt.split()),
                    "completion_token":len(result["response"].split()),
                    "latency": end_time-start_time
                }
            else:
                print(f"Failed to generate text: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
 
 
@pytest.fixture
def llm_client():
    client=OllamaTestClient(model="tinyllama")
    return client
 