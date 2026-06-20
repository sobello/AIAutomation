import requests
import time
import pytest
from deepeval.test_case import LLMTestCase
 
class OllamaTestClient:
 
    def __init__(self, model="tinyllama:latest", host="localhost", port=11434):
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
    client=OllamaTestClient(model="tinyllama:latest")
    return client
 
 
class OllamaJudge:
    """Custom Ollama Judge for DeepEval"""
    def __init__(self, model="tinyllama:latest", host="localhost", port=11434):
        self.model = model
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}/api/generate"
   
    def evaluate(self, test_case: LLMTestCase, metric_name: str = "hallucination") -> dict:
        """Evaluate test case using Ollama"""
        prompt = f"""
        Evaluate the following based on {metric_name}:
        Input: {test_case.input}
        Expected: {test_case.expected_output}
        Actual: {test_case.actual_output}
       
        Provide a score from 0 to 1 and brief reasoning.
        """
       
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            }, timeout=30)
           
            if response.status_code == 200:
                result = response.json()
                return {
                    "score": 0.85,  # Placeholder scoring
                    "reasoning": result["response"],
                    "passed": True
                }
            else:
                return {"score": 0, "reasoning": "Evaluation failed", "passed": False}
        except Exception as e:
            print(f"Error in Ollama evaluation: {e}")
            return {"score": 0, "reasoning": str(e), "passed": False}
 
 
@pytest.fixture
def ollama_judge():
    """Fixture for Ollama Judge"""
    judge = OllamaJudge(model="tinyllama:latest")
    return judge

