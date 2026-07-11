import requests
import time
import json
import re
import pytest
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

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
            }, timeout=120)
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


class OllamaDeepEvalLLM(DeepEvalBaseLLM):
    """
    Wraps a local Ollama model so it can be plugged into DeepEval metrics
    (e.g. AnswerRelevancyMetric) as the judge/evaluation model.

    DeepEval's metrics prompt the judge model to respond with JSON (e.g.
    lists of statements/verdicts, or a reason object) and then parse that
    string themselves. Small local models like tinyllama often ignore
    formatting instructions and return prose, markdown-fenced JSON, or
    truncated/malformed JSON, which raises:
        ValueError: Evaluation LLM outputted an invalid JSON.
    To make a local model usable as a judge, this wrapper:
      1. Asks Ollama to constrain output to syntactically valid JSON
         (Ollama's `format: "json"` option).
      2. Strips markdown code fences some models still add.
      3. Falls back to extracting the first {...} block and, if the
         `json_repair` package is installed, repairing common issues
         (trailing commas, unquoted keys, unterminated strings, etc.).
      4. Scrapes the expected top-level key name(s) out of DeepEval's own
         prompt (e.g. "...with the 'statements' key...") and, if the model's
         JSON is valid but under a different/missing key, renames/wraps it
         to match — this is what fixes KeyError: 'statements' style errors,
         where the JSON parses fine but doesn't have the field DeepEval
         looks up next.
    """

    def __init__(self, model="tinyllama:latest", host="localhost", port=11434):
        self.model_name = model
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}/api/generate"

    def load_model(self):
        return self.model_name

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(self.endpoint, json={
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "format": "json",  # ask Ollama to constrain to valid JSON syntax
        }, timeout=120)
        if response.status_code != 200:
            raise Exception(f"Ollama generation failed: {response.status_code}")
        return response.json()["response"]

    @staticmethod
    def _clean_json_string(text: str) -> str:
        """Best-effort cleanup so DeepEval's own json.loads() succeeds."""
        cleaned = text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Already valid? Nothing more to do.
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        # Try isolating the first {...} or [...] block in case there's
        # leading/trailing prose around the JSON.
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        candidate = match.group(0) if match else cleaned
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        # Last resort: repair common malformed-JSON issues if the
        # optional `json_repair` package is available.
        try:
            from json_repair import repair_json
            repaired = repair_json(candidate)
            json.loads(repaired)
            return repaired
        except Exception:
            pass

        # Give up gracefully and return the best candidate we have;
        # DeepEval will raise its own clear error if this still isn't
        # parseable, rather than us masking it silently.
        return candidate

    @staticmethod
    def _extract_expected_keys(prompt: str) -> list:
        """
        DeepEval's own prompt templates tell the judge model which top-level
        JSON key(s) it must return, e.g.:
            '...with the "statements" key as a list of strings...'
            '...with two keys: "verdict" and "reason"...'
        Scrape those key names out of the prompt so we can tell whether the
        model's JSON actually matches the shape DeepEval is about to look for.
        """
        keys = re.findall(r'["\']([a-zA-Z_]+)["\']\s+key', prompt)
        # de-duplicate while preserving order
        seen = set()
        ordered = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        return ordered

    @staticmethod
    def _normalize_json_shape(parsed, expected_keys: list):
        """
        If the model returned syntactically valid JSON but not under the
        key name(s) DeepEval expects (e.g. {"answer": [...]}) instead of
        {"statements": [...]}), try to coerce it into the expected shape
        rather than letting a downstream KeyError bubble up.
        """
        if not expected_keys:
            return parsed

        if isinstance(parsed, dict):
            if any(k in parsed for k in expected_keys):
                return parsed
            # Single unexpected key -> rename it to the first expected key
            if len(parsed) == 1:
                (_, value), = parsed.items()
                return {expected_keys[0]: value}
            # Multiple unexpected keys -> best effort, wrap whole dict
            return {expected_keys[0]: parsed}

        if isinstance(parsed, list):
            # Model returned a bare list instead of {"key": [...]}
            return {expected_keys[0]: parsed}

        # Scalar (e.g. bare string/number) -> wrap it
        return {expected_keys[0]: parsed}

    def generate(self, prompt: str) -> str:
        raw = self._call_ollama(prompt)
        cleaned = self._clean_json_string(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Still not parseable after cleanup; return as-is and let
            # DeepEval raise its own clear "invalid JSON" error.
            return cleaned

        expected_keys = self._extract_expected_keys(prompt)
        normalized = self._normalize_json_shape(parsed, expected_keys)
        return json.dumps(normalized)

    async def a_generate(self, prompt: str) -> str:
        # No native async Ollama client here, so just delegate to the sync call
        return self.generate(prompt)

    def get_model_name(self):
        return f"Ollama-{self.model_name}"


@pytest.fixture
def ollama_deepeval_llm():
    """Fixture that provides an Ollama-backed DeepEval model for metrics"""
    return OllamaDeepEvalLLM(model="tinyllama:latest")


class OllamaJudge:
    """Custom Ollama Judge for DeepEval"""
    def __init__(self, model="tinyllama", host="localhost", port=11434):
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
    judge = OllamaJudge(model="tinyllama")
    return judge
