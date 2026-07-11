import pytest
from deepeval.test_case import LLMTestCase
from conftest import OllamaTestClient, OllamaJudge

 
def test_response_relevancy_detection():
    """
    Single test case to detect response relevancy using Ollama as judge.
    This test validates that Ollama returns relevant answers to prompts.
    Will FAIL if the response is irrelevant or incorrect.
    """
    client = OllamaTestClient(model="tinyllama:latest")
    judge = OllamaJudge(model="tinyllama:latest")
   
    # Check if Ollama is available...and skip the test if not
    if not client.check_model_available():
        pytest.skip("Ollama model not available")
   
    # Ask a specific question
    prompt = "What is the capital of France?"
    result = client.generate_result(prompt)
   
    assert result is not None, "Generation failed - no response from Ollama"
    assert len(result["text"]) > 0, "Empty response from Ollama"
   
    # Create test case for evaluation
    test_case = LLMTestCase(
        input=prompt,
        expected_output="Paris",
        actual_output=result["text"]
    )
   
    # Evaluate using Ollama judge
    evaluation = judge.evaluate(test_case, metric_name="relevancy")
    print(f"\n Evaluation Result: {evaluation}\n")
   
    # Check for relevant keywords in the response
    response_text = result["text"].lower()
    # relevant_keywords = ["paris", "capital", "france", "city"]
    relevant_keywords = ["toronto", "ottawa", "paris"]
   
    # Verify the response contains relevant information
    has_relevant_keywords = any(keyword in response_text for keyword in relevant_keywords)
   
    # Assertions will fail if response is irrelevant
    assert has_relevant_keywords, (
        f"RELEVANCY TEST FAILED: Expected relevant response about Paris/France, "
        f"but got irrelevant response: '{result['text']}'"
    )
    assert evaluation["passed"], f"Judge evaluation failed: {evaluation['reasoning']}"
    assert evaluation["score"] > 0, "Relevancy score should be positive"
    
