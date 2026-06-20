def test_basic_response(llm_client):
    """Test basic response generation using the llm_client fixture"""
    response = llm_client.generate_result("What is the capital of France?", max_tokens=50, temperature=0.5)
    assert response is not None, "Response should not be None"
    assert response.get("text") is not None, "Response text should not be None"
    assert len(response.get("text")) > 0, "Response text should not be empty"
 
 
def test_simpleqa(llm_client):
    """Test simple question answering capability of the model"""
    prompt = "What is capital of France?"
    response = llm_client.generate_result(prompt, max_tokens=10, temperature=0.5)
    response_text = response.get("text", "").strip().lower()
    print(f"\nModel response for capital of France: '{response_text}'")
    assert "paris" in response_text, f"Expected 'paris' in response', got '{response_text}'"

def test_multiple_turns(llm_client):
    """Test multiple turns of conversation"""
    prompt1 = """
    user: My name is sam.
    Assistant: Hello Sam! How can I assist you today?
    user: What is my name?"""
    response1 = llm_client.generate_result(prompt1, max_tokens=10, temperature=0.5)
    response_text1 = response1.get("text", "").strip().lower()
    assert "sam" in response_text1, f"Expected 'sam' show in response', got '{response_text1}'"


# def test_canada_population(llm_client):
#     """Test the model's knowledge of Canada's current population."""
#     prompt = "What is the current population of Canada?"
#     response = llm_client.generate_result(prompt, max_tokens=20, temperature=0.5)
#     response_text = response.get("text", "").strip().lower()
#     print(f"Model response for Canada population: '{response_text}'")
#     assert "38 million" in response_text or "38,000,000" in response_text or "38m" in response_text, \
#         f"Expected Canada population to be reported as 38 million, got '{response_text}'"

def test_capital_of_canada(llm_client):
    """Test the question about the capital of Canada."""
    prompt = "What is the capital of Canada?"
    response = llm_client.generate_result(prompt, max_tokens=10, temperature=0.5)
    response_text = response.get("text", "").strip().lower()
    print(f"\n Model response for Canada capital: '{response_text}'")
    assert "ottawa" in response_text, \
        f"Expected 'ottawa' in response, got '{response_text}'"

