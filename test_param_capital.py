import pytest

# @pytest.mark.parametrize is a pytest decorator that lets you run the same test function multiple
# times with different input values — avoiding copy-pasting the same test logic.

@pytest.mark.parametrize("prompt, assert_value", [
    ("What is the capital of Canada?", "ottawa"),
    ("What is the capital of Canada?", "madrid"),
    ("What is the capital of Japan?", "tokyo"),
])
def test_capital_of_canada(llm_client, prompt, assert_value):
    """Test the question about the capital of Canada."""
    response = llm_client.generate_result(prompt, max_tokens=10, temperature=0.5)
    response_text = response.get("text", "").strip().lower()
    print(f"\n Model response for '{prompt}': '{response_text}'")
    assert assert_value in response_text, \
        f"Expected '{assert_value}' in response, got '{response_text}'"


