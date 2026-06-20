def test_math_works():
    result = 2 + 2          # This is what we GOT
    assert result == 4      # This is what we EXPECTED
 
def test_capital_of_france():
    ai_answer = "The capital of France is Paris."
    # .lower() makes Paris/PARIS/paris all match
    assert "paris" in ai_answer.lower()
 
def test_intentional_failure():         # ← designed to fail!
    ai_answer = "The capital of France is London."
    assert "paris" in ai_answer.lower(),  "AI gave wrong city!"
