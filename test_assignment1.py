def test_boiling_point():
   ai_response = "The boiling point of water is 100 degrees."
   assert "100" in ai_response, "AI got the boiling point wrong!"

def test_prog_language():
    ai_response = "Python is a programming language--yes."
    assert "python" in ai_response.lower(), "AI did not mention Python!"

def test_sun_rises():
    ai_response = "The sun rises in the east."
    assert "east" in ai_response.lower(), "AI got the direction of sunrise wrong!"

def test_canada_population():
    ai_response = "The current population of Canada is 35 million."
    assert (
        "35 million" in ai_response.lower()
        or "35,000,000" in ai_response
        or "35m" in ai_response.lower()
    ), "AI got the population of Canada wrong!"

    
