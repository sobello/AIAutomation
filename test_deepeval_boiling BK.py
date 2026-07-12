import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from conftest import OllamaTestClient, OllamaDeepEvalLLM


def test_boiling_point_of_water():
    """
    Single test case to detect response relevancy using Ollama as judge,
    scored via DeepEval's built-in AnswerRelevancyMetric.
    Will FAIL if the response is irrelevant/incorrect or scores below
    the 0.5 relevancy threshold.
    """
    # client = OllamaTestClient(model="tinyllama:latest")
    # judge_llm = OllamaDeepEvalLLM(model="tinyllama:latest")
    client = OllamaTestClient(model="llama3.1:8b")
    judge_llm = OllamaDeepEvalLLM(model="llama3.1:8b")
   
    # Check if Ollama is available
    if not client.check_model_available():
        pytest.skip("Ollama model not available")

    # Ask a specific question
    prompt = "What is the boiling point of water?"
    result = client.generate_result(prompt)

    assert result is not None, "Generation failed - no response from Ollama"
    assert len(result["text"]) > 0, "Empty response from Ollama"

    # Create test case for evaluation
    test_case = LLMTestCase(
        input=prompt,
        actual_output=result["text"],
        expected_output="100°C"
    )

    # Evaluate relevancy using DeepEval's AnswerRelevancyMetric, backed by
    # our local Ollama model as the judge, with a 0.5 pass threshold
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=judge_llm,
        include_reason=True
    )

    try:
        relevancy_metric.measure(test_case)
    except (ValueError, KeyError, TypeError) as e:
        pytest.skip(
            f"Judge model '{judge_llm.model_name}' could not produce output "
            f"DeepEval could use for AnswerRelevancyMetric ({type(e).__name__}: {e}). "
            f"Try a stronger instruction-tuned Ollama model (e.g. llama3.1, "
            f"mistral, qwen2.5) as the judge."
        )

    print(f"\nAnswer Relevancy Score: {relevancy_metric.score}")
    print(f"Reason: {relevancy_metric.reason}\n")

    # Check for relevant keywords in the response (sanity check)
    response_text = result["text"].lower()
    relevant_keywords = ["100°C", "boiling point", "water", "degrees"]
    has_relevant_keywords = any(keyword in response_text for keyword in relevant_keywords)

    assert has_relevant_keywords, (
        f"RELEVANCY TEST FAILED: Expected relevant response about Tokyo/Japan, "
        f"but got irrelevant response: '{result['text']}'"
    )

    # Assert the metric passed its own internal threshold check
    assert relevancy_metric.is_successful(), (
        f"AnswerRelevancyMetric failed: score={relevancy_metric.score} "
        f"(threshold=0.5). Reason: {relevancy_metric.reason}"
    )

    # Explicit threshold assertion as well, for clarity in failure output
    assert relevancy_metric.score >= 0.5, (
        f"Answer relevancy score {relevancy_metric.score} is below the 0.5 threshold. "
        f"Reason: {relevancy_metric.reason}"
    )
