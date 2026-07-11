from modelbench.benchmark.prompts import Prompt, SUITE_V1, SUITE_VERSION


def test_prompt_has_required_fields():
    p = Prompt(text="Hello", max_tokens=20, label="short_factual")
    assert p.text == "Hello"
    assert p.max_tokens == 20
    assert p.label == "short_factual"


def test_suite_v1_has_five_prompts():
    assert len(SUITE_V1) == 5


def test_suite_v1_all_have_max_tokens():
    for p in SUITE_V1:
        assert p.max_tokens > 0, f"Prompt '{p.label}' missing max_tokens"


def test_suite_version_is_string():
    assert isinstance(SUITE_VERSION, str)
    assert SUITE_VERSION == "v1"
