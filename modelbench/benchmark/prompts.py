from dataclasses import dataclass

SUITE_VERSION = "v1"


@dataclass(frozen=True)
class Prompt:
    text: str
    max_tokens: int
    label: str


SUITE_V1: list[Prompt] = [
    Prompt(
        text="What is the capital of France?",
        max_tokens=20,
        label="short_factual_capital",
    ),
    Prompt(
        text="List the planets in the solar system.",
        max_tokens=40,
        label="short_factual_list",
    ),
    Prompt(
        text="Explain the difference between TCP and UDP in simple terms.",
        max_tokens=100,
        label="medium_reasoning_networking",
    ),
    Prompt(
        text="Write a Python function that reverses a string.",
        max_tokens=80,
        label="medium_instruction_code",
    ),
    Prompt(
        text="Write a three-sentence story about a robot learning to cook.",
        max_tokens=60,
        label="medium_creative_story",
    ),
]
