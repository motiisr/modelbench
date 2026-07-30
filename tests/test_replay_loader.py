import pytest

from modelbench.replay.loader import Sample, load_samples


def test_load_samples_parses_prompt_only_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"prompt": "What is 2+2?"}\n{"prompt": "Say hi"}\n')

    samples = load_samples(path)

    assert samples == [
        Sample(prompt="What is 2+2?", expected=None, task_type=None),
        Sample(prompt="Say hi", expected=None, task_type=None),
    ]


def test_load_samples_parses_optional_fields(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"prompt": "What is 2+2?", "expected": "4", "task_type": "math"}\n')

    samples = load_samples(path)

    assert samples == [Sample(prompt="What is 2+2?", expected="4", task_type="math")]


def test_load_samples_skips_blank_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"prompt": "a"}\n\n   \n{"prompt": "b"}\n')

    samples = load_samples(path)

    assert len(samples) == 2


def test_load_samples_raises_on_malformed_json_with_line_number(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"prompt": "a"}\nnot json\n{"prompt": "b"}\n')

    with pytest.raises(ValueError, match="line 2"):
        load_samples(path)


def test_load_samples_raises_on_missing_prompt_field(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"expected": "4"}\n')

    with pytest.raises(ValueError, match="prompt"):
        load_samples(path)


def test_load_samples_raises_on_empty_file(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text("")

    with pytest.raises(ValueError, match="[Nn]o samples"):
        load_samples(path)
