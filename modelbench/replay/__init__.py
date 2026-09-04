"""Replay engine — verify candidate models against real traffic samples.

Usage::

    from modelbench.replay import Sample, Candidate, ReplayRunner

    samples = [Sample(prompt="2+2?", expected="4")]
    candidates = [Candidate(name="my-model", backend=ollama, model_id="llama3.3:70b")]
    report = ReplayRunner(repeats=3).run(samples, candidates)

    for cr in report.candidates:
        print(f"{cr.name}: score={cr.mean_score}, latency={cr.latency_p50_ms}ms")
"""

from modelbench.replay.loader import Sample, load_samples
from modelbench.replay.registry import resolve_candidate
from modelbench.replay.runner import Candidate, CandidateReport, ReplayReport, ReplayRunner

__all__ = [
    "Sample",
    "load_samples",
    "Candidate",
    "CandidateReport",
    "ReplayReport",
    "ReplayRunner",
    "resolve_candidate",
]
