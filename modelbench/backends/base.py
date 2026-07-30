from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    """Result of a single inference call.

    tokens_per_sec methodology differs by backend and is NOT directly
    comparable across them: OllamaBackend derives it from the server-reported
    eval_duration (pure generation time, excludes network), while the cloud
    backends (OpenAI/Anthropic) derive it from wall-clock total_latency_ms
    (includes network/protocol overhead). A replay comparison table must
    treat throughput as backend-relative context, not a globally ranked
    metric, until this is normalized.
    """

    ttft_ms: float          # time to first token, milliseconds
    total_latency_ms: float # end-to-end latency, milliseconds
    output_tokens: int      # number of tokens generated
    tokens_per_sec: float   # output throughput (see methodology note above)
    output_text: str = ""   # generated text; empty for hardware-benchmark-only callers   # output throughput (see methodology note above)


class BackendRunner(ABC):
    @abstractmethod
    def start(self, model_id: str) -> None:
        """Spawn the backend server process, wait until ready."""

    @abstractmethod
    def warm_up(self) -> None:
        """Run 5 dummy inferences. Not recorded."""

    @abstractmethod
    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        """Single inference. Returns result with timing measured."""

    @abstractmethod
    def stop(self) -> None:
        """Gracefully stop the backend. Wait for GPU memory release."""
