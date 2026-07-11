from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    ttft_ms: float          # time to first token, milliseconds
    total_latency_ms: float # end-to-end latency, milliseconds
    output_tokens: int      # number of tokens generated
    tokens_per_sec: float   # output throughput


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
