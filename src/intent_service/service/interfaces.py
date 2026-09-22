"""Service-layer ports. The service depends on THESE, never on a concrete
ML framework or a concrete piece of infrastructure.

Protocols, not ABCs: an adapter satisfies a port by having the right shape,
so no adapter ever has to import this module. That keeps the dependency
arrow pointing one way -- inward -- which is the whole clean-architecture
bargain.

Each Protocol here is a seam that lets `TriageService` be unit-tested with a
six-line fake instead of a real TF-IDF + SVM pipeline and a real Redis, and
lets either be replaced (SVM -> transformer, Redis -> Memcached) by writing
one new adapter with zero changes to the service or API layers.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class IntentModel(Protocol):
    """Anything that can score a raw message against the label set."""

    model_version: str
    labels: list[str]

    def predict(self, text: str) -> tuple[str, float]:
        """Return (predicted_label, confidence) for one message."""
        ...


@runtime_checkable
class DecisionCache(Protocol):
    """A best-effort store for previously-computed decisions.

    Every method is explicitly allowed to fail silently and return None: the
    cache is an optimisation, never a source of truth. A service that goes
    down because its cache went down has turned an optimisation into a
    dependency, which is the opposite of what a cache is for.
    """

    def get(self, key: str) -> str | None:
        """Return the stored payload, or None on a miss OR any backend error."""
        ...

    def set(self, key: str, value: str) -> None:
        """Store a payload. Swallows backend errors by contract."""
        ...

    def healthy(self) -> bool:
        """True when the backend answered a ping. Reported by /v1/ready as
        informational detail -- an unhealthy cache must not make the service
        unready."""
        ...
