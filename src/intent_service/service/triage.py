"""Use-case orchestration: triage one customer message into a decision.

Note what is ABSENT here: no FastAPI, no sklearn, no file paths, no Redis,
no environment variables. Pure orchestration = trivially testable (construct
with a fake IntentModel in one line, no mocking framework needed).

The split of responsibilities is the point:
  - the ADAPTER knows how to get a score out of a fitted pipeline
  - the DOMAIN knows what a score means operationally (`policy.decide`)
  - THIS class knows the order those two happen in, and nothing else

A broken model must surface as an exception here, not be swallowed -- the
caller (the API layer) decides how to respond (503, alert, retry).
"""

import hashlib
from dataclasses import dataclass
from typing import cast

from intent_service.domain.entities import CustomerMessage, Intent
from intent_service.domain.policy import PolicyThresholds, TriageDecision, decide
from intent_service.service.interfaces import DecisionCache, IntentModel


@dataclass(frozen=True)
class TriageResult:
    """The decision plus one piece of operational metadata the domain has no
    business knowing about: whether we had to compute it."""

    decision: TriageDecision
    cache_hit: bool


@dataclass(frozen=True)
class TriageService:
    """Constructor dependency injection: the service is handed its model,
    its thresholds and (optionally) its cache. It never goes looking for
    any of them."""

    model: IntentModel
    thresholds: PolicyThresholds
    cache: DecisionCache | None = None

    def cache_key(self, text: str) -> str:
        """Hash the input rather than keying on it directly.

        Two reasons, both operational: the raw text is customer data and
        does not belong in a cache key that shows up in backend monitoring
        tools, and a hash bounds the key length regardless of input size.
        The model version and thresholds are folded in so a retrain or a
        policy change invalidates every entry automatically -- serving a
        decision made under superseded rules would be a silent correctness
        bug, and this is what prevents it.
        """
        material = (
            f"{self.model.model_version}|"
            f"{self.thresholds.auto_route_floor}|{self.thresholds.reject_floor}|{text}"
        )
        return "decision:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def triage(self, message: CustomerMessage) -> TriageResult:
        key = self.cache_key(message.text)

        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                # Validated on the way out, not trusted blindly: a cache can
                # hold a payload written by an older, differently-shaped
                # version of this code. If it no longer parses, fall through
                # and recompute rather than serving a malformed decision.
                try:
                    return TriageResult(
                        decision=TriageDecision.model_validate_json(cached), cache_hit=True
                    )
                except ValueError:
                    pass

        label, confidence = self.model.predict(message.text)
        # The adapter guarantees `label` is one of `self.model.labels`, which
        # must equal the domain's Intent set (checked at load time in
        # SklearnIntentModel.load) -- pydantic still validates it for real
        # inside `decide`, this cast only satisfies the type checker.
        decision = decide(
            text=message.text,
            intent=cast(Intent, label),
            confidence=confidence,
            model_version=self.model.model_version,
            thresholds=self.thresholds,
        )

        if self.cache is not None:
            self.cache.set(key, decision.model_dump_json())

        return TriageResult(decision=decision, cache_hit=False)
