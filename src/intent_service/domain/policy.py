"""Pure business rules: turning a model score into an operational decision.

This is the heart of the domain layer and the reason the service is a
*decision* system rather than a classifier wrapped in HTTP. The model says
"this message looks like a complaint, confidence 0.62". That is not a
decision. The policy below is what says "route it to quality_assurance at
urgent priority, automatically, without a human".

Rules for this file (enforced by import-linter):
- stdlib + pydantic only -- no sklearn, no FastAPI, no I/O, no logging
- deterministic and total: the same inputs always yield the same decision,
  and every input yields one

Keeping the policy pure is what makes the behavioural suite possible. An
invariance or directional test can call `decide()` a thousand times with
crafted inputs and never touch a model, a socket, or a clock.
"""

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intent_service.domain.entities import Intent

# --- the decision vocabulary ------------------------------------------------

# Three mutually exclusive outcomes. This closed set is the product contract:
# a caller can build a workflow around exactly these branches and never have
# to handle an "other".
Action = Literal["auto_route", "human_review", "reject"]

Priority = Literal["normal", "urgent"]

Department = Literal[
    "quality_assurance",
    "sales",
    "technical_support",
    "customer_relations",
    "logistics",
    "returns",
    "safety",
]

# Where an escalated message goes, regardless of what the classifier thought it
# was about. See `decide()` for why this overrides DEPARTMENT_BY_INTENT.
SAFETY_DEPARTMENT: Department = "safety"

# Which team owns which kind of message. A map, not a chain of ifs: adding a
# seventh intent is a one-line change that the type checker forces you to
# make (the dict is exhaustively typed over `Intent`).
DEPARTMENT_BY_INTENT: dict[Intent, Department] = {
    "complaint": "quality_assurance",
    "price_inquiry": "sales",
    "support_request": "technical_support",
    "praise": "customer_relations",
    "order_status": "logistics",
    "return_refund": "returns",
}

# Words that describe a situation where a delayed response is a safety
# problem, not a service-quality problem. Presence of any of these overrides
# the model entirely -- see `decide()`.
#
# Deliberately a small, auditable, hand-curated list rather than something
# learned: a safety escalation rule must be reviewable by a non-engineer,
# and must not change silently when the model is retrained.
URGENCY_TERMS: frozenset[str] = frozenset(
    {
        "حريق",
        "حرائق",
        "تسرب",
        "غاز",
        "دخان",
        "انفجار",
        "اختناق",
        "صعقة",
        "كهربائي",
        "خطر",
        "طوارئ",
        "اسعاف",
        "إسعاف",
        "اصابة",
        "إصابة",
        "نزيف",
        "سام",
        "احتراق",
        "شرارة",
    }
)

# Arabic diacritics (harakat) and the tatweel elongation character. Both are
# presentational: "حريق" and "حَـــريق" are the same word to a reader, so the
# urgency scan must see them as the same word too. Stripping them here is
# what makes the invariance behavioural test pass for real rather than by
# luck.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_NON_WORD = re.compile(r"[^\w؀-ۿ]+")


class PolicyThresholds(BaseModel):
    """The two numbers that separate the three outcomes.

    Injected rather than hard-coded so operations can tighten or loosen the
    automation boundary without a code change -- and so tests can pin exact
    values instead of depending on whatever production happens to run.
    """

    model_config = ConfigDict(frozen=True)

    # At or above this, the service acts on the model's answer by itself.
    auto_route_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    # Below this, the model is guessing; the message is not worth a human's
    # time either, so it is rejected back to the sender for clarification.
    reject_floor: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _reject_inverted_thresholds(self) -> "PolicyThresholds":
        if self.reject_floor > self.auto_route_floor:
            raise ValueError(
                f"reject_floor ({self.reject_floor}) must not exceed "
                f"auto_route_floor ({self.auto_route_floor}); otherwise no "
                "confidence value could ever produce a human_review outcome."
            )
        return self


class TriageDecision(BaseModel):
    """What the service decided, and enough context to defend the decision.

    `reason` exists because an operator looking at a rejected message needs
    to know whether the model was unsure or the policy escalated -- a bare
    label is not auditable.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    action: Action
    department: Department
    priority: Priority
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    urgency_signals: list[str]
    reason: str
    model_version: str


def normalise(text: str) -> str:
    """Fold away everything that changes how a message looks but not what it
    means: Unicode composition, diacritics, tatweel, and punctuation.

    Used only for the urgency scan. The model gets the raw text -- the
    trained TF-IDF pipeline has its own preprocessing and second-guessing it
    here would reintroduce train/serve skew.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = _DIACRITICS.sub("", folded)
    return _NON_WORD.sub(" ", folded).strip()


def detect_urgency(text: str) -> list[str]:
    """Return every urgency term present, in a stable (sorted) order.

    Sorted rather than in-text order so the golden file records a canonical
    value: reordering the same two words in a sentence must not produce a
    different recorded output.
    """
    tokens = set(normalise(text).split())
    return sorted(tokens & URGENCY_TERMS)


def decide(
    *,
    text: str,
    intent: Intent,
    confidence: float,
    model_version: str,
    thresholds: PolicyThresholds,
) -> TriageDecision:
    """Map (message, model output) onto exactly one of three actions.

    The ordering of the branches below IS the policy, and it is deliberate:

    1. Urgency wins over everything, including a low-confidence model. A
       message mentioning a gas leak is escalated even if the classifier has
       no idea what it is about -- the cost of a false escalation is one
       wasted human minute; the cost of a missed one is not comparable. This
       is the invariant the directional behavioural test pins down: adding an
       urgency term can only ever raise the priority, never lower it, and can
       never turn an actionable message into a rejection.
    2. Confident enough -> act automatically.
    3. Unsure but plausible -> a human decides.
    4. Below even that -> reject, with a reason the sender can act on.
    """
    signals = detect_urgency(text)
    department = DEPARTMENT_BY_INTENT[intent]

    if signals:
        # The department is overridden too, not just the priority.
        #
        # This is the lesson of a real failure. Escalating on urgency while
        # still taking the DEPARTMENT from the model means a message the
        # classifier misread goes to the wrong team -- urgently. Observed:
        # "في تسرب غاز من السخان" scored 0.37 as `praise`, so it was correctly
        # marked urgent and then correctly sent to... customer relations.
        #
        # An escalation the model does not understand is exactly the case where
        # its opinion is worth least. So when the policy overrides the model, it
        # overrides it completely: urgent, auto-routed, and to the team that
        # handles emergencies.
        return TriageDecision(
            action="auto_route",
            department=SAFETY_DEPARTMENT,
            priority="urgent",
            intent=intent,
            confidence=confidence,
            urgency_signals=signals,
            reason=(
                "safety_escalation: message contains urgency terms "
                f"({', '.join(signals)}); routed to {SAFETY_DEPARTMENT} at "
                "urgent priority, overriding both the model's department "
                f"({department}) and its confidence ({confidence:.3f})"
            ),
            model_version=model_version,
        )

    if confidence >= thresholds.auto_route_floor:
        action: Action = "auto_route"
        reason = (
            f"confident_classification: confidence {confidence:.3f} >= "
            f"auto_route_floor {thresholds.auto_route_floor:.3f}"
        )
    elif confidence >= thresholds.reject_floor:
        action = "human_review"
        reason = (
            f"uncertain_classification: confidence {confidence:.3f} is below "
            f"auto_route_floor {thresholds.auto_route_floor:.3f}; a human "
            "confirms the department before routing"
        )
    else:
        action = "reject"
        reason = (
            f"unintelligible: confidence {confidence:.3f} is below "
            f"reject_floor {thresholds.reject_floor:.3f}; the message carries "
            "no recognisable intent and is returned for clarification"
        )

    return TriageDecision(
        action=action,
        department=department,
        priority="normal",
        intent=intent,
        confidence=confidence,
        urgency_signals=[],
        reason=reason,
        model_version=model_version,
    )
