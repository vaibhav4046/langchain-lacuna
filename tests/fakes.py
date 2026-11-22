"""Envelopes shaped like Lacuna's `lacuna_ask` structured output (taken from real snapshot answers)."""

from __future__ import annotations

from typing import Any

from langchain_lacuna import LacunaAnswer, LacunaError


def ev(quote: str, claim: int, standing: str, title: str = "Data planning call") -> dict[str, Any]:
    return {
        "spanId": claim + 1,
        "claimId": claim,
        "standing": standing,
        "quote": quote,
        "sessionId": 99,
        "sessionTitle": title,
        "messageId": 7,
        "role": "user",
        "ts": "2025-04-20T11:48:00.000Z",
    }


ENVELOPES: dict[tuple[str, str, str | None], dict[str, Any]] = {
    ("Bellwether", "beta_partner", None): {
        "status": "answered",
        "answer": "Halverd",
        "reasonCode": None,
        "claimId": 797,
        "supersededClaims": [247, 224],
        "evidence": [ev("Correction on Bellwether: the beta partner is now Halverd.", 797, "current")],
        "evidenceTotal": 1,
    },
    ("notify-relay", "budget_code", None): {
        "status": "abstained",
        "answer": None,
        "reasonCode": "contradicted",
        "claimId": None,
        "supersededClaims": [],
        "evidence": [
            ev("notify-relay bills to BC-9699.", 360, "current_conflicting", "Trust incident follow-up"),
            ev("notify-relay bills to BC-6530.", 363, "current_conflicting", "Mobile roadmap catch-up"),
        ],
        "evidenceTotal": 2,
    },
    ("Meridian", "migration_window", None): {
        "status": "abstained",
        "answer": None,
        "reasonCode": "never_stated",
        "claimId": None,
        "supersededClaims": [],
        "evidence": [],
        "evidenceTotal": 0,
    },
}


class FakeLacuna:
    def __init__(self) -> None:
        self.asked: list[tuple[str, str, str | None]] = []

    async def ask(self, subject: str, predicate: str, via: str | None = None) -> LacunaAnswer:
        self.asked.append((subject, predicate, via))
        raw = ENVELOPES.get((subject, predicate, via))
        if raw is None:
            raise LacunaError("MCP error -32603: the graph did not answer this read (HydraTransportError)")
        return LacunaAnswer.from_json(subject, predicate, via, raw)
