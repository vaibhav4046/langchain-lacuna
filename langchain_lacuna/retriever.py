"""A LangChain retriever over Lacuna that keeps provenance and never hands over evidence for a non-answer."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from langchain_lacuna.client import LacunaAnswer, LacunaAsker

SEPARATOR = "|"


def parse_query(query: str) -> tuple[str, str, str | None]:
    """`"subject | predicate"` or `"subject | predicate | via"` -> (subject, predicate, via).

    Lacuna answers exact (subject, predicate) questions, not free text, so the retriever takes them
    in this form. An agent can produce it from a sentence; see `create_abstaining_agent`.
    """
    parts = [p.strip() for p in query.split(SEPARATOR)]
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(
            f'LacunaRetriever expects "subject {SEPARATOR} predicate" or "subject {SEPARATOR} predicate '
            f'{SEPARATOR} via", for example "Bellwether | beta_partner". Got: {query[:80]!r}'
        )
    return parts[0], parts[1], parts[2] if len(parts) == 3 else None


def to_documents(answer: LacunaAnswer) -> list[Document]:
    """One Document per quotation, each carrying where it came from and how Lacuna resolved the question."""
    shared: dict[str, Any] = {
        "source": "lacuna",
        "subject": answer.subject,
        "predicate": answer.predicate,
        "via": answer.via,
        "status": answer.status,
        "answer": answer.answer,
        "reason_code": answer.reason_code,
        "answer_claim_id": answer.claim_id,
        "superseded_claims": list(answer.superseded_claims),
        "evidence_total": answer.evidence_total,
    }
    return [
        Document(
            page_content=e.quote,
            id=None if e.span_id is None else f"lacuna-span-{e.span_id}",
            metadata={
                **shared,
                "claim_id": e.claim_id,
                "span_id": e.span_id,
                "standing": e.standing,
                "session_id": e.session_id,
                "session_title": e.session_title,
                "message_id": e.message_id,
                "role": e.role,
                "ts": e.ts,
            },
        )
        for e in answer.evidence
    ]


class LacunaRetriever(BaseRetriever):
    """Retrieves the quotations that support Lacuna's answer to `"subject | predicate [| via]"`.

    When Lacuna abstains (the fact was never stated, was retracted, is contradicted, or the hop is
    unconnected) it returns no documents by default, so a downstream chain has nothing to build a
    wrong answer from. Set `include_abstained_evidence=True` to get those quotations too, each marked
    with `status="abstained"`, its `reason_code` and its `standing`.

    Async only: the MCP session lives on one event loop, so use `ainvoke`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(exclude=True, description="A LacunaAsker, usually a connected LacunaClient.")
    include_abstained_evidence: bool = False

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError("LacunaRetriever is async only; call it with `ainvoke`.")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        subject, predicate, via = parse_query(query)
        asker: LacunaAsker = self.client
        answer = await asker.ask(subject, predicate, via)
        if not answer.answered and not self.include_abstained_evidence:
            return []
        return to_documents(answer)
