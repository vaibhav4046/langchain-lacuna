"""Against Lacuna's own MCP server reading the graph snapshot in the Lacuna repository.

Needs LACUNA_DIR pointing at a Lacuna checkout with `npm ci` done (CI pins the commit).
"""

from __future__ import annotations

import os
import shutil

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_lacuna import LacunaClient, LacunaRetriever, create_abstaining_agent
from langchain_lacuna.testing import ScriptedChatModel

LACUNA_DIR = os.environ.get("LACUNA_DIR")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not LACUNA_DIR or shutil.which("node") is None, reason="LACUNA_DIR or node missing"),
]

# Lacuna's own demo questions over its snapshot (src/server/examples.ts), covering each outcome.
CASES = [
    ("Meridian", "launch_date", None, "answered", None),
    ("Bellwether", "beta_partner", None, "answered", None),
    ("Junco", "launch_date", None, "abstained", "retracted"),
    ("notify-relay", "budget_code", None, "abstained", "contradicted"),
    ("replay-queue", "contact", "vendor", "answered", None),
    ("Meridian", "contact", "vendor", "abstained", "unconnected"),
    ("Meridian", "migration_window", None, "abstained", "never_stated"),
]


async def test_outcomes_through_retriever_and_agent() -> None:
    assert LACUNA_DIR
    async with LacunaClient.snapshot(LACUNA_DIR) as lacuna:
        assert "lacuna_ask" in await lacuna.list_tool_names()
        retriever = LacunaRetriever(client=lacuna)
        for subject, predicate, via, status, reason in CASES:
            answer = await lacuna.ask(subject, predicate, via)
            assert (answer.status, answer.reason_code) == (status, reason), (subject, predicate, via)
            query = " | ".join(p for p in (subject, predicate, via) if p)
            docs = await retriever.ainvoke(query)
            if status == "answered":
                assert docs and all(d.metadata["claim_id"] and d.metadata["session_title"] for d in docs)
            else:
                assert docs == []

        bellwether = await lacuna.ask("Bellwether", "beta_partner")
        assert bellwether.answer == "Halverd" and len(bellwether.superseded_claims) == 2

        call = {"name": "LacunaLookup", "args": {"subject": "notify-relay", "predicate": "budget_code"}}
        model = ScriptedChatModel(responses=[AIMessage("", tool_calls=[{**call, "id": "x", "type": "tool_call"}])])
        agent = create_abstaining_agent(model, lacuna)
        result = await agent.ainvoke({"messages": [HumanMessage("What budget code does notify-relay bill to?")]})
        reply = result["messages"][-1].text
        assert "the sources disagree" in reply and "BC-9699" in reply and "BC-6530" in reply
