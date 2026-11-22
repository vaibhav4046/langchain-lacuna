from __future__ import annotations

from typing import Any

import pytest
from fakes import FakeLacuna
from langchain_core.messages import AIMessage, HumanMessage

from langchain_lacuna import LacunaAnswer, LacunaError, LacunaRetriever, create_abstaining_agent, parse_query
from langchain_lacuna.testing import ScriptedChatModel


def lookup(subject: str, predicate: str, via: str | None = None) -> AIMessage:
    args: dict[str, Any] = {"subject": subject, "predicate": predicate, "via": via}
    return AIMessage("", tool_calls=[{"name": "LacunaLookup", "args": args, "id": "l1", "type": "tool_call"}])


# --- parse_query -------------------------------------------------------------


def test_parse_query_forms() -> None:
    assert parse_query("Bellwether | beta_partner") == ("Bellwether", "beta_partner", None)
    assert parse_query(" replay-queue|contact | vendor ") == ("replay-queue", "contact", "vendor")


@pytest.mark.parametrize("bad", ["who is the beta partner?", "a | ", "a | b | c | d", "| b"])
def test_parse_query_rejects_free_text(bad: str) -> None:
    with pytest.raises(ValueError, match="subject"):
        parse_query(bad)


# --- retriever ---------------------------------------------------------------


async def test_answered_question_returns_quotes_with_provenance() -> None:
    [doc] = await LacunaRetriever(client=FakeLacuna()).ainvoke("Bellwether | beta_partner")
    assert doc.page_content == "Correction on Bellwether: the beta partner is now Halverd."
    assert doc.id == "lacuna-span-798"
    m = doc.metadata
    assert m["source"] == "lacuna" and m["status"] == "answered" and m["answer"] == "Halverd"
    assert m["claim_id"] == 797 and m["answer_claim_id"] == 797 and m["standing"] == "current"
    assert m["superseded_claims"] == [247, 224]
    assert m["session_title"] == "Data planning call" and m["ts"] == "2025-04-20T11:48:00.000Z"


async def test_abstention_returns_no_documents_by_default() -> None:
    retriever = LacunaRetriever(client=FakeLacuna())
    assert await retriever.ainvoke("notify-relay | budget_code") == []
    assert await retriever.ainvoke("Meridian | migration_window") == []


async def test_abstained_evidence_is_opt_in_and_marked() -> None:
    retriever = LacunaRetriever(client=FakeLacuna(), include_abstained_evidence=True)
    docs = await retriever.ainvoke("notify-relay | budget_code")
    assert [d.metadata["standing"] for d in docs] == ["current_conflicting", "current_conflicting"]
    assert {d.metadata["status"] for d in docs} == {"abstained"}
    assert {d.metadata["reason_code"] for d in docs} == {"contradicted"}


async def test_retriever_is_async_only_and_errors_propagate() -> None:
    retriever = LacunaRetriever(client=FakeLacuna())
    with pytest.raises(NotImplementedError):
        retriever.invoke("Bellwether | beta_partner")
    with pytest.raises(LacunaError):
        await retriever.ainvoke("Redshank | launch_date")


def test_envelope_rejects_unknown_status() -> None:
    with pytest.raises(LacunaError, match="unexpected status"):
        LacunaAnswer.from_json("a", "b", None, {"status": "maybe"})


# --- agent -------------------------------------------------------------------


async def run(model: ScriptedChatModel, question: str) -> tuple[dict[str, Any], FakeLacuna]:
    lacuna = FakeLacuna()
    result = await create_abstaining_agent(model, lacuna).ainvoke({"messages": [HumanMessage(question)]})
    return result, lacuna


async def test_agent_answers_with_citations() -> None:
    model = ScriptedChatModel(
        responses=[lookup("Bellwether", "beta_partner"), AIMessage("Bellwether's beta partner is Halverd.")]
    )
    result, lacuna = await run(model, "Who is the beta partner for Bellwether?")
    reply = result["messages"][-1].text
    assert reply.startswith("Bellwether's beta partner is Halverd.")
    assert "Sources:" in reply and "now Halverd" in reply and "Data planning call" in reply
    assert lacuna.asked == [("Bellwether", "beta_partner", None)]


async def test_agent_replaces_an_answer_that_contradicts_lacuna() -> None:
    model = ScriptedChatModel(
        responses=[lookup("Bellwether", "beta_partner"), AIMessage("The beta partner is Stonecrop.")]
    )
    result, _ = await run(model, "Who is the beta partner for Bellwether?")
    reply = result["messages"][-1].text
    assert "Stonecrop" not in reply
    assert reply.startswith("The beta_partner of Bellwether is Halverd.")


async def test_agent_abstains_on_contradiction_without_asking_the_model() -> None:
    # Only one scripted reply: if the agent called the model to answer, it would run out and raise.
    model = ScriptedChatModel(responses=[lookup("notify-relay", "budget_code")])
    result, _ = await run(model, "What budget code does notify-relay bill to?")
    reply = result["messages"][-1].text
    assert reply.startswith("I don't know the budget_code of notify-relay: the sources disagree")
    assert "BC-9699" in reply and "BC-6530" in reply
    assert result["result"].reason_code == "contradicted"


async def test_agent_abstains_when_never_stated() -> None:
    model = ScriptedChatModel(responses=[lookup("Meridian", "migration_window")])
    result, _ = await run(model, "When is the migration window for Meridian?")
    assert "nothing in the record ever states it" in result["messages"][-1].text


async def test_agent_abstains_when_not_a_lookup() -> None:
    model = ScriptedChatModel(responses=[AIMessage("Hello!")])
    result, lacuna = await run(model, "hi")
    assert "isn't a lookup" in result["messages"][-1].text and lacuna.asked == []


async def test_agent_abstains_on_invalid_lookup_arguments() -> None:
    model = ScriptedChatModel(responses=[lookup("", "beta_partner")])
    result, lacuna = await run(model, "?")
    assert "isn't a lookup" in result["messages"][-1].text and lacuna.asked == []


async def test_agent_abstains_when_lacuna_cannot_be_read() -> None:
    model = ScriptedChatModel(responses=[lookup("Redshank", "launch_date")])
    result, _ = await run(model, "When does Redshank launch?")
    reply = result["messages"][-1].text
    assert reply.startswith("I can't answer that: Lacuna could not be read") and "LacunaError" in reply
