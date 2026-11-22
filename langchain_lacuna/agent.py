"""A LangGraph agent that answers from Lacuna with citations, and abstains when the evidence does not hold."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, ValidationError

from langchain_lacuna.client import LacunaAnswer, LacunaAsker, LacunaError

PLAN_PROMPT = (
    "Turn the user's question into one Lacuna lookup by calling LacunaLookup. subject is the entity named in the "
    "question, predicate is the property asked for in snake_case (for example launch_date, beta_partner, "
    "budget_code, contact), and via is set only when the property belongs to something the subject points to "
    "(for example 'the contact for the vendor behind X' is subject X, predicate contact, via vendor). "
    "If the question is not a lookup of one fact, do not call the tool."
)
ANSWER_PROMPT = (
    "Answer the user's question in one or two sentences using only the quotations below. State the answer "
    "exactly as given. Do not add facts that are not in the quotations."
)

WHY = {
    "never_stated": "nothing in the record ever states it",
    "retracted": "it was stated and later withdrawn, and nothing has replaced it",
    "contradicted": "the sources disagree and nothing resolves which one is right",
    "unconnected": "the link that question depends on is not in the record",
    "out_of_scope": "the record does not cover that subject",
}


class LacunaLookup(BaseModel):
    """Look up one fact in Lacuna."""

    subject: str = Field(min_length=1, max_length=200, description="The entity the question is about.")
    predicate: str = Field(min_length=1, max_length=200, description="The property asked for, in snake_case.")
    via: str | None = Field(default=None, max_length=200, description="A single relation to follow first, if any.")


class LacunaAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    lookup: LacunaLookup | None
    result: LacunaAnswer | None
    error: str | None


def _cite(answer: LacunaAnswer, limit: int = 3) -> str:
    lines = [
        f'- "{e.quote}" ({e.session_title or "unknown session"}, {e.ts or "undated"}, {e.standing or "?"})'
        for e in answer.evidence[:limit]
    ]
    return "\n".join(lines)


def abstention_text(state: LacunaAgentState) -> str:
    """The reply when the agent will not answer. Written from Lacuna's envelope, never by the model."""
    result, lookup = state.get("result"), state.get("lookup")
    if lookup is None:
        return "I can't answer that: it isn't a lookup of one fact I can check in Lacuna."
    asked = f"{lookup.predicate} of {lookup.subject}" + (f" via {lookup.via}" if lookup.via else "")
    if result is None:
        return f"I can't answer that: Lacuna could not be read for {asked} ({state.get('error') or 'unknown error'})."
    why = WHY.get(result.reason_code or "", f"Lacuna abstained ({result.reason_code})")
    text = f"I don't know the {asked}: {why}."
    if result.evidence:
        text += "\nWhat the record does say:\n" + _cite(result)
    return text


def create_abstaining_agent(
    model: BaseChatModel, client: LacunaAsker
) -> CompiledStateGraph[LacunaAgentState, Any, Any, Any]:
    """plan (model picks the lookup) -> retrieve (Lacuna) -> answer (model, with citations) or abstain (no model).

    The model never decides whether there is an answer: Lacuna does. On abstention the reply is built
    from Lacuna's reason code and quotations, and an answer the model writes that does not contain
    Lacuna's value is replaced with a plain one, so the agent cannot state something the record does not.
    """
    planner = model.bind_tools([LacunaLookup], tool_choice="auto")

    async def plan(state: LacunaAgentState) -> dict[str, Any]:
        reply = await planner.ainvoke([SystemMessage(PLAN_PROMPT), *state["messages"]])
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            return {"lookup": None}
        try:
            return {"lookup": LacunaLookup.model_validate(calls[0]["args"])}
        except ValidationError:
            return {"lookup": None}

    async def retrieve(state: LacunaAgentState) -> dict[str, Any]:
        lookup = state["lookup"]
        assert lookup is not None
        try:
            return {"result": await client.ask(lookup.subject, lookup.predicate, lookup.via), "error": None}
        except (LacunaError, OSError) as error:
            return {"result": None, "error": f"{type(error).__name__}: {error}"}

    async def answer(state: LacunaAgentState) -> dict[str, Any]:
        result = state["result"]
        assert result is not None and result.answer is not None
        question = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        quotes = "\n".join(f'"{e.quote}"' for e in result.evidence)
        reply = await model.ainvoke(
            [
                SystemMessage(f"{ANSWER_PROMPT}\n\nLacuna's answer: {result.answer}\nQuotations:\n{quotes}"),
                HumanMessage(question.text if question else ""),
            ]
        )
        text = reply.text.strip()
        if result.answer.lower() not in text.lower():
            text = f"The {result.predicate} of {result.subject} is {result.answer}."
        return {"messages": [AIMessage(f"{text}\nSources:\n{_cite(result)}")]}

    async def abstain(state: LacunaAgentState) -> dict[str, Any]:
        return {"messages": [AIMessage(abstention_text(state))]}

    graph = StateGraph(LacunaAgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("abstain", abstain)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "retrieve" if s.get("lookup") else "abstain", ["retrieve", "abstain"])
    graph.add_conditional_edges(
        "retrieve",
        lambda s: "answer" if (r := s.get("result")) is not None and r.answered and r.answer else "abstain",
        ["answer", "abstain"],
    )
    graph.add_edge("answer", END)
    graph.add_edge("abstain", END)
    return graph.compile()
