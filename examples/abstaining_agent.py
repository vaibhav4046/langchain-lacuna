"""End-to-end demo: a LangGraph agent answers from Lacuna with citations, and abstains when it should.

    LACUNA_DIR=/path/to/lacuna python examples/abstaining_agent.py                 # offline, scripted model
    LACUNA_DIR=/path/to/lacuna python examples/abstaining_agent.py --model gemini  # GOOGLE_API_KEY + [gemini]

LACUNA_DIR is a Lacuna checkout with `npm ci` done. The server is Lacuna's own MCP server reading the
graph snapshot checked into that repository: no database, no token, no network.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_lacuna import LacunaClient, create_abstaining_agent
from langchain_lacuna.testing import ScriptedChatModel

# (question, the lookup a model should plan, what a model would write if Lacuna answers)
QUESTIONS: list[tuple[str, dict[str, str], str]] = [
    ("Who is the beta partner for Bellwether?", {"subject": "Bellwether", "predicate": "beta_partner"},
     "Bellwether's beta partner is Halverd."),
    ("Who is our contact for the vendor behind replay-queue?",
     {"subject": "replay-queue", "predicate": "contact", "via": "vendor"},
     "Our contact for replay-queue's vendor is Farah Haddad."),
    ("When does Junco launch?", {"subject": "Junco", "predicate": "launch_date"}, ""),
    ("What budget code does notify-relay bill to?", {"subject": "notify-relay", "predicate": "budget_code"}, ""),
    ("When is the migration window for Meridian?", {"subject": "Meridian", "predicate": "migration_window"}, ""),
]  # fmt: skip


def scripted(lookup: dict[str, str], prose: str) -> ScriptedChatModel:
    plan = AIMessage("", tool_calls=[{"name": "LacunaLookup", "args": lookup, "id": "l1", "type": "tool_call"}])
    return ScriptedChatModel(responses=[plan, AIMessage(prose)] if prose else [plan])


def gemini() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI  # optional extra

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


async def main(model_name: str) -> None:
    lacuna_dir = os.environ.get("LACUNA_DIR")
    if not lacuna_dir:
        raise SystemExit("Set LACUNA_DIR to a Lacuna checkout with `npm ci` done.")
    async with LacunaClient.snapshot(lacuna_dir) as lacuna:
        for question, lookup, prose in QUESTIONS:
            model = gemini() if model_name == "gemini" else scripted(lookup, prose)
            result = await create_abstaining_agent(model, lacuna).ainvoke({"messages": [HumanMessage(question)]})
            print(f"Q: {question}\nA: {result['messages'][-1].text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["scripted", "gemini"], default="scripted")
    asyncio.run(main(parser.parse_args().model))
