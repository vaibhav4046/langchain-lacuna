# langchain-lacuna

A LangChain retriever and an abstaining LangGraph agent for [Lacuna](https://github.com/vaibhav4046/lacuna),
agent memory with temporal recall, exact provenance, contradiction handling and explicit abstention.

Most retrievers return the nearest text and leave the model to decide whether it answers the question.
Lacuna resolves the question first: a claim that was corrected, withdrawn or contradicted, or a fact
that was never stated, comes back as an abstention with a reason, not as a similar-looking sentence.
This package keeps that property inside LangChain:

- **`LacunaClient`**: one MCP session with a Lacuna server, over Streamable HTTP or stdio, returning
  Lacuna's answer envelope as typed dataclasses (`LacunaAnswer`, `Evidence`).
- **`LacunaRetriever`**: a `BaseRetriever`. Each `Document` is one quotation with its provenance in
  `metadata` (claim id, span id, session, message, speaker role, timestamp, standing, superseded
  claims). When Lacuna abstains it returns **no documents** by default, so a RAG chain has nothing to
  build a wrong answer from.
- **`create_abstaining_agent(model, client)`**: a LangGraph agent. The model turns a sentence into a
  lookup; Lacuna decides whether there is an answer. If there is, the model phrases it and the agent
  appends citations; if not, the reply is written from Lacuna's reason code and quotations, without the
  model.

## Install

Not yet on PyPI. From GitHub:

```bash
pip install "langchain-lacuna @ git+https://github.com/vaibhav4046/langchain-lacuna"
```

Python 3.10+.

## Quickstart (offline, no API keys, no database)

Lacuna's repository contains a recorded snapshot of its demo graph. This package ships a small
launcher, [`scripts/lacuna-snapshot-mcp.mts`](scripts/lacuna-snapshot-mcp.mts), that runs **Lacuna's
own MCP server code** over that snapshot (the same wiring as Lacuna's `scripts/mcp.ts` and
`src/snapshot/serve.ts`). You need Node 20.11+.

```bash
git clone https://github.com/vaibhav4046/lacuna && (cd lacuna && npm ci)
git clone https://github.com/vaibhav4046/langchain-lacuna && cd langchain-lacuna
uv sync
LACUNA_DIR=../lacuna uv run python examples/abstaining_agent.py
```

The model in the example is scripted (it plans the lookup and phrases the answer), so no key is
needed; the answers, abstentions and quotations come from Lacuna. Output from a run on 2026-11-21:

```text
Q: Who is the beta partner for Bellwether?
A: Bellwether's beta partner is Halverd.
Sources:
- "Correction on Bellwether: the beta partner is now Halverd." (Data planning call, 2025-04-20T11:48:00.000Z, current)

Q: Who is our contact for the vendor behind replay-queue?
A: Our contact for replay-queue's vendor is Farah Haddad.
Sources:
- "replay-queue is supplied by Northfold." (Growth operations check-in, 2025-05-04T12:36:00.000Z, current)
- "Our contact at Northfold is Farah Haddad." (Platform operations check-in, 2025-05-06T11:18:00.000Z, current)

Q: When does Junco launch?
A: I don't know the launch_date of Junco: it was stated and later withdrawn, and nothing has replaced it.
What the record does say:
- "Junco is scheduled to launch on 13 May 2026." (Mobile roadmap catch-up, 2025-02-17T11:42:00.000Z, superseded)
- "We have pulled the launch date for Junco and there is no date on it now." (Growth handover notes, 2025-02-19T10:06:00.000Z, withdrawal_current)

Q: What budget code does notify-relay bill to?
A: I don't know the budget_code of notify-relay: the sources disagree and nothing resolves which one is right.
What the record does say:
- "notify-relay bills to BC-9699." (Trust incident follow-up, 2025-04-14T10:24:00.000Z, current_conflicting)
- "notify-relay bills to BC-6530." (Mobile roadmap catch-up, 2025-04-16T09:48:00.000Z, current_conflicting)

Q: When is the migration window for Meridian?
A: I don't know the migration_window of Meridian: nothing in the record ever states it.
```

The example uses a scripted model. No hosted model has been run against this package; any chat model
that supports tool calling can be passed to `create_abstaining_agent`, but that is untested here.

## Use the retriever

```python
from langchain_lacuna import LacunaClient, LacunaRetriever

async with LacunaClient.snapshot("../lacuna") as lacuna:  # or LacunaClient.http(url), LacunaClient.stdio(dir)
    retriever = LacunaRetriever(client=lacuna)
    docs = await retriever.ainvoke("Bellwether | beta_partner")
    docs[0].page_content  # 'Correction on Bellwether: the beta partner is now Halverd.'
    docs[0].metadata["answer"]  # 'Halverd'
    docs[0].metadata["superseded_claims"]  # the two earlier partners' claim ids

    await retriever.ainvoke("notify-relay | budget_code")  # [] : contradicted, so nothing to cite
    # include_abstained_evidence=True returns those quotes, marked status="abstained", reason_code=...
```

Queries are `"subject | predicate"` or `"subject | predicate | via"` (one hop). Lacuna answers exact
questions, not free text; the agent below is the way to go from a sentence to that form. For the
full envelope, call `await lacuna.ask(subject, predicate, via)`.

**Which server?**

| Constructor | Server | Notes |
| --- | --- | --- |
| `LacunaClient.snapshot(lacuna_dir)` | Lacuna's MCP server over the checked-in snapshot | Offline. Source checkout of this package only (the launcher is in `scripts/`). |
| `LacunaClient.stdio(lacuna_dir)` | `npm run mcp -- --stdio` in a Lacuna checkout | Needs Lacuna's `.env.local` and a seeded HydraDB store. Not exercised by this package's tests. |
| `LacunaClient.http(url)` | Any Lacuna Streamable HTTP endpoint | Defaults to the public `https://lacuna-five.vercel.app/mcp`. On 2026-11-21 that endpoint answered MCP but its HydraDB backend returned `402 FREE_PLAN_DEPRECATED`, so live reads failed; not exercised by the tests. |

## Design notes

- **Lacuna decides, not the model.** The agent's routing is on Lacuna's `status`. The answer node only
  runs when Lacuna answered; the abstain node never calls the model.
- **Grounding check.** If the model's phrasing does not contain Lacuna's answer string, it is replaced
  with a plain sentence that does (tested with a model that says the superseded partner instead).
- **Abstention is a result.** `never_stated`, `retracted`, `contradicted`, `unconnected` and
  `out_of_scope` each produce a specific sentence, followed by what the record does say.
- **Errors are not abstentions.** A failed read raises `LacunaError` from the client and retriever; the
  agent turns it into "Lacuna could not be read" rather than pretending the fact is unknown.
- **Quotations are data.** They are passed to the model as quoted text for phrasing, never as
  instructions; Lacuna's own server says the same.

## Tests

```bash
uv run pytest -m "not integration"                  # 17 tests: fake Lacuna, scripted model
LACUNA_DIR=../lacuna uv run pytest -m integration   # Lacuna's MCP server over its snapshot
uv run ruff check . && uv run mypy                  # mypy --strict on the package
```

The integration test checks seven of Lacuna's demo questions end to end (two answers, a one-hop
answer, retracted, contradicted, unconnected, never stated) through the client, the retriever and the
agent. CI runs everything above, the example and a package build on Python 3.10 and 3.13, against a
pinned Lacuna commit.

## Limitations

- **Async only** (`ainvoke`); the MCP session belongs to one event loop.
- **Structured queries.** The retriever does not parse sentences. Lacuna's own `lacuna_read_question`
  tool does, but over the snapshot store it returned `no_subject` for every question tried, so this
  package does not depend on it.
- **Unknown subjects over the snapshot** fail as a read error (`HydraTransportError`) rather than the
  abstention, because the snapshot only holds recorded replies.
- **Only `lacuna_ask` is wrapped.** `lacuna_explain`, `lacuna_timeline`, `search` and `fetch` are not.
- **No hosted model has been run** against this agent; the example and tests use a scripted model.
- Not published to PyPI yet.

## License

Apache-2.0, the same as Lacuna.
