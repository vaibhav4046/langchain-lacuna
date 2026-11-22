"""A long-lived MCP session with a Lacuna server, and the answer envelope it returns."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol

from mcp import Client, StdioServerParameters
from mcp.types import TextContent

PUBLIC_ENDPOINT = "https://lacuna-five.vercel.app/mcp"
SNAPSHOT_SERVER = Path(__file__).resolve().parent.parent / "scripts" / "lacuna-snapshot-mcp.mts"

ReasonCode = Literal["never_stated", "retracted", "contradicted", "unconnected", "out_of_scope"]


class LacunaError(RuntimeError):
    """Lacuna could not be read (a tool failure or a protocol error), as opposed to an abstention."""


@dataclass(frozen=True)
class Evidence:
    """One quotation supporting (or, when abstaining, explaining) an answer."""

    quote: str
    claim_id: int | None
    span_id: int | None
    standing: str | None
    session_id: int | None
    session_title: str | None
    message_id: int | None
    role: str | None
    ts: str | None

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Evidence:
        return cls(
            quote=str(raw.get("quote", "")),
            claim_id=raw.get("claimId"),
            span_id=raw.get("spanId"),
            standing=raw.get("standing"),
            session_id=raw.get("sessionId"),
            session_title=raw.get("sessionTitle"),
            message_id=raw.get("messageId"),
            role=raw.get("role"),
            ts=raw.get("ts"),
        )


@dataclass(frozen=True)
class LacunaAnswer:
    """Lacuna's envelope for one (subject, predicate, via) question. Status is answered or abstained, never both."""

    subject: str
    predicate: str
    via: str | None
    status: Literal["answered", "abstained"]
    answer: str | None
    reason_code: ReasonCode | None
    claim_id: int | None
    superseded_claims: tuple[int, ...]
    evidence: tuple[Evidence, ...]
    evidence_total: int
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    @classmethod
    def from_json(cls, subject: str, predicate: str, via: str | None, raw: Mapping[str, Any]) -> LacunaAnswer:
        status = raw.get("status")
        if status not in ("answered", "abstained"):
            raise LacunaError(f"Lacuna returned an unexpected status: {status!r}")
        evidence = tuple(Evidence.from_json(e) for e in raw.get("evidence") or [])
        return cls(
            subject=subject,
            predicate=predicate,
            via=via,
            status=status,
            answer=raw.get("answer"),
            reason_code=raw.get("reasonCode"),
            claim_id=raw.get("claimId"),
            superseded_claims=tuple(raw.get("supersededClaims") or ()),
            evidence=evidence,
            evidence_total=int(raw.get("evidenceTotal", len(evidence))),
            raw=dict(raw),
        )


class LacunaAsker(Protocol):
    """Anything that can answer a Lacuna question. `LacunaClient` is the real one; tests use fakes."""

    async def ask(self, subject: str, predicate: str, via: str | None = None) -> LacunaAnswer: ...


class LacunaClient:
    """One MCP session with a Lacuna server, over stdio or Streamable HTTP.

    ```python
    async with LacunaClient.snapshot("/path/to/lacuna") as lacuna:
        answer = await lacuna.ask("Bellwether", "beta_partner")
    ```
    """

    def __init__(self, server: str | StdioServerParameters) -> None:
        self._server = server
        self._client: Client | None = None

    @classmethod
    def http(cls, url: str = PUBLIC_ENDPOINT) -> LacunaClient:
        """A Lacuna MCP endpoint over Streamable HTTP, such as the public read-only workspace."""
        return cls(url)

    @classmethod
    def stdio(cls, lacuna_dir: str | Path, env: Mapping[str, str] | None = None) -> LacunaClient:
        """Lacuna's own stdio server (`npm run mcp -- --stdio`) in a checkout configured with `.env.local`."""
        npm = shutil.which("npm") or "npm"
        return cls(
            StdioServerParameters(
                command=npm, args=["run", "--silent", "mcp", "--", "--stdio"], cwd=str(lacuna_dir), env=dict(env or {})
            )
        )

    @classmethod
    def snapshot(cls, lacuna_dir: str | Path) -> LacunaClient:
        """Lacuna's MCP server reading the graph snapshot checked into the Lacuna repository: no database,
        no token, no network. Needs a Lacuna checkout with `npm ci` done. Only available from a source
        checkout of this package (the launcher lives in `scripts/`)."""
        if not SNAPSHOT_SERVER.exists():
            raise FileNotFoundError(f"snapshot launcher not found at {SNAPSHOT_SERVER}")
        node = shutil.which("node") or "node"
        root = str(Path(lacuna_dir).resolve())
        return cls(
            StdioServerParameters(
                command=node, args=["--import", "tsx", str(SNAPSHOT_SERVER)], cwd=root, env={"LACUNA_DIR": root}
            )
        )

    async def __aenter__(self) -> LacunaClient:
        client = Client(self._server)
        await client.__aenter__()
        self._client = client
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(exc_type, exc, tb)

    async def list_tool_names(self) -> list[str]:
        return [tool.name for tool in (await self._session().list_tools()).tools]

    async def ask(self, subject: str, predicate: str, via: str | None = None) -> LacunaAnswer:
        arguments: dict[str, Any] = {"subject": subject, "predicate": predicate}
        if via:
            arguments["via"] = via
        result = await self._session().call_tool("lacuna_ask", arguments)
        if result.is_error:
            text = " ".join(b.text for b in result.content if isinstance(b, TextContent))
            raise LacunaError(text or "lacuna_ask failed")
        if not isinstance(result.structured_content, dict):
            raise LacunaError("lacuna_ask returned no structured content")
        return LacunaAnswer.from_json(subject, predicate, via, result.structured_content)

    def _session(self) -> Client:
        if self._client is None:
            raise RuntimeError("LacunaClient is not connected; use `async with LacunaClient...`.")
        return self._client
