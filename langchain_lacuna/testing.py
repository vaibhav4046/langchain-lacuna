"""A scripted chat model, so the example agent and tests run offline with no API key."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Replies with the given AI messages in order, ignoring its input. Tool calls are part of the script."""

    responses: list[AIMessage]
    _turn: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._turn >= len(self.responses):
            raise RuntimeError("ScriptedChatModel ran out of scripted responses")
        reply = self.responses[self._turn]
        self._turn += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self
