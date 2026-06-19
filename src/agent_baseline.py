from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        # Calculate prompt tokens processed
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        # Live path
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history = []
        for m in session.messages[:-1]:
            if m["role"] == "user":
                chat_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                chat_history.append(AIMessage(content=m["content"]))

        try:
            response = self.langchain_agent.invoke(chat_history + [HumanMessage(content=message)])
            reply_content = response.content
        except Exception as e:
            # Fallback to offline if LLM invocation fails
            reply_content = f"Error calling live LLM: {str(e)}. Fallback offline: {message[:30]}"

        session.messages.append({"role": "assistant", "content": reply_content})

        # Track response tokens
        reply_tokens = estimate_tokens(reply_content)
        session.token_usage += reply_tokens

        return {"content": reply_content, "role": "assistant"}

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        # Calculate prompt tokens processed
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        # Generate a short deterministic reply
        reply_content = f"Tôi là Baseline Agent. Nhận được: {message[:30]}..."
        session.messages.append({"role": "assistant", "content": reply_content})

        # Update token counts
        reply_tokens = estimate_tokens(reply_content)
        session.token_usage += reply_tokens

        return {"content": reply_content, "role": "assistant"}

    def _maybe_build_langchain_agent(self):
        """Optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
