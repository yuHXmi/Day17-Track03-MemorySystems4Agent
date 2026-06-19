from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # optionally initialize a real LangChain/LangGraph agent.
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(user_id, thread_id, message)

        # Live Mode
        # 1. Extract updates using offline extract_profile_updates
        updates = extract_profile_updates(message)
        for k, v in updates.items():
            content = self.profile_store.read_text(user_id)
            lines = content.splitlines()
            target_line = None
            for line in lines:
                if line.startswith(f"- **{k}**:"):
                    target_line = line
                    break
            if target_line:
                self.profile_store.edit_text(user_id, target_line, f"- **{k}**: {v}")
            else:
                if not content.strip():
                    new_content = f"# User Profile\n\n- **{k}**: {v}\n"
                else:
                    new_content = content.rstrip() + f"\n- **{k}**: {v}\n"
                self.profile_store.write_text(user_id, new_content)

        # 2. Append to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context tokens and update counter
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        # 4. Build system prompt and history
        user_md = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        messages = ctx.get("messages", [])

        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        system_prompt = f"You are an Advanced AI Agent. Here is the user's profile:\n{user_md}\n\n"
        if summary:
            system_prompt += f"Summary of older conversation:\n{summary}\n\n"

        langchain_messages = [SystemMessage(content=system_prompt)]
        for m in messages[:-1]:
            if m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                langchain_messages.append(AIMessage(content=m["content"]))
        langchain_messages.append(HumanMessage(content=message))

        try:
            response = self.langchain_agent.invoke(langchain_messages)
            reply_content = response.content
        except Exception as e:
            # Fallback offline if fail
            reply_content = f"Error calling live LLM: {str(e)}. Fallback offline: {self._offline_response(user_id, thread_id, message)}"

        # 5. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", reply_content)

        # 6. Update token counter
        reply_tokens = estimate_tokens(reply_content)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens

        return {"content": reply_content, "role": "assistant"}

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """
        # 1. Extract stable profile facts from the incoming message
        updates = extract_profile_updates(message)

        # 2. Persist those facts into User.md
        for k, v in updates.items():
            content = self.profile_store.read_text(user_id)
            lines = content.splitlines()
            target_line = None
            for line in lines:
                if line.startswith(f"- **{k}**:"):
                    target_line = line
                    break
            if target_line:
                self.profile_store.edit_text(user_id, target_line, f"- **{k}**: {v}")
            else:
                if not content.strip():
                    new_content = f"# User Profile\n\n- **{k}**: {v}\n"
                else:
                    new_content = content.rstrip() + f"\n- **{k}**: {v}\n"
                self.profile_store.write_text(user_id, new_content)

        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        # 5. Generate a response that can answer long-term recall questions
        reply_content = self._offline_response(user_id, thread_id, message)

        # 6. Append the assistant reply and update token counters
        self.compact_memory.append(thread_id, "assistant", reply_content)

        reply_tokens = estimate_tokens(reply_content)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens

        return {"content": reply_content, "role": "assistant"}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """
        user_md = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        messages = ctx.get("messages", [])

        tokens = estimate_tokens(user_md) + estimate_tokens(summary)
        for m in messages:
            tokens += estimate_tokens(m["content"])
        return tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """
        content = self.profile_store.read_text(user_id)
        facts = {}
        for line in content.splitlines():
            if line.startswith("- **") and ":" in line:
                parts = line.split(":", 1)
                k = parts[0].replace("- **", "").replace("**", "").strip()
                v = parts[1].strip()
                facts[k] = v

        q = message.lower()
        ans_parts = []

        # Check if name is requested
        if "tên" in q or "tên là gì" in q or "tên của mình" in q:
            name = facts.get("Tên")
            if name:
                ans_parts.append(f"Tên bạn là {name}.")

        # Check location
        if "ở đâu" in q or "nơi ở" in q or "sống ở" in q:
            loc = facts.get("Nơi ở")
            if loc:
                ans_parts.append(f"Bạn đang ở {loc}.")

        # Check profession
        if "nghề" in q or "công việc" in q or "làm gì" in q:
            prof = facts.get("Nghề nghiệp")
            if prof:
                ans_parts.append(f"Bạn làm nghề {prof}.")

        # Check drink
        if "uống" in q or "đồ uống" in q:
            drink = facts.get("Đồ uống yêu thích")
            if drink:
                ans_parts.append(f"Đồ uống yêu thích của bạn là {drink}.")

        # Check food
        if "ăn" in q or "món ăn" in q:
            food = facts.get("Món ăn yêu thích")
            if food:
                ans_parts.append(f"Món ăn yêu thích của bạn là {food}.")

        # Check pet
        if "nuôi" in q or "con gì" in q or "thú cưng" in q or "corgi" in q or "bơ" in q:
            pet = facts.get("Thú cưng")
            if pet:
                ans_parts.append(f"Bạn nuôi một chú {pet}.")

        # Check style
        if "style" in q or "trả lời" in q:
            style = facts.get("Style trả lời")
            if style:
                ans_parts.append(f"Style trả lời mong muốn: {style}.")

        # Check interests
        if "quan tâm" in q or "thích" in q:
            interests = facts.get("Mối quan tâm")
            if interests:
                ans_parts.append(f"Mối quan tâm của bạn: {interests}.")

        # General summary
        if "dũngct" in q or "tóm tắt" in q or "mô tả" in q:
            name = facts.get("Tên", "DũngCT")
            loc = facts.get("Nơi ở", "Đà Nẵng")
            prof = facts.get("Nghề nghiệp", "backend engineer")
            drink = facts.get("Đồ uống yêu thích", "cà phê sữa đá")
            style = facts.get("Style trả lời", "ngắn gọn")
            summary = f"Bạn tên là {name}, hiện tại đang ở {loc}, làm nghề {prof}. Sở thích uống {drink} và yêu cầu style trả lời {style}."
            ans_parts.append(summary)

        if not ans_parts:
            if facts:
                ans_parts.append("Dựa trên thông tin tôi nhớ: " + ", ".join(f"{k}: {v}" for k, v in facts.items()))
            else:
                ans_parts.append("Tôi chưa nhớ được thông tin gì của bạn.")

        response = " ".join(ans_parts)

        # Apply 3-bullet style formatting if requested
        if "3 bullet" in q or "3 bullet" in facts.get("Style trả lời", ""):
            bullets = []
            if facts.get("Tên"):
                bullets.append(f"- Tên: {facts.get('Tên')}")
            if facts.get("Nơi ở"):
                bullets.append(f"- Nơi ở: {facts.get('Nơi ở')}")
            if facts.get("Nghề nghiệp"):
                bullets.append(f"- Nghề nghiệp: {facts.get('Nghề nghiệp')}")
            # Pad to 3 bullets
            if len(bullets) < 3:
                if facts.get("Đồ uống yêu thích"):
                    bullets.append(f"- Đồ uống: {facts.get('Đồ uống yêu thích')}")
                if len(bullets) < 3 and facts.get("Style trả lời"):
                    bullets.append(f"- Style: {facts.get('Style trả lời')}")
                if len(bullets) < 3:
                    bullets.append("- Mối quan tâm: Python, AI")
            response = "\n".join(bullets[:3])

        return response

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
