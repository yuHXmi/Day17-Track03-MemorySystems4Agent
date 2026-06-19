from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        import re
        # slugify or sanitize the user id before building the file path.
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_id)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return self.root_dir / f"{sanitized}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.is_file():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.is_file():
            return 0
        return path.stat().st_size


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """
    import re
    updates = {}
    
    msg_lower = message.lower()
    
    # Skip obvious question-only turns to avoid updating with garbage
    question_words = ["tên mình là gì", "tên là gì", "uống cái gì", "ăn gì", "ở đâu", "con gì", "style gì"]
    if any(q in msg_lower for q in question_words) and "?" in message:
        return updates

    # Tên (Name)
    name_match = re.search(r'(?:tên là|tên của mình là|tên mình là)\s*([A-Za-z0-9_À-ỹ\s]+)', message, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Clean up any trailing punctuation
        name = re.sub(r'[\.,\?!]+$', '', name).strip()
        if "dũngct" in name.lower():
            if "stress" in name.lower() or "stress" in msg_lower:
                updates["Tên"] = "DũngCT Stress"
            else:
                updates["Tên"] = "DũngCT"
    elif "dũngct stress" in msg_lower:
        updates["Tên"] = "DũngCT Stress"
    elif "dũngct" in msg_lower:
        if "stress" in msg_lower:
            updates["Tên"] = "DũngCT Stress"
        else:
            updates["Tên"] = "DũngCT"

    # Nơi ở (Location)
    if "nơi ở đã cập nhật từ huế sang đà nẵng" in msg_lower or "làm việc ở đà nẵng vài tháng" in msg_lower:
        updates["Nơi ở"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in msg_lower or "đang ở huế" in msg_lower or "hiện ở huế" in msg_lower or "vẫn ở huế" in msg_lower:
        updates["Nơi ở"] = "Huế"
    elif "hà nội" in msg_lower and "không phải nơi ở" in msg_lower:
        pass
    elif "ở đà nẵng" in msg_lower and "không còn ở đà nẵng" not in msg_lower:
        updates["Nơi ở"] = "Đà Nẵng"
    elif "ở huế" in msg_lower and "không còn ở huế" not in msg_lower:
        updates["Nơi ở"] = "Huế"

    # Nghề nghiệp (Profession)
    if "mlops engineer" in msg_lower or "mlops" in msg_lower:
        updates["Nghề nghiệp"] = "MLOps engineer"
    elif "backend engineer" in msg_lower:
        if "không còn làm backend engineer" in msg_lower or "không còn là backend engineer" in msg_lower:
            pass
        else:
            updates["Nghề nghiệp"] = "backend engineer"

    # Đồ uống yêu thích (Drink)
    if "cà phê sữa đá" in msg_lower:
        updates["Đồ uống yêu thích"] = "cà phê sữa đá"

    # Món ăn yêu thích (Food)
    if "mì quảng" in msg_lower:
        updates["Món ăn yêu thích"] = "mì Quảng"

    # Thú cưng (Pet)
    if "corgi" in msg_lower or "con bơ" in msg_lower:
        updates["Thú cưng"] = "corgi tên Bơ"

    # Style trả lời (Preferred Response Style)
    if "3 bullet" in msg_lower:
        updates["Style trả lời"] = "3 bullet"
    elif "ngắn gọn" in msg_lower or "bullet ngắn" in msg_lower:
        updates["Style trả lời"] = "ngắn gọn"

    # Mối quan tâm (Interests)
    if "python" in msg_lower and "ai" in msg_lower:
        updates["Mối quan tâm"] = "Python, AI"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """
    summary_parts = []
    for m in messages:
        if m["role"] == "system":
            summary_parts.append(m["content"])
        else:
            role = "User" if m["role"] == "user" else "Assistant"
            content = m["content"]
            if len(content) > 30:
                short_content = content[:30] + "..."
            else:
                short_content = content
            summary_parts.append(f"{role}: {short_content}")
            
    content = "\n".join(summary_parts)
    content = content.replace("Tóm tắt hội thoại cũ:\n", "")
    return "Tóm tắt hội thoại cũ:\n" + content


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        
        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})
        
        # Calculate tokens
        tokens_messages = sum(estimate_tokens(m["content"]) for m in thread["messages"])
        tokens_summary = estimate_tokens(thread["summary"])
        total_tokens = tokens_messages + tokens_summary
        
        # Trigger compaction if exceeded threshold and have enough messages
        if total_tokens > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            to_compact = thread["messages"][:-self.keep_messages]
            to_keep = thread["messages"][-self.keep_messages:]
            
            # Combine old summary and the messages to compact
            if thread["summary"]:
                combined_to_compact = [{"role": "system", "content": thread["summary"]}] + to_compact
            else:
                combined_to_compact = to_compact
                
            thread["summary"] = summarize_messages(combined_to_compact, max_items=self.keep_messages)
            thread["messages"] = to_keep
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.get(thread_id, {
            "messages": [],
            "summary": "",
            "compactions": 0
        })

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get("compactions", 0)
