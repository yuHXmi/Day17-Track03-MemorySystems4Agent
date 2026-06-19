from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    import os
    from dotenv import load_dotenv

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load .env file
    load_dotenv(root / ".env")
    load_dotenv()

    # Directories
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Compact Memory Settings
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    # Provider and Model
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        if os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            provider = "openai"

    model_name = os.getenv("LLM_MODEL", "")
    if not model_name:
        if provider == "openrouter":
            model_name = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
        else:
            model_name = "gpt-4o-mini"

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    api_key = None
    base_url = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    judge_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_config,
    )
