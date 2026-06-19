from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""
    from config import LabConfig
    from model_provider import ProviderConfig

    model_cfg = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=50,  # low threshold for testing compaction
        compact_keep_messages=2,     # keep only 2 messages
        model=model_cfg,
        judge_model=model_cfg
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # Test write
    path = store.write_text(user_id, "# Profile\n- **Tên**: DũngCT\n- **Nơi ở**: Đà Nẵng")
    assert path.is_file()

    # Test read
    content = store.read_text(user_id)
    assert "DũngCT" in content
    assert "Đà Nẵng" in content

    # Test edit
    changed = store.edit_text(user_id, "- **Nơi ở**: Đà Nẵng", "- **Nơi ở**: Huế")
    assert changed

    content_new = store.read_text(user_id)
    assert "Huế" in content_new
    assert "Đà Nẵng" not in content_new

    # Test file size
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)

    user_id = "test_user"
    thread_id = "thread_1"

    # Send multiple messages to trigger compaction
    agent.reply(user_id, thread_id, "Chào bạn, mình tên là DũngCT.")
    agent.reply(user_id, thread_id, "Mình đang làm việc ở Đà Nẵng.")
    agent.reply(user_id, thread_id, "Đồ uống yêu thích là cà phê sữa đá.")
    agent.reply(user_id, thread_id, "Mình cũng thích mì Quảng và corgi.")

    # Check compaction happened
    assert agent.compaction_count(thread_id) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)

    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "test_user"

    # Session 1: Feed facts to both agents
    thread_1 = "thread_1"
    baseline.reply(user_id, thread_1, "Chào bạn, mình tên là DũngCT.")
    baseline.reply(user_id, thread_1, "Đồ uống yêu thích là cà phê sữa đá.")

    advanced.reply(user_id, thread_1, "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, thread_1, "Đồ uống yêu thích là cà phê sữa đá.")

    # Session 2: Fresh thread
    thread_2 = "thread_2"
    q = "Nhắc lại tên mình và đồ uống yêu thích của mình"

    ans_baseline = baseline.reply(user_id, thread_2, q)["content"]
    ans_advanced = advanced.reply(user_id, thread_2, q)["content"]

    # Baseline should NOT recall
    assert "dũngct" not in ans_baseline.lower()
    assert "cà phê sữa đá" not in ans_baseline.lower()

    # Advanced SHOULD recall from User.md
    assert "dũngct" in ans_advanced.lower()
    assert "cà phê sữa đá" in ans_advanced.lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)

    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "test_user"
    thread_id = "thread_long"

    # Send a series of messages to both
    msgs = [
        "Tin Artemis III của NASA được công bố vào tháng 6 năm 2026.",
        "Nhiệm vụ này sẽ đưa phi hành đoàn bay quanh Mặt Trăng vào năm 2027.",
        "Máy bay siêu thanh X-59 đã đạt vận tốc Mach 1.1.",
        "Tổ chức Khí tượng Thế giới cảnh báo El Nino quay trở lại.",
        "British Columbia công bố kế hoạch điện sạch mới ngày 15 tháng 6.",
        "Chúng ta cần cân bằng giữa mở rộng công suất và tiết kiệm năng lượng."
    ]

    for m in msgs:
        baseline.reply(user_id, thread_id, m)
        advanced.reply(user_id, thread_id, m)

    # Compare cumulative prompt load
    prompt_load_baseline = baseline.prompt_token_usage(thread_id)
    prompt_load_advanced = advanced.prompt_token_usage(thread_id)

    # Advanced's cumulative prompt load is lower because of message pruning after compaction
    assert prompt_load_advanced < prompt_load_baseline
