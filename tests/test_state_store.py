import sqlite3
import pytest
from finverse.state.store import StateStore


def test_versions_increment_and_old_versions_remain(tmp_path):
    s = StateStore(tmp_path / "s.sqlite")
    assert s.put("r", "2020-01-31", "agent_output", "a", {"x": 1}, ["src1"]) == 1
    assert s.put("r", "2020-01-31", "agent_output", "a", {"x": 2}) == 2
    assert s.get("r", "2020-01-31", "agent_output", "a")["payload"] == {"x": 2}
    assert s.get("r", "2020-01-31", "agent_output", "a", version=1)["payload"] == {"x": 1}
    assert s.get("r", "2020-01-31", "agent_output", "a", version=1)["source_ids"] == ["src1"]
    assert [h["version"] for h in s.history("r", "2020-01-31", "agent_output", "a")] == [1, 2]
    assert s.get("r", "2020-02-29", "agent_output", "a") is None


def test_records_are_immutable(tmp_path):
    s = StateStore(tmp_path / "s.sqlite")
    s.put("r", "2020-01-31", "regime", "hmm", {"p": 0.3})
    s.commit()
    with pytest.raises(sqlite3.DatabaseError):
        s.db.execute("UPDATE records SET payload='{}'")
    with pytest.raises(sqlite3.DatabaseError):
        s.db.execute("DELETE FROM records")


def test_unknown_kind_rejected(tmp_path):
    s = StateStore(tmp_path / "s.sqlite")
    with pytest.raises(ValueError):
        s.put("r", "2020-01-31", "bogus", "k", {})
