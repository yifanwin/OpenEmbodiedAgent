from __future__ import annotations

import sys
from pathlib import Path

import pytest

REKEP_DIR = Path(__file__).parent.parent / "tools" / "rekep"
sys.path.insert(0, str(REKEP_DIR))

from robot_factory import (  # noqa: E402
    create_robot_adapter,
    list_adapter_families,
    register_adapter_factory,
    unregister_adapter_factory,
)


class _DummyAdapter:
    def __init__(self, family: str):
        self.family = family

    def connect(self):
        return {"ok": True, "family": self.family}

    def close(self):
        return None

    def get_runtime_state(self):
        return {"source": self.family, "connected": True}

    def execute_action(self, action, execute_motion=False):
        return {"ok": True, "action": action, "executed": bool(execute_motion)}


def test_builtin_families_present():
    families = list_adapter_families()
    assert "dobot" in families
    assert "cellbot" in families


def test_create_cellbot_adapter_from_factory():
    adapter = create_robot_adapter(
        robot_family="cellbot",
        robot_driver="cellbot_sdk",
        host="127.0.0.1",
        port=9000,
    )
    connect = adapter.connect()
    assert connect["ok"] is True
    result = adapter.execute_action({"type": "movej"}, execute_motion=False)
    assert result["ok"] is True
    assert result["dry_run"] is True


def test_register_custom_adapter_factory():
    family = "unit_test_family"

    def _factory(**_kwargs):
        return _DummyAdapter(family)

    register_adapter_factory(family, _factory, overwrite=True)
    try:
        adapter = create_robot_adapter(robot_family=family, robot_driver="dummy")
        assert adapter.connect()["family"] == family
    finally:
        unregister_adapter_factory(family)


def test_unknown_family_raises():
    with pytest.raises(RuntimeError, match="Unsupported robot family"):
        create_robot_adapter(robot_family="not_exist_family", robot_driver="dummy")
