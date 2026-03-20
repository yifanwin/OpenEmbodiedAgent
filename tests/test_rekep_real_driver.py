from __future__ import annotations

from hal.drivers.rekep_real_driver import ReKepRealDriver


def test_real_action_alias_maps_to_bridge_action(monkeypatch):
    called = {}

    def fake_invoke(self, action, params):
        called["action"] = action
        called["params"] = params
        return {"ok": True, "preflight": {"status": "ready"}}, None

    monkeypatch.setattr(ReKepRealDriver, "_invoke_bridge", fake_invoke)
    driver = ReKepRealDriver()
    msg = driver.execute_action("real_preflight", {})

    assert called["action"] == "preflight"
    assert "preflight" in msg.lower()


def test_high_level_action_is_translated_to_instruction(monkeypatch):
    called = {}

    def fake_invoke(self, action, params):
        called["action"] = action
        called["params"] = params
        return {"ok": True, "action": action}, None

    monkeypatch.setattr(ReKepRealDriver, "_invoke_bridge", fake_invoke)
    driver = ReKepRealDriver()
    msg = driver.execute_action("pick_up", {"target": "apple"})

    assert called["action"] == "execute"
    assert "instruction" in called["params"]
    assert "apple" in str(called["params"]["instruction"]).lower()
    assert "succeeded" in msg.lower()


def test_unknown_action_returns_error_string():
    driver = ReKepRealDriver()
    msg = driver.execute_action("__nonexistent__", {})
    assert "unknown action" in msg.lower()
