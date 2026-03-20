from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class RobotAdapter(Protocol):
    """Hardware abstraction interface for robot backends.

    Upper-layer runtime/supervisor code should only rely on this contract,
    while vendor-specific drivers translate these operations to the actual SDK.
    """

    def connect(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...

    def get_runtime_state(self) -> Dict[str, Any]: ...

    def execute_action(self, action: Dict[str, Any], execute_motion: bool = False) -> Dict[str, Any]: ...
