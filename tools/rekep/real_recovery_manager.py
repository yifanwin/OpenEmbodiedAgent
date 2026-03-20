from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RecoveryDecision:
    action: str = "continue"
    retry_stage: bool = False
    reobserve: bool = False
    replan: bool = False
    abort: bool = False
    reason: str = ""


class RealRecoveryManager:
    def __init__(self, *, max_stage_retries: int = 2):
        self.max_stage_retries = int(max_stage_retries)
        self.stage_retry_counts: Dict[int, int] = {}

    def reset(self):
        self.stage_retry_counts = {}

    def decide(self, *, stage: int, monitor_result: Dict[str, Any] | None, execution_error: str = "") -> Dict[str, Any]:
        monitor_result = monitor_result or {}
        status = str(monitor_result.get("status", "on_track"))
        suggested = str(monitor_result.get("suggested_action", "continue"))
        retry_count = self.stage_retry_counts.get(stage, 0)

        if execution_error:
            if retry_count < self.max_stage_retries:
                self.stage_retry_counts[stage] = retry_count + 1
                return RecoveryDecision(
                    action="retry_stage",
                    retry_stage=True,
                    reobserve=True,
                    replan=True,
                    reason=f"execution error: {execution_error}",
                ).__dict__
            return RecoveryDecision(
                action="abort",
                abort=True,
                reason=f"execution error exceeded retry budget: {execution_error}",
            ).__dict__

        if status == "no_keypoints":
            if retry_count < self.max_stage_retries:
                self.stage_retry_counts[stage] = retry_count + 1
                return RecoveryDecision(
                    action="reobserve",
                    retry_stage=True,
                    reobserve=True,
                    replan=False,
                    reason="missing keypoints; recapture observation",
                ).__dict__
            return RecoveryDecision(action="abort", abort=True, reason="missing keypoints after retries").__dict__

        if suggested == "replan" or status == "deviation":
            if retry_count < self.max_stage_retries:
                self.stage_retry_counts[stage] = retry_count + 1
                return RecoveryDecision(
                    action="replan",
                    retry_stage=True,
                    reobserve=True,
                    replan=True,
                    reason="constraint deviation detected",
                ).__dict__
            return RecoveryDecision(action="abort", abort=True, reason="constraint deviation exceeded retries").__dict__

        return RecoveryDecision(action="continue", reason="monitor ok").__dict__
