from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RealObservation:
    frame_path: str
    depth_path: str
    overlay_path: str = ""
    capture_info: Dict[str, Any] = field(default_factory=dict)
    keypoint_obs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealStagePlan:
    stage: int
    plan_actions: List[Dict[str, Any]] = field(default_factory=list)
    plan_raw_output_path: str = ""
    stage_constraints: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RealStageExecution:
    stage: int
    observation: Optional[RealObservation] = None
    plan: Optional[RealStagePlan] = None
    execution_records: List[Dict[str, Any]] = field(default_factory=list)
    execution_error: str = ""


@dataclass
class RealTaskRuntime:
    task: str
    instruction: str
    planning_observation: Optional[RealObservation] = None
    generated_program_dir: str = ""
    generated_program_info: Dict[str, Any] = field(default_factory=dict)
    stage_executions: List[RealStageExecution] = field(default_factory=list)

    def add_stage_execution(self, stage_execution: RealStageExecution):
        self.stage_executions.append(stage_execution)

    @property
    def ok(self) -> bool:
        return all(not item.execution_error for item in self.stage_executions)
