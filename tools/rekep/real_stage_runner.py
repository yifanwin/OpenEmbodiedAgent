import copy
import json
from pathlib import Path
import time

from real_runtime import RealObservation, RealStageExecution, RealStagePlan


class RealStageRunner:
    def __init__(
        self,
        *,
        env,
        adapter,
        model,
        temperature,
        max_tokens,
        camera_calibration,
        emit_progress,
        ask_image_question,
        parse_plan_from_vlm_text,
        keypoint_tracker=None,
        constraint_monitor=None,
        recovery_manager=None,
        constraint_evaluator=None,
        grasp_state_estimator=None,
        grasp_descend_m=0.0,
        grasp_descend_min_z_mm=40.0,
    ):
        self.env = env
        self.adapter = adapter
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.camera_calibration = camera_calibration
        self.emit_progress = emit_progress
        self.ask_image_question = ask_image_question
        self.parse_plan_from_vlm_text = parse_plan_from_vlm_text
        self.keypoint_tracker = keypoint_tracker
        self.constraint_monitor = constraint_monitor
        self.recovery_manager = recovery_manager
        self.constraint_evaluator = constraint_evaluator
        self.grasp_state_estimator = grasp_state_estimator
        self.grasp_descend_m = max(0.0, float(grasp_descend_m or 0.0))
        self.grasp_descend_min_z_mm = float(grasp_descend_min_z_mm)

    def _inject_grasp_descend_action(self, actions, stage_info):
        debug = {
            "grasp_descend_m": float(self.grasp_descend_m),
            "applied": False,
            "reason": "",
        }
        if self.grasp_descend_m <= 1e-9:
            debug["reason"] = "disabled"
            return actions, debug
        grasp_keypoint = stage_info.get("grasp_keypoint", -1)
        if grasp_keypoint in (-1, None):
            debug["reason"] = "not_grasp_stage"
            return actions, debug

        close_index = -1
        for idx, action in enumerate(actions):
            action_type = str(action.get("type", "")).strip().lower()
            if action_type in {"close_gripper", "grasp", "pick", "pick_grasp"}:
                close_index = idx
                break
        if close_index < 0:
            debug["reason"] = "no_close_action"
            return actions, debug

        base_index = -1
        base_action = None
        for idx in range(close_index - 1, -1, -1):
            action = actions[idx]
            if str(action.get("type", "")).strip().lower() == "movel":
                base_index = idx
                base_action = action
                break
        if base_action is None:
            debug["reason"] = "no_preclose_movel"
            return actions, debug

        pose = base_action.get("pose")
        if not isinstance(pose, list) or len(pose) < 6:
            debug["reason"] = "invalid_preclose_pose"
            return actions, debug

        descend_mm = float(self.grasp_descend_m) * 1000.0
        target_z = max(float(self.grasp_descend_min_z_mm), float(pose[2]) - descend_mm)
        if target_z >= float(pose[2]) - 1e-6:
            debug["reason"] = "clamped_no_motion"
            return actions, debug

        descend_action = copy.deepcopy(base_action)
        descend_action["pose"] = list(descend_action["pose"])
        descend_action["pose"][2] = float(target_z)

        adjusted = list(actions[: base_index + 1]) + [descend_action] + list(actions[base_index + 1 :])
        debug.update(
            {
                "applied": True,
                "reason": "insert_before_close",
                "insert_after_action_index": int(base_index + 1),
                "before_close_action_index": int(close_index + 2),
                "original_pose": list(pose),
                "descend_pose": list(descend_action["pose"]),
            }
        )
        return adjusted, debug

    def _build_plan_payload(self, raw_answer, cleaned_text, actions, runtime_adjustments):
        payload = {}
        text = str(cleaned_text or "").strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                payload = dict(parsed)
            elif isinstance(parsed, list):
                payload = {"actions": parsed}
            else:
                payload = {"raw_answer": raw_answer}
        else:
            payload = {"raw_answer": raw_answer}
        payload["actions"] = actions
        if runtime_adjustments:
            payload["runtime_adjustments"] = runtime_adjustments
        return payload

    def execute_stage(
        self,
        *,
        prefix,
        stage_info,
        stage_index,
        total_stages,
        capture_fn,
        keypoint_localizer,
        overlay_drawer,
        planner_prompt_builder,
        instruction,
        execute_motion,
        action_interval_s=0.0,
    ):
        attempt = 0
        while True:
            attempt += 1
            rgb, depth, capture = self.env.capture_rgbd(f"{prefix}_attempt{attempt}", capture_fn)
            keypoint_obs = keypoint_localizer(capture.frame_path, depth, self.camera_calibration)
            if self.keypoint_tracker is not None:
                keypoint_obs = self.keypoint_tracker.update(keypoint_obs, frame_path=capture.frame_path)
            initial_keypoints_3d = dict(keypoint_obs.get("keypoints_3d", {})) if isinstance(keypoint_obs.get("keypoints_3d"), dict) else {}
            overlay_path = str(Path(capture.frame_path).with_name(Path(capture.frame_path).stem + ".keypoints.png"))
            overlay_drawer(capture.frame_path, keypoint_obs, overlay_path)
            capture = self.env.add_overlay(capture, overlay_path, keypoint_obs)

            planner_prompt = planner_prompt_builder(
                instruction=instruction,
                stage_info=stage_info,
                keypoint_obs=keypoint_obs,
                camera_calibration=self.camera_calibration,
                current_stage=stage_index,
                total_stages=total_stages,
            )
            raw_answer, planner_vlm_cfg = self.ask_image_question(
                image_bytes=Path(capture.frame_path).read_bytes(),
                question=planner_prompt,
                default_model=self.model,
                system_prompt="You are a precise real robot planner for ReKep stages. Output strict JSON only.",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            actions, cleaned_text = self.parse_plan_from_vlm_text(raw_answer)
            actions, runtime_adjustments = self._inject_grasp_descend_action(actions, stage_info)
            plan_path = str(Path(capture.frame_path).with_suffix('.stage_plan.txt'))
            plan_payload = self._build_plan_payload(raw_answer, cleaned_text, actions, runtime_adjustments)
            Path(plan_path).write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.emit_progress(f"[real-stage-runner] stage={stage_index} attempt={attempt} planned_actions={len(actions)}")
            if runtime_adjustments.get("applied"):
                self.emit_progress(
                    f"[real-stage-runner] stage={stage_index} injected pre-close descend "
                    f"{float(runtime_adjustments.get('grasp_descend_m', 0.0)):.4f}m"
                )

            execution_records = []
            execution_error = ""
            for idx, action in enumerate(actions, start=1):
                try:
                    result = self.adapter.execute_action(action, execute_motion=bool(execute_motion))
                    execution_records.append({"index": idx, "ok": True, "result": result, "action": action})
                    self.emit_progress(f"[real-stage-runner][stage={stage_index}] action[{idx}] {action.get('type')} ok")
                    if bool(execute_motion) and float(action_interval_s) > 0.0 and idx < len(actions):
                        self.emit_progress(
                            f"[real-stage-runner][stage={stage_index}] action[{idx}] cooldown {float(action_interval_s):.1f}s"
                        )
                        time.sleep(float(action_interval_s))
                except Exception as exc:
                    execution_records.append({"index": idx, "ok": False, "error": str(exc), "action": action})
                    execution_error = str(exc)
                    self.emit_progress(f"[real-stage-runner][stage={stage_index}] action[{idx}] {action.get('type')} failed: {exc}")
                    break

            grasp_state = None
            if self.grasp_state_estimator is not None:
                grasp_state = self.grasp_state_estimator.update_from_adapter(
                    self.adapter,
                    keypoint_obs=keypoint_obs,
                    stage_info=stage_info,
                )

            constraint_eval = None
            if self.constraint_evaluator is not None:
                grasped_keypoints = []
                grasp_keypoint = stage_info.get("grasp_keypoint", -1)
                if grasp_keypoint not in (-1, None):
                    grasped_keypoints.append(int(grasp_keypoint))
                constraint_eval = self.constraint_evaluator.evaluate_stage(
                    stage_info=stage_info,
                    keypoint_obs=keypoint_obs,
                    grasped_keypoints=grasped_keypoints,
                    grasp_state=grasp_state,
                )
                self.emit_progress(f"[real-stage-runner] stage={stage_index} constraint_ok={constraint_eval.get('ok')} subgoal_ok={constraint_eval.get('subgoal_ok')} path_ok={constraint_eval.get('path_ok')}")

            monitor_result = None
            if self.constraint_monitor is not None:
                monitor_result = self.constraint_monitor.evaluate(stage_info=stage_info, keypoint_obs=keypoint_obs, constraint_eval=constraint_eval)
                self.emit_progress(f"[real-stage-runner] stage={stage_index} monitor_status={monitor_result.get('status')} score={monitor_result.get('score')}")

            recovery_result = None
            if self.recovery_manager is not None:
                recovery_result = self.recovery_manager.decide(
                    stage=stage_index,
                    monitor_result=monitor_result,
                    execution_error=execution_error,
                )
                self.emit_progress(f"[real-stage-runner] stage={stage_index} recovery_action={recovery_result.get('action')} reason={recovery_result.get('reason')}")

            stage_execution = RealStageExecution(
                stage=stage_index,
                observation=RealObservation(
                    frame_path=capture.frame_path,
                    depth_path=capture.depth_path,
                    overlay_path=capture.overlay_path,
                    capture_info=capture.capture_info or {},
                    keypoint_obs=keypoint_obs,
                ),
                plan=RealStagePlan(
                    stage=stage_index,
                    plan_actions=actions,
                    plan_raw_output_path=plan_path,
                    stage_constraints={
                        "subgoal_constraints_path": stage_info.get("subgoal_constraints_path"),
                        "path_constraints_path": stage_info.get("path_constraints_path"),
                        "grasp_keypoint": stage_info.get("grasp_keypoint"),
                        "release_keypoint": stage_info.get("release_keypoint"),
                        "grasp_state": grasp_state or {},
                        "constraint_eval": constraint_eval or {},
                        "monitor_result": monitor_result or {},
                        "recovery_result": recovery_result or {},
                        "runtime_adjustments": runtime_adjustments,
                        "initial_keypoints_3d": initial_keypoints_3d,
                        "attempt": attempt,
                    },
                    notes=(planner_vlm_cfg or {}).get("model", ""),
                ),
                execution_records=execution_records,
                execution_error=execution_error,
            )
            if recovery_result and recovery_result.get("retry_stage"):
                continue
            return stage_execution
