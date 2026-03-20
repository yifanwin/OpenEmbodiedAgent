import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import parse

from vlm_client import request_chat_completion


class RealTimeConstraintGenerator:
    def __init__(self, repo_dir, model="gpt-5.4", temperature=0.0, max_tokens=2048):
        self.repo_dir = Path(repo_dir)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.base_dir = self.repo_dir / "vlm_query"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template = (self.base_dir / "prompt_template.txt").read_text(encoding="utf-8")
        self.client_config = None
        self.task_dir = None

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _safe_int(value, fallback=-1):
        try:
            return int(value)
        except Exception:
            return int(fallback)

    @staticmethod
    def _normalize_text(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    def _build_index_guard(self, metadata, prior_issues=None):
        task_schema = (metadata or {}).get("task_schema", {}) if isinstance(metadata, dict) else {}
        keypoints = task_schema.get("keypoints") if isinstance(task_schema.get("keypoints"), list) else []
        lines = []
        if keypoints:
            lines.append("Use exact keypoint IDs from this table. Never remap ID meanings:")
            for item in keypoints:
                if not isinstance(item, dict):
                    continue
                kid = self._safe_int(item.get("id"), -1)
                label = str(item.get("label", ""))
                obj = str(item.get("object", ""))
                purpose = str(item.get("purpose", ""))
                lines.append(f"- id {kid}: label={label}, object={obj}, purpose={purpose}")
            grasp_candidates = []
            for item in keypoints:
                if not isinstance(item, dict):
                    continue
                text = self._normalize_text(item.get("label", "")) + " " + self._normalize_text(item.get("purpose", ""))
                if any(token in text for token in ("grasp", "stem", "handle", "pick", "pickup", "grip")):
                    grasp_candidates.append(self._safe_int(item.get("id"), -1))
            grasp_candidates = sorted({x for x in grasp_candidates if x >= 0})
            if grasp_candidates:
                lines.append(f"- First grasp stage must use one of grasp-like IDs: {grasp_candidates}.")
        lines.extend(
            [
                "- Never set grasp_keypoints to plate/bowl/rim/table/support points unless instruction explicitly asks to grasp them.",
                "- For each grasp stage s with grasp_keypoints[s-1]=k, stage{s}_subgoal constraints must reference keypoints[k].",
                "- Keep grasp_keypoints/release_keypoints lengths exactly equal to num_stages.",
            ]
        )
        if prior_issues:
            lines.append("Previous output failed checks; fix all issues below:")
            for issue in prior_issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)

    def _build_prompt(self, image_path, instruction, index_guard_text=""):
        img_base64 = self._encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        if index_guard_text:
            prompt_text = (
                f"{prompt_text}\n\n"
                "## Additional Hard Constraints\n"
                f"{index_guard_text}\n"
            )
        (self.task_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ],
            }
        ]
        return messages, prompt_text

    def _validate_generation_consistency(self, metadata):
        issues = []
        metadata = metadata if isinstance(metadata, dict) else {}
        num_stages = self._safe_int(metadata.get("num_stages"), 0)
        if num_stages <= 0:
            issues.append("num_stages must be > 0")
            return issues
        grasp = metadata.get("grasp_keypoints") if isinstance(metadata.get("grasp_keypoints"), list) else []
        release = metadata.get("release_keypoints") if isinstance(metadata.get("release_keypoints"), list) else []
        if len(grasp) != num_stages:
            issues.append(f"grasp_keypoints length {len(grasp)} != num_stages {num_stages}")
        if len(release) != num_stages:
            issues.append(f"release_keypoints length {len(release)} != num_stages {num_stages}")

        task_schema = metadata.get("task_schema") if isinstance(metadata.get("task_schema"), dict) else {}
        schema_keypoints = task_schema.get("keypoints") if isinstance(task_schema.get("keypoints"), list) else []
        schema_ids = {self._safe_int(item.get("id"), -1) for item in schema_keypoints if isinstance(item, dict)}
        schema_objects = {self._safe_int(item.get("id"), -1): self._normalize_text(item.get("object", "")) for item in schema_keypoints if isinstance(item, dict)}
        schema_text = {
            self._safe_int(item.get("id"), -1): (
                self._normalize_text(item.get("label", "")) + " " + self._normalize_text(item.get("purpose", ""))
            )
            for item in schema_keypoints
            if isinstance(item, dict)
        }
        container_tokens = ("plate", "bowl", "rim", "dish", "tray", "container", "table", "mat", "desk", "盘", "碗", "盆")

        # Basic index range checks.
        for stage_idx in range(min(num_stages, len(grasp))):
            g = self._safe_int(grasp[stage_idx], -1)
            if g != -1 and schema_ids and g not in schema_ids:
                issues.append(f"stage {stage_idx + 1} grasp_keypoint {g} is not in schema ids")
        for stage_idx in range(min(num_stages, len(release))):
            r = self._safe_int(release[stage_idx], -1)
            if r != -1 and schema_ids and r not in schema_ids:
                issues.append(f"stage {stage_idx + 1} release_keypoint {r} is not in schema ids")

        # Release must refer to previously grasped keypoints.
        seen_grasped = set()
        for stage_idx in range(num_stages):
            if stage_idx < len(grasp):
                g = self._safe_int(grasp[stage_idx], -1)
                if g >= 0:
                    seen_grasped.add(g)
            if stage_idx < len(release):
                r = self._safe_int(release[stage_idx], -1)
                if r >= 0 and r not in seen_grasped:
                    issues.append(
                        f"stage {stage_idx + 1} release_keypoint {r} was never grasped in previous/current stages"
                    )

        # Semantic check: first grasp should prefer grasp-like keypoints and avoid container/support objects.
        first_grasp = None
        for value in grasp:
            candidate = self._safe_int(value, -1)
            if candidate >= 0:
                first_grasp = candidate
                break
        if first_grasp is not None and schema_keypoints:
            obj_text = schema_objects.get(first_grasp, "")
            label_purpose = schema_text.get(first_grasp, "")
            grasp_like_ids = sorted(
                {
                    kid
                    for kid, text in schema_text.items()
                    if kid >= 0 and any(token in text for token in ("grasp", "stem", "handle", "pick", "pickup", "grip"))
                }
            )
            if grasp_like_ids and first_grasp not in grasp_like_ids:
                issues.append(
                    f"first grasp_keypoint={first_grasp} is not in grasp-like ids {grasp_like_ids}"
                )
            if any(token in obj_text for token in container_tokens) and not any(
                token in label_purpose for token in ("grasp", "stem", "handle", "pick", "pickup", "grip")
            ):
                issues.append(
                    f"first grasp_keypoint={first_grasp} appears to be container/support object ({obj_text})"
                )

        # Constraint file check: each grasp stage should reference its grasp keypoint index.
        for stage_idx in range(min(num_stages, len(grasp))):
            g = self._safe_int(grasp[stage_idx], -1)
            if g < 0:
                continue
            subgoal_path = self.task_dir / f"stage{stage_idx + 1}_subgoal_constraints.txt"
            if not subgoal_path.exists():
                issues.append(f"missing subgoal constraints file for stage {stage_idx + 1}")
                continue
            subgoal_text = subgoal_path.read_text(encoding="utf-8")
            keypoint_ref = re.search(rf"keypoints\s*\[\s*{g}\s*\]", subgoal_text) is not None
            if not keypoint_ref:
                issues.append(
                    f"stage {stage_idx + 1} grasp_keypoint={g} not referenced in subgoal constraints"
                )
        return issues

    def _parse_and_save_constraints(self, output):
        lines = output.split("\n")
        functions = {}
        start = None
        name = None
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if start is not None and line.startswith("    return "):
                functions[name] = lines[start : i + 1]
                start = None
                name = None
        groupings = {}
        for fn_name in functions:
            key = "_".join(fn_name.split("_")[:-1])
            groupings.setdefault(key, []).append(fn_name)
        for key, names in groupings.items():
            with (self.task_dir / f"{key}_constraints.txt").open("w", encoding="utf-8") as f:
                for fn_name in names:
                    f.write("\n".join(functions[fn_name]) + "\n\n")

    @staticmethod
    def _parse_other_metadata(output):
        data_dict = {}
        num_stages = None
        for line in output.split("\n"):
            matched = parse.parse("num_stages = {num_stages}", line)
            if matched is not None:
                num_stages = matched
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict["num_stages"] = int(num_stages["num_stages"])

        grasp_keypoints = None
        for line in output.split("\n"):
            matched = parse.parse("grasp_keypoints = {grasp_keypoints}", line)
            if matched is not None:
                grasp_keypoints = matched
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        data_dict["grasp_keypoints"] = [int(x.strip()) for x in grasp_keypoints["grasp_keypoints"].replace("[", "").replace("]", "").split(",") if x.strip()]

        release_keypoints = None
        for line in output.split("\n"):
            matched = parse.parse("release_keypoints = {release_keypoints}", line)
            if matched is not None:
                release_keypoints = matched
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        data_dict["release_keypoints"] = [int(x.strip()) for x in release_keypoints["release_keypoints"].replace("[", "").replace("]", "").split(",") if x.strip()]
        return data_dict

    def _save_metadata(self, metadata):
        with (self.task_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def generate(self, image_path, instruction, metadata):
        safe_name = instruction.lower().replace(" ", "_")
        task_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + safe_name
        self.task_dir = self.base_dir / task_name
        self.task_dir.mkdir(parents=True, exist_ok=True)

        input_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        max_attempts = 2
        prior_issues = []
        accepted_output = ""
        accepted_prompt_text = ""
        accepted_metadata = dict(input_metadata)
        validation_issues = []
        total_elapsed = 0.0
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            index_guard = self._build_index_guard(input_metadata, prior_issues=prior_issues if attempt > 1 else None)
            messages, prompt_text = self._build_prompt(image_path, instruction, index_guard_text=index_guard)
            started = time.time()
            response, self.client_config = request_chat_completion(
                messages=messages,
                default_model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=600,
            )
            elapsed = time.time() - started
            total_elapsed += elapsed
            output = response["choices"][0]["message"]["content"]
            (self.task_dir / f"output_raw_attempt{attempt}.txt").write_text(output, encoding="utf-8")
            (self.task_dir / f"prompt_attempt{attempt}.txt").write_text(prompt_text, encoding="utf-8")
            trace = {
                "attempt": attempt,
                "instruction": instruction,
                "prompt_text": prompt_text,
                "response_text": output,
                "model": self.client_config["model"] if self.client_config else self.model,
                "base_url": self.client_config["base_url"] if self.client_config else "",
                "api_key_env": self.client_config["api_key_env"] if self.client_config else "",
                "prompt_path": str(self.task_dir / f"prompt_attempt{attempt}.txt"),
                "response_path": str(self.task_dir / f"output_raw_attempt{attempt}.txt"),
                "elapsed_s": elapsed,
            }
            (self.task_dir / f"vlm_trace_attempt{attempt}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            parsed_metadata = dict(input_metadata)
            parse_error = ""
            try:
                self._parse_and_save_constraints(output)
                parsed_metadata.update(self._parse_other_metadata(output))
                validation_issues = self._validate_generation_consistency(parsed_metadata)
            except Exception as exc:
                parse_error = str(exc)
                validation_issues = [f"parse/generation error: {parse_error}"]
            if not validation_issues:
                accepted_output = output
                accepted_prompt_text = prompt_text
                accepted_metadata = parsed_metadata
                break
            prior_issues = list(validation_issues)
            accepted_output = output
            accepted_prompt_text = prompt_text
            accepted_metadata = parsed_metadata

        (self.task_dir / "output_raw.txt").write_text(accepted_output, encoding="utf-8")
        (self.task_dir / "prompt.txt").write_text(accepted_prompt_text, encoding="utf-8")
        summary_trace = {
            "instruction": instruction,
            "prompt_text": accepted_prompt_text,
            "response_text": accepted_output,
            "model": self.client_config["model"] if self.client_config else self.model,
            "base_url": self.client_config["base_url"] if self.client_config else "",
            "api_key_env": self.client_config["api_key_env"] if self.client_config else "",
            "prompt_path": str(self.task_dir / "prompt.txt"),
            "response_path": str(self.task_dir / "output_raw.txt"),
            "elapsed_s": total_elapsed,
            "generation_validation": {
                "ok": not validation_issues,
                "issues": validation_issues,
                "attempts": attempts_used,
            },
        }
        (self.task_dir / "vlm_trace.json").write_text(json.dumps(summary_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        accepted_metadata["generation_validation"] = summary_trace["generation_validation"]
        self._save_metadata(accepted_metadata)
        return {
            "program_dir": str(self.task_dir),
            "metadata": accepted_metadata,
            "client_config": self.client_config or {},
            "trace_path": str(self.task_dir / "vlm_trace.json"),
            "raw_output_path": str(self.task_dir / "output_raw.txt"),
        }


def load_generated_program(program_dir):
    program_dir = Path(program_dir)
    metadata = json.loads((program_dir / "metadata.json").read_text(encoding="utf-8"))
    stages = []
    num_stages = int(metadata.get("num_stages", 0) or 0)
    for stage_idx in range(1, num_stages + 1):
        subgoal_path = program_dir / f"stage{stage_idx}_subgoal_constraints.txt"
        path_path = program_dir / f"stage{stage_idx}_path_constraints.txt"
        stages.append(
            {
                "stage": stage_idx,
                "subgoal_constraints_path": str(subgoal_path),
                "path_constraints_path": str(path_path),
                "subgoal_constraints": subgoal_path.read_text(encoding="utf-8") if subgoal_path.exists() else "",
                "path_constraints": path_path.read_text(encoding="utf-8") if path_path.exists() else "",
                "grasp_keypoint": (metadata.get("grasp_keypoints") or [])[stage_idx - 1] if stage_idx - 1 < len(metadata.get("grasp_keypoints") or []) else -1,
                "release_keypoint": (metadata.get("release_keypoints") or [])[stage_idx - 1] if stage_idx - 1 < len(metadata.get("release_keypoints") or []) else -1,
            }
        )
    return {"program_dir": str(program_dir), "metadata": metadata, "stages": stages}
