import datetime
import json
import os

import cv2

from environment import ReKepOGEnv
from utils import get_config
from vlm_client import ask_image_question


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
SCENE_QA_DIR = os.path.join(BASE_DIR, "scene_qa")
SCENE_QA_SYSTEM_PROMPT = (
    "You are a robotics scene analyst. Answer only from visible evidence in the current camera frame. "
    "If the image is ambiguous or an object is not clearly visible, say so explicitly instead of guessing."
)


def _headless_enabled_from_env():
    return os.environ.get("OMNIGIBSON_HEADLESS", "0").strip().lower() in {"1", "true", "t", "yes", "on"}


class SceneQuestionAnswerer:
    def __init__(self, scene_file, camera_id=0, verbose=False):
        global_config = get_config(config_path=DEFAULT_CONFIG_PATH)
        self.global_config = global_config
        self.scene_file = os.path.abspath(scene_file)
        self.camera_id = int(camera_id)
        self.verbose = verbose
        print(f"[scene_qa] runtime headless={_headless_enabled_from_env()} camera_id={self.camera_id}")
        self.env = ReKepOGEnv(global_config["env"], self.scene_file, verbose=verbose)

    def close(self):
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None

    def _capture_rgb(self):
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        if self.camera_id not in cam_obs:
            raise ValueError(f"Camera {self.camera_id} is not available. Cameras: {sorted(cam_obs.keys())}")
        return cam_obs[self.camera_id]["rgb"]

    def _save_snapshot(self, rgb):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(SCENE_QA_DIR, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"camera_{self.camera_id}.png")
        cv2.imwrite(save_path, rgb[..., ::-1])
        return os.path.abspath(save_path)

    def _save_vlm_trace(self, snapshot_path, question, answer, vlm_config):
        trace_path = os.path.join(os.path.dirname(snapshot_path), "vlm_trace.json")
        payload = {
            "question": question,
            "answer": answer,
            "snapshot_path": snapshot_path,
            "camera_id": self.camera_id,
            "vlm": {
                "model": vlm_config["model"],
                "base_url": vlm_config["base_url"],
                "api_key_env": vlm_config["api_key_env"],
            },
        }
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return trace_path

    def answer_question(self, question):
        if not question or not question.strip():
            raise ValueError("A non-empty scene question is required")
        question = question.strip()
        print(f"[vlm] scene_question: {question}")
        rgb = self._capture_rgb()
        snapshot_path = self._save_snapshot(rgb)
        ok, encoded = cv2.imencode(".png", rgb[..., ::-1])
        if not ok:
            raise RuntimeError("Failed to encode scene snapshot as PNG")
        answer, vlm_config = ask_image_question(
            image_bytes=encoded.tobytes(),
            question=question,
            default_model=self.global_config["constraint_generator"]["model"],
            system_prompt=SCENE_QA_SYSTEM_PROMPT,
            temperature=self.global_config["constraint_generator"]["temperature"],
            max_tokens=min(int(self.global_config["constraint_generator"]["max_tokens"]), 1024),
        )
        print(f"[vlm] resolved_model: {vlm_config['model']}")
        print("[vlm] assistant_response_begin")
        print(answer)
        print("[vlm] assistant_response_end")
        trace_path = self._save_vlm_trace(snapshot_path, question, answer, vlm_config)
        print(f"[vlm] trace_path: {trace_path}")
        return {
            "question": question,
            "answer": answer,
            "snapshot_path": snapshot_path,
            "vlm_trace_path": trace_path,
            "camera_id": self.camera_id,
            "vlm": {
                "model": vlm_config["model"],
                "base_url": vlm_config["base_url"],
                "api_key_env": vlm_config["api_key_env"],
            },
        }
