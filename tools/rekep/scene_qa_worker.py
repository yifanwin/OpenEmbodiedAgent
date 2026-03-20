#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback


def _bootstrap_headless_from_argv():
    if "--no-headless" in sys.argv:
        os.environ["OMNIGIBSON_HEADLESS"] = "0"
        os.environ.pop("OMNIGIBSON_REMOTE_STREAMING", None)
        os.environ.pop("QT_QPA_PLATFORM", None)
        return
    if "--headless" in sys.argv:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"
        os.environ.pop("OMNIGIBSON_REMOTE_STREAMING", None)
        os.environ.pop("DISPLAY", None)
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


_bootstrap_headless_from_argv()

from scene_qa import SceneQuestionAnswerer


def main():
    parser = argparse.ArgumentParser(description="Worker process for ReKep scene question answering")
    parser.add_argument("--scene_file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    answerer = None
    try:
        answerer = SceneQuestionAnswerer(
            scene_file=args.scene_file,
            camera_id=args.camera_id,
            verbose=False,
        )
        result = answerer.answer_question(args.question)
        with open(args.result_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if answerer is not None:
            answerer.close()


if __name__ == "__main__":
    main()
