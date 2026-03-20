import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from dobot_control.cameras.realsense_stream import RealSenseStreamClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show RealSense stream in a UI window.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--topic", default="realsense")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--window-name", default="realsense_stream_client")
    return parser


def main(args: argparse.Namespace) -> None:
    print(f"Connecting to tcp://{args.host}:{args.port}, topic={args.topic}")
    client = RealSenseStreamClient(
        host=args.host,
        port=args.port,
        topic=args.topic,
        timeout_ms=args.timeout_ms,
    )
    client.run_ui(window_name=args.window_name)


if __name__ == "__main__":
    main(build_parser().parse_args())
