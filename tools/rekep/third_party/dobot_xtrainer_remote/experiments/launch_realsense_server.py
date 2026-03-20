import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from dobot_control.cameras.realsense_stream import (  # noqa: E402
    RealSenseStreamServer,
    list_connected_realsense_serials,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a RealSense stream server.")
    parser.add_argument("--serial-number", default="338522301270")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--topic", default="realsense")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show-ui", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    serials = list_connected_realsense_serials()
    print(f"Detected RealSense serials: {serials}")
    if args.serial_number not in serials:
        raise RuntimeError(
            f"Target serial {args.serial_number} not found. Available serials: {serials}"
        )

    print(f"Streaming port: {args.port}")
    server = RealSenseStreamServer(
        serial_number=args.serial_number,
        host=args.host,
        port=args.port,
        topic=args.topic,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        flip=args.flip,
        show_ui=args.show_ui,
    )
    server.serve()


if __name__ == "__main__":
    main(build_parser().parse_args())
