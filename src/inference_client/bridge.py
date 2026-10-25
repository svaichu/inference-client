"""The control loop: Station <-> remote GR00T policy server.

Two modes:

``live`` (default) -- connects to a running Station daemon and a running
ElRobot, fetches frames, calls the remote policy, sends sync-write commands.
Requires ``--norma-core-path`` (a norma-core checkout) and a Station daemon
reachable at ``--server``.

``--offline PATH`` -- replays a recorded ``.parquet`` dataset (needs the
``offline`` extra) through the same observation-building + policy-call code,
printing the predicted action per frame instead of sending it anywhere. No
Station, no robot, no norma-core checkout needed. Use this to test the
tunnel + policy server + conversion logic before touching hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .frame import StationFrame
from .policy_client import PolicyClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-host", required=True, help="Cloud GR00T server host (tunnel address)")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--api-token", default=None)

    p.add_argument("--task", required=True, help="Language instruction the policy is conditioned on")
    p.add_argument("--image-keys", default="cam0,cam1", help="Comma-separated, in Frame image order")
    p.add_argument("--state-key", default="state", help="Must match the checkpoint's modality config")

    p.add_argument("--offline", default=None, help="Replay this .parquet file instead of live hardware")

    # Live-mode only:
    p.add_argument("--server", default="localhost", help="Station daemon host[:port], default port 8888")
    p.add_argument(
        "--norma-core-path",
        default=os.environ.get("NORMA_CORE_PATH"),
        help="Path to a norma-core checkout (or set NORMA_CORE_PATH). Required for live mode.",
    )
    p.add_argument("--bus-serial", default=None, help="ST3215 bus id, from the station web viewer")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument(
        "--max-delta-ticks", type=int, default=200,
        help="Abort a tick if the predicted goal is more than this many ticks from the "
        "current position on any joint. 0 disables the check.",
    )
    p.add_argument("--auto", action="store_true", help="Run continuously instead of interactive SPACE-per-tick")
    p.add_argument("--max-ticks", type=int, default=0, help="Stop after N ticks in --auto mode. 0 = unlimited")
    return p.parse_args()


def run_offline(args) -> None:
    image_keys = tuple(args.image_keys.split(","))
    with PolicyClient(host=args.policy_host, port=args.policy_port, api_token=args.api_token) as policy:
        if not policy.ping():
            raise SystemExit(f"Can't reach policy server at {args.policy_host}:{args.policy_port}")
        print("modality config:", policy.get_modality_config())

        for i, frame in enumerate(StationFrame.iter_parquet(args.offline)):
            frame.task = frame.task or args.task
            obs = frame.to_observation(image_keys, args.state_key)
            action, info = policy.get_action(obs)
            print(f"frame {i}: action keys={list(action.keys())} info={info}")


async def run_live_tick(link, policy, args, motor_ids: list[int], last_id: bytes) -> bytes:
    from .station_live import unnormalize_goal

    image_keys = tuple(args.image_keys.split(","))
    frame_reader, frame_id = await link.fetch_fresh_frame(last_id, args.timeout)
    station_frame = StationFrame.from_live(frame_reader, task=args.task)

    obs = station_frame.to_observation(image_keys, args.state_key)
    action, _info = policy.get_action(obs)
    action_key = next(iter(action))  # single-key action spaces are the common case
    goal_norm = action[action_key][0, 0]  # first step of the predicted chunk

    ranges = station_frame.joint_ranges()
    raw_goals = [unnormalize_goal(g, lo, hi) for g, (lo, hi) in zip(goal_norm, ranges)]

    deltas = [abs(raw - j.position) for raw, j in zip(raw_goals, station_frame.joints)]
    max_delta = max(deltas) if deltas else 0
    if args.max_delta_ticks and max_delta > args.max_delta_ticks:
        print(f"  aborted tick: max|delta|={max_delta} > {args.max_delta_ticks}")
        return frame_id

    cmd = link.build_sync_write_command(args.bus_serial, motor_ids, raw_goals)
    await link.send(cmd)
    print(f"  sent sync-write, max|delta|={max_delta}")
    return frame_id


async def run_live(args) -> None:
    from .station_live import StationLink, add_norma_core_to_path

    if not args.norma_core_path:
        raise SystemExit("--norma-core-path (or NORMA_CORE_PATH) is required for live mode")
    if not args.bus_serial:
        raise SystemExit("--bus-serial is required for live mode")
    add_norma_core_to_path(args.norma_core_path)

    logger = logging.getLogger("inference_client")
    logging.basicConfig(level=logging.INFO)
    motor_ids = [int(x) for x in args.motor_ids.split(",")]

    link = await StationLink.connect(args.server, logger)
    with PolicyClient(host=args.policy_host, port=args.policy_port, api_token=args.api_token) as policy:
        if not policy.ping():
            raise SystemExit(f"Can't reach policy server at {args.policy_host}:{args.policy_port}")
        print("modality config:", policy.get_modality_config())

        tick = 0
        last_id = b""
        try:
            while True:
                if args.max_ticks and tick >= args.max_ticks:
                    print(f"reached --max-ticks={args.max_ticks}, stopping")
                    return
                tick += 1
                last_id = await run_live_tick(link, policy, args, motor_ids, last_id)
                if not args.auto:
                    input(f"tick {tick} done -- press Enter for next, Ctrl-C to quit ")
        except KeyboardInterrupt:
            print("\ninterrupted -- stopping")


def main() -> None:
    args = parse_args()
    if args.offline:
        run_offline(args)
    else:
        asyncio.run(run_live(args))


if __name__ == "__main__":
    main()
