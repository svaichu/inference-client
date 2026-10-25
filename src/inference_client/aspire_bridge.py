"""Robot-side bridge for an ASPIRE policy server (``elrobot_remote_server``).

Unlike ``bridge.py``, the server does not infer an action from every frame. The ASPIRE agent
decides what to do through its tools; this loop only reports the arm and camera and executes
whatever trajectory chunks the server hands back. One tick:

1. fetch a fresh Station frame (state ``position_norm`` x8, JPEGs -> RGB);
2. ``get_action(obs, options)`` where ``options`` acknowledges the chunk executed last tick;
3. if the reply carries a chunk, send each step as a sync-write at the server's ``dt``;
   otherwise (idle) wait ``poll_s`` and ask again.

Every step is checked against ``--max-step-ticks`` (and the first against ``--max-delta-ticks``
from the arm's current position) before it is sent; a violation abandons the chunk and tells the
server, which fails the agent's tool call with that reason.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

import numpy as np

from .frame import StationFrame
from .policy_client import PolicyClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-host", required=True, help="ASPIRE server host (tunnel address)")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--api-token", default=None)
    p.add_argument("--task", default="", help="Informational; sent as the language field")
    p.add_argument(
        "--cameras", default="overhead=0",
        help="alias=index pairs mapping server camera names to Frame image order",
    )
    p.add_argument("--server", default="localhost", help="Station daemon host[:port]")
    p.add_argument("--norma-core-path", default=os.environ.get("NORMA_CORE_PATH"))
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--max-delta-ticks", type=int, default=200,
                   help="Largest allowed gap between the arm and a chunk's first step")
    p.add_argument("--max-step-ticks", type=int, default=40,
                   help="Largest allowed change between consecutive steps (~0.06 rad)")
    return p.parse_args()


def check_chunk(targets_raw: list[list[int]], current: list[int], max_delta: int, max_step: int) -> str | None:
    """Reason to refuse a chunk of raw tick targets, or None if it is safe to send."""
    previous, limit = current, max_delta
    for index, step in enumerate(targets_raw):
        jump = max(abs(a - b) for a, b in zip(step, previous))
        if limit and jump > limit:
            return f"step {index} moves {jump} ticks (limit {limit})"
        previous, limit = step, max_step
    return None


def build_observation(frame: StationFrame, cameras: dict[str, int], task: str) -> dict:
    return {
        "state": {"state": np.array([[[j.position_norm for j in frame.joints]]], np.float32)},
        "video": {a: frame.images[i].decode()[None, None] for a, i in cameras.items()},
        "language": {"task": [[task]]},
    }


async def run(args: argparse.Namespace) -> None:
    from .station_live import StationLink, add_norma_core_to_path, unnormalize_goal

    if not args.norma_core_path:
        raise SystemExit("--norma-core-path (or NORMA_CORE_PATH) is required")
    add_norma_core_to_path(args.norma_core_path)
    logging.basicConfig(level=logging.INFO)

    cameras = {k: int(v) for k, v in (pair.split("=") for pair in args.cameras.split(","))}
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    link = await StationLink.connect(args.server, logging.getLogger("inference_client"))

    with PolicyClient(host=args.policy_host, port=args.policy_port, api_token=args.api_token) as policy:
        if not policy.ping():
            raise SystemExit(f"Can't reach the ASPIRE server at {args.policy_host}:{args.policy_port}")
        print("server:", policy.get_modality_config())

        last_id, options = b"", None
        while True:
            reader, last_id = await link.fetch_fresh_frame(last_id, args.timeout)
            frame = StationFrame.from_live(reader, task=args.task)
            action, info = policy.get_action(build_observation(frame, cameras, args.task), options)
            options = None

            if info.get("closed"):
                print("server closed; exiting")
                return
            targets = action["action"][0]
            if info.get("idle") or len(targets) == 0:
                await asyncio.sleep(float(info.get("poll_s", 0.05)))
                continue

            ranges = frame.joint_ranges()
            raw = [
                [unnormalize_goal(g, lo, hi) for g, (lo, hi) in zip(step, ranges)]
                for step in targets
            ]
            problem = check_chunk(raw, [j.position for j in frame.joints], args.max_delta_ticks, args.max_step_ticks)
            if problem:
                print(f"  refusing chunk {info['chunk']}: {problem}")
                options = {"fault": problem}
                continue

            dt, t0 = float(info["dt"]), time.perf_counter()
            for n, step in enumerate(raw):
                await link.send(link.build_sync_write_command(args.bus_serial, motor_ids, step))
                await asyncio.sleep(max(0.0, t0 + (n + 1) * dt - time.perf_counter()))
            options = {"ack": {"segment": info["segment"], "chunk": info["chunk"]}}
            print(f"  segment {info['segment']} chunk {info['chunk']}{' (last)' if info['last'] else ''}")


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping")


if __name__ == "__main__":
    main()
