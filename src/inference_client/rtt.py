"""Round-trip-time probe for the remote GR00T policy server.

Answers: "how long does one control tick spend waiting on the cloud, and
is action caching / chunk prefetching needed to hit the control rate?"

Two probes, run back to back against the same server:

``ping``       -- tiny request, no payload, no model. Network floor (tunnel +
                  ZMQ + server loop), roughly the cost of one empty round trip.
``get_action`` -- a real observation (from ``--offline`` .parquet, or a
                  synthetic one) through the full path. Each sample is split
                  into client-side phases:

    encode  msgpack-serialize the observation
    wire    send() + recv() blocking: upload + server (deserialize, model,
            serialize) + download. The server's part can't be separated from
            here; ``wire - ping`` is an estimate of upload + inference.
    decode  msgpack-deserialize the reply

The report ends with a caching budget: given ``--hz`` control rate, how many
cached actions from the previous chunk are needed to cover the latency.

    inference-rtt --policy-host <tunnel-host> --offline dataset.parquet --task "..."
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import zmq

from .frame import StationFrame
from .policy_client import MsgSerializer, PolicyClient


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def timed_call(policy: PolicyClient, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
    """``PolicyClient.call_endpoint`` with per-phase timing. Raises like it does."""
    request: dict = {"endpoint": endpoint}
    if requires_input:
        request["data"] = data
    if policy.api_token:
        request["api_token"] = policy.api_token

    t0 = time.perf_counter()
    payload = MsgSerializer.to_bytes(request)
    t1 = time.perf_counter()
    try:
        policy.socket.send(payload)
        message = policy.socket.recv()
    except zmq.error.Again:
        policy._init_socket()
        raise
    t2 = time.perf_counter()
    if message == b"ERROR":
        raise RuntimeError("Server error. Make sure the correct policy server is running.")
    response = MsgSerializer.from_bytes(message)
    t3 = time.perf_counter()
    if isinstance(response, dict) and "error" in response:
        raise RuntimeError(f"Server error: {response['error']}")
    return {
        "encode_ms": _ms(t0, t1),
        "wire_ms": _ms(t1, t2),
        "decode_ms": _ms(t2, t3),
        "total_ms": _ms(t0, t3),
        "up_bytes": len(payload),
        "down_bytes": len(message),
        "response": response,
    }


def stats(xs: list[float]) -> dict:
    s = sorted(xs)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]

    return {
        "n": len(s),
        "min": s[0],
        "mean": statistics.fmean(s),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "max": s[-1],
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }


def build_observation(args) -> dict:
    image_keys = tuple(args.image_keys.split(","))
    if args.offline:
        frame = next(iter(StationFrame.iter_parquet(args.offline)))
        frame.task = frame.task or args.task
        return frame.to_observation(image_keys, args.state_key)

    # Synthetic: random pixels are incompressible, but the wire carries decoded
    # uint8 arrays (not JPEG), so payload size is the same as a real frame.
    h, w = (int(x) for x in args.image_size.split("x"))
    rng = np.random.default_rng(0)
    return {
        "video": {k: rng.integers(0, 256, (1, 1, h, w, 3), dtype=np.uint8) for k in image_keys},
        "state": {args.state_key: np.zeros((1, 1, args.state_dim), dtype=np.float32)},
        "language": {"task": [[args.task]]},
    }


def chunk_len(action: dict) -> int | None:
    """Steps in the predicted action chunk (axis 1 of (B, T, D)), if shaped that way."""
    first = next(iter(action.values()), None)
    return int(first.shape[1]) if isinstance(first, np.ndarray) and first.ndim >= 3 else None


def run(args) -> dict:
    result: dict = {"target": f"{args.policy_host}:{args.policy_port}"}
    with PolicyClient(
        host=args.policy_host, port=args.policy_port, timeout_ms=args.timeout_ms, api_token=args.api_token
    ) as policy:
        if not policy.ping():
            raise SystemExit(f"Can't reach policy server at {args.policy_host}:{args.policy_port}")

        for _ in range(args.warmup):
            timed_call(policy, "ping", requires_input=False)
        pings = [timed_call(policy, "ping", requires_input=False) for _ in range(args.pings)]
        result["ping_ms"] = stats([p["total_ms"] for p in pings])

        t0 = time.perf_counter()
        obs = build_observation(args)
        result["obs_build_ms"] = _ms(t0, time.perf_counter())  # JPEG decode etc.; cacheable client-side

        req = {"observation": obs, "options": None}
        for _ in range(args.warmup):  # first call pays CUDA / compile warm-up on the server
            timed_call(policy, "get_action", req)
        runs = [timed_call(policy, "get_action", req) for _ in range(args.samples)]

        for phase in ("encode_ms", "wire_ms", "decode_ms", "total_ms"):
            result[phase] = stats([r[phase] for r in runs])
        result["up_kb"] = runs[0]["up_bytes"] / 1024
        result["down_kb"] = runs[0]["down_bytes"] / 1024
        action, _info = runs[0]["response"]
        result["chunk_len"] = chunk_len(action)

    period = 1000.0 / args.hz
    total = result["total_ms"]
    result["budget"] = {
        "hz": args.hz,
        "period_ms": period,
        "inference_est_ms": max(0.0, result["wire_ms"]["p50"] - result["ping_ms"]["p50"]),
        "over_budget_pct": 100.0 * sum(r["total_ms"] > period for r in runs) / len(runs),
        # Actions to have queued so the next chunk lands before the cache runs dry.
        "cached_steps_needed_p50": int(np.ceil(total["p50"] / period)),
        "cached_steps_needed_p95": int(np.ceil(total["p95"] / period)),
    }
    return result


def _row(label: str, s: dict) -> str:
    return (
        f"  {label:<12}{s['min']:8.2f}{s['mean']:8.2f}{s['p50']:8.2f}"
        f"{s['p95']:8.2f}{s['p99']:8.2f}{s['max']:8.2f}{s['std']:8.2f}"
    )


def report(r: dict) -> None:
    b = r["budget"]
    print(f"\ntarget {r['target']}   (all times in ms)")
    print(f"  {'':<12}{'min':>8}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}{'std':>8}")
    print(_row("ping", r["ping_ms"]))
    for phase in ("encode_ms", "wire_ms", "decode_ms", "total_ms"):
        print(_row(phase.removesuffix("_ms").replace("total", "get_action"), r[phase]))
    print(f"\n  payload: up {r['up_kb']:.0f} KiB, down {r['down_kb']:.1f} KiB; obs build (JPEG decode) {r['obs_build_ms']:.1f} ms")
    print(f"  wire - ping ~ upload + inference: {b['inference_est_ms']:.1f} ms")
    chunk = r["chunk_len"]
    print(f"\ncaching budget @ {b['hz']:g} Hz (tick = {b['period_ms']:.0f} ms)")
    print(f"  ticks over budget:          {b['over_budget_pct']:.0f}%")
    print(f"  cached actions to cover p50: {b['cached_steps_needed_p50']}, p95: {b['cached_steps_needed_p95']}")
    if chunk:
        ok = chunk >= b["cached_steps_needed_p95"]
        print(f"  server returns {chunk}-step chunks -> {'enough to prefetch the next chunk' if ok else 'NOT enough to hide p95 latency'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-host", required=True)
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--api-token", default=None)
    p.add_argument("--timeout-ms", type=int, default=15000)
    p.add_argument("--task", default="measure rtt")
    p.add_argument("--image-keys", default="cam0,cam1")
    p.add_argument("--state-key", default="state")
    p.add_argument("--offline", default=None, help="Take the first frame of this .parquet as the observation")
    p.add_argument("--image-size", default="480x640", help="HxW of synthetic images when no --offline")
    p.add_argument("--state-dim", type=int, default=8, help="Joint count for the synthetic state")
    p.add_argument("--samples", type=int, default=50, help="Timed get_action calls")
    p.add_argument("--pings", type=int, default=100, help="Timed ping calls")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--hz", type=float, default=10.0, help="Control rate to budget against")
    p.add_argument("--json", action="store_true", help="Print raw results as JSON instead of the table")
    args = p.parse_args()

    result = run(args)
    print(json.dumps(result, indent=2)) if args.json else report(result)


if __name__ == "__main__":
    main()
