# inference-client

The local-client half of remote GR00T inference for ElRobot. Runs on the
machine wired to the robot (where Station already runs). Standalone env —
**no torch, no transformers, no CUDA**; those stay on the cloud server
(`finetune-groot/vendor/Isaac-GR00T`, `gr00t/eval/run_gr00t_server.py`).

See `/workspace/finetune-groot/remote-inference/plan.md` for the overall
cloud/local split, and `DATA_FORMAT.md` here for the observation schema.

## Install

```bash
cd /workspace/inference-client
uv venv && uv pip install -e .            # live mode only
uv pip install -e ".[offline]"            # + offline .parquet replay
```

## Run

Live, against hardware (needs a norma-core checkout for `station_py` +
generated protobufs — pure stdlib on their end, so this adds no new deps):

```bash
inference-bridge \
  --policy-host <tunnel-host> --policy-port 5555 \
  --server localhost \
  --norma-core-path /path/to/norma-core \
  --bus-serial 5AB9068807 \
  --task "place the yellow cube on top of the other" \
  --auto --max-ticks 200
```

Offline, replaying a recorded dataset against the remote policy server with
no robot or Station connection — use this first to check the tunnel +
server + conversion code:

```bash
inference-bridge \
  --policy-host <tunnel-host> --policy-port 5555 \
  --offline ~/datasets/dataset-cube.parquet \
  --task "put the cube inside box"
```

## Layout

```
src/inference_client/
  policy_client.py   # ZMQ client for the GR00T server, lifted out so it never imports torch
  frame.py            # StationFrame: the one obs schema, from live or .parquet, -> GR00T obs dict
  station_live.py     # live-only: wraps norma-core's station_py + protobuf (lazy import)
  bridge.py            # CLI: live control loop, or --offline replay
```
