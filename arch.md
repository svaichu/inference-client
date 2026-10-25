# Architecture memory: GR00T remote inference for ElRobot

Context for picking this back up cold. This package is **GR00T-specific** —
it exists to drive NVIDIA GR00T N1.7 (via `finetune-groot/vendor/Isaac-GR00T`)
on ElRobot through norma-core's Station, not a generic multi-policy
framework. If a different model needs serving later, treat this as a
template to fork, not something to generalize preemptively.

## The split, and why

Two machines, cleanly separated by the one thing that has to cross a
network link:

- **Cloud server** (`finetune-groot/vendor/Isaac-GR00T`, runs on this GPU
  box): loads the checkpoint, serves `gr00t/eval/run_gr00t_server.py` — a ZMQ
  `REP` socket (`gr00t.policy.server_client.PolicyServer`), msgpack_numpy
  wire format, default port 5555. All of torch/transformers/CUDA lives here.
  See `finetune-groot/remote-inference/plan.md`.
- **Local client** (this package, `/workspace/inference-client`): runs next
  to the robot, where Station already runs. Talks to Station over its own
  protocol (port 8888, Gremlin/protobuf, pure stdlib — see below) and to the
  cloud server over one ZMQ round-trip per tick (`get_action`). No torch.

Rationale, from GR00T's own `real_world_deployment.md`: keep robot-side code
light, push only inference off-box. Camera frames + joint state (small,
JPEG-compressed) cross the wire as the observation; the action chunk comes
back; motor commands stay local.

## Why `PolicyClient` is copied, not imported

`gr00t.policy.__init__.py` eagerly does `from .gr00t_policy import Gr00tPolicy`,
which imports torch + transformers at module scope. There is no way to
`from gr00t.policy.server_client import PolicyClient` without that import
chain firing — even though `PolicyClient` itself only needs
`pyzmq`/`msgpack`/`msgpack_numpy`/`numpy`. Fix: `policy_client.py` here is a
lifted copy of `MsgSerializer` + `PolicyClient` from
`vendor/Isaac-GR00T/gr00t/policy/server_client.py` (NVIDIA, Apache-2.0), with
the `BasePolicy` ABC plumbing dropped (this client only ever calls
`get_action`/`get_modality_config`/`ping`/`reset`, never needs strict
server-side validation). Verified in a clean venv: installs 5 small
packages, zero torch/transformers in `sys.modules`.

If NVIDIA's server-side wire format changes, this file goes stale silently
— there's no shared source of truth. Re-diff against
`vendor/Isaac-GR00T/gr00t/policy/server_client.py` if `get_action` calls
start failing with deserialization errors after a GR00T repo bump.

## Station / norma-core, what's actually there

`/workspace/norma-core` is the **code and docs**, not a running instance —
no live Station daemon, no recorded `.parquet` found anywhere on this box as
of 2026-10-24. Everything in `frame.py`/`station_live.py` is built from
reading norma-core's source directly, not from an actual running system:

- `protobufs/drivers/inferences/normvla.proto` — the live `Frame` schema
  (joints + 2 JPEGs, no task string).
- `docs/datasets/export-parquet/datasets-export.md` — the recorded dataset
  schema (same joints/images structs + `task` + episode timing columns),
  produced by Station's `dataset-generator` binary.
- `software/station/shared/station_py/client.py` — `new_station_client(addr,
  logger)` (default port 8888), `send_commands(client, [cmd])`,
  `client.read_from_tail(queue_id, ...)`. Pure stdlib (asyncio/struct/socket)
  — Gremlin is norma-core's own wire format, not the `protobuf` pip package,
  so importing it adds no new dependency.
- `software/ai/smolvla_py/scripts/run_policy.py` — the existing reference
  pattern (SmolVLA run in-process). `bridge.py`'s live loop mirrors its
  fetch/predict/clip/send structure, swapping the in-process
  `predict_action_chunk` for one `PolicyClient.get_action()` call.

"ElRobot" (user's "elrobot") = norma-core's actual 7+1 DoF arm hardware
project, not a typo.

## Data format — single source of truth: `StationFrame`

Both the live protobuf `Frame` and recorded `.parquet` rows describe the
identical schema (joints: range_min/max, position(_norm), goal(_norm),
current_ma, velocity; images: JPEG bytes; parquet additionally carries
`task`). `frame.py:StationFrame` is the one place that owns this, with
`.from_live()` / `.iter_parquet()` constructors and a single
`.to_observation()` that builds GR00T's expected
`{"video": {...uint8 (1,1,H,W,3)...}, "state": {...float32 (1,1,D)...},
"language": {"task": [[str]]}}`. Full field-by-field spec in
`DATA_FORMAT.md`.

**Known hazard, not yet solved by code**: joint ordering and camera-key
naming must match whatever the checkpoint was trained with. There is no
name-based remapping — `to_observation` trusts `Frame` order and whatever
`--image-keys`/`--state-key` you pass. The bridge calls
`get_modality_config()` at startup and prints it so you can check by hand,
but nothing validates the match automatically. If actions come back
nonsensical, check this first.

## Open items (as of 2026-10-24)

- A GR00T checkpoint for ElRobot now exists: `finetune-groot` `outputs/checkpoints/elrobot_pizza*`
  (`new_embodiment`, 2 cameras `cam0,cam1`, 8 joints, ABSOLUTE `goal_norm` actions). Station may still
  publish more than 2 images; only the first 2 are fed to the model (`--image-keys cam0,cam1`).
- Tunnel between cloud server and local client (Tailscale / SSH reverse
  tunnel / ngrok) was discussed but not set up or decided in this session.
- `src/inference_client/station_live.py` was written from reading
  `run_policy.py` and the proto files, not tested against a live Station —
  treat the exact `station_py`/protobuf call shapes as best-effort until
  run once for real.
