# Expected data format

One schema, two sources. Both go through `StationFrame` (`src/inference_client/frame.py`)
so the conversion to a GR00T observation is written once.

## Source 1: live Station daemon

Station publishes to the `inference/normvla` Gremlin/ZMQ queue at ~10 Hz. Wire
schema: `normvla.proto` in norma-core
(`protobufs/drivers/inferences/normvla.proto`):

```protobuf
message Frame {
  bytes  global_frame_id    = 1;
  uint64 monotonic_stamp_ns = 2;
  repeated Joint joints = 10;
  repeated Image images = 11;
}

message Joint {
  uint32 range_min;      uint32 range_max;
  uint32 position;       float  position_norm;   // position, in [0,1] over [range_min, range_max]
  uint32 goal;            float  goal_norm;
  uint32 current_ma;     uint32 velocity;
  uint64 monotonic_stamp_ns;
}

message Image {
  bytes jpeg;             // 224x224
  uint64 monotonic_stamp_ns;
}
```

No `task` field here — the live Frame doesn't carry it; the instruction is
fixed for the run and passed on the bridge's `--task` flag.

## Source 2: recorded `.parquet` dataset

Produced by Station's `dataset-generator` (see norma-core's
`docs/datasets/export-parquet/datasets-export.md`). One row = one frame, same
`joints`/`images` structs as above, plus:

| column | type | meaning |
|---|---|---|
| `episode_start_ns` | uint64 | wall-clock UNIX-ns of the episode's first frame |
| `global_frame_id` | binary | LE-encoded frame id |
| `timestamp_ns_since_episode_start` | uint64 | per-episode time, starts at 0 |
| `joints` | list\<struct\> | same fields as the live `Joint` message |
| `images` | list\<struct{jpeg: binary}\> | one JPEG per camera |
| `task` | string | natural-language task — present here, absent live |

Use this for offline testing (`inference-bridge --offline path/to.parquet`):
exercises the exact same obs-building + policy-call code, no hardware or
Station connection needed.

## What `StationFrame.to_observation()` builds

GR00T's `Gr00tPolicy.check_observation` (vendor/Isaac-GR00T) expects:

```python
{
  "video":    {cam_key: np.ndarray[uint8,   (B=1, T=1, H, W, C=3)]},   # decoded, not JPEG
  "state":    {state_key: np.ndarray[float32, (B=1, T=1, D)]},         # D = number of joints
  "language": {"task": [[task_str]]},                                 # shape (B, T)
}
```

Translation rules, enforced in `StationFrame.to_observation`:

- **Images**: JPEG bytes are decoded to `HWC uint8` RGB (`PIL.Image` ->
  `np.asarray`). The server's own `check_observation` rejects anything else —
  GR00T does not accept raw JPEG.
- **State**: built directly from each joint's `position_norm`, in `Frame`
  order. **This order must match however the checkpoint was trained** — there
  is no name-based remapping here. If training used a different joint order
  or a subset, fix it before this call, not after.
- **Images keys / camera order**: `--image-keys` must list the checkpoint's
  camera names in the same order as `Frame.images` arrives (camera 0, camera
  1, ...). Wrong order silently feeds the wrong camera into the wrong model
  input.
- **`state_key`**: must be the single state modality key the checkpoint
  expects (default `"state"` — but confirm, don't assume).

**Don't guess these names or orderings.** Call
`policy_client.get_modality_config()` once at startup (the bridge script
already does this and prints it) and match `--image-keys` / `--state-key` /
joint order to what it reports, rather than hardcoding blind guesses.

## What comes back

`PolicyClient.get_action()` returns `(action, info)` where `action` is
`{action_key: np.ndarray[float32, (B=1, T=chunk_len, D)]}` — a chunk of future
goal-normalized values, one scalar per joint, in the same `D`-length order as
the state went in. The bridge takes `action[action_key][0, 0]` (the first
step of the chunk), unnormalizes it against each joint's live
`(range_min, range_max)`, and sync-writes the result.
