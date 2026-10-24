"""The one data shape this package passes around: a Station observation frame.

Both data sources Station produces describe the *same* schema — this module
is the single place that owns it, so live and offline code paths produce
identical ``StationFrame`` objects and feed the policy identically.

1. **Live**, from the running Station daemon's ``inference/normvla`` ZMQ/Gremlin
   queue: a ``normvla.Frame`` protobuf message (joints + camera JPEGs, ~10 Hz,
   no task string — the task is fixed for the run and passed on the CLI).
2. **Offline**, from a recorded ``.parquet`` dataset (norma-core's
   ``dataset-generator`` output — see norma-core's
   ``docs/datasets/export-parquet/datasets-export.md``): one row per frame,
   same joint/image structs, plus ``task`` and episode timing columns.

See ``DATA_FORMAT.md`` at the repo root for the full field-by-field spec and
for how a ``StationFrame`` is converted into the GR00T policy's expected
``{"video": ..., "state": ..., "language": ...}`` observation dict.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage


@dataclass
class Joint:
    range_min: int
    range_max: int
    position: int
    position_norm: float
    goal: int
    goal_norm: float
    current_ma: int = 0
    velocity: int = 0


@dataclass
class CameraImage:
    jpeg: bytes

    def decode(self) -> np.ndarray:
        """JPEG bytes -> HWC uint8 RGB array. GR00T wants decoded images, not JPEG."""
        with PILImage.open(io.BytesIO(self.jpeg)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)


@dataclass
class StationFrame:
    joints: list[Joint]
    images: list[CameraImage]
    task: str | None = None

    @classmethod
    def from_live(cls, frame_reader, task: str | None = None) -> "StationFrame":
        """Build from a ``normvla.FrameReader`` (the live protobuf message).

        ``frame_reader`` is duck-typed: anything with ``.get_joints()`` and
        ``.get_images()`` returning objects with the matching getters works,
        so this doesn't import norma-core's generated protobuf code — the
        caller (``station_live.py``) does that, since it needs norma-core on
        ``sys.path`` anyway.
        """
        joints = [
            Joint(
                range_min=int(j.get_range_min()),
                range_max=int(j.get_range_max()),
                position=int(j.get_position()),
                position_norm=float(j.get_position_norm()),
                goal=int(j.get_goal()),
                goal_norm=float(j.get_goal_norm()),
                current_ma=int(j.get_current_ma()),
                velocity=int(j.get_velocity()),
            )
            for j in (frame_reader.get_joints() or [])
        ]
        images = [CameraImage(jpeg=bytes(im.get_jpeg())) for im in (frame_reader.get_images() or [])]
        return cls(joints=joints, images=images, task=task)

    @classmethod
    def iter_parquet(cls, path: str):
        """Yield one ``StationFrame`` per row of a recorded dataset.

        Requires the ``offline`` extra (``pyarrow``). Schema per
        norma-core's dataset-export doc — see DATA_FORMAT.md.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for batch in table.to_batches():
            cols = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                joints = [
                    Joint(
                        range_min=j["range_min"],
                        range_max=j["range_max"],
                        position=j["position"],
                        position_norm=j["position_norm"],
                        goal=j["goal"],
                        goal_norm=j["goal_norm"],
                        current_ma=j.get("current_ma", 0),
                        velocity=j.get("velocity", 0),
                    )
                    for j in cols["joints"][i]
                ]
                images = [CameraImage(jpeg=img["jpeg"]) for img in cols["images"][i]]
                task = cols["task"][i] if "task" in cols else None
                yield cls(joints=joints, images=images, task=task)

    def to_observation(self, image_keys: tuple[str, ...], state_key: str = "state") -> dict:
        """Build the nested dict GR00T's ``Gr00tPolicy.get_action`` expects.

        Shapes (batch=1, time=1, per Gr00tPolicy.check_observation):
          video[cam]:   uint8  (1, 1, H, W, 3)
          state[key]:   float32 (1, 1, D)     -- D = number of joints, in Frame order
          language.task: [[task_str]]

        Caller must ensure ``image_keys`` (and joint ordering) match the
        checkpoint's training config -- call ``policy.get_modality_config()``
        once to confirm, don't hardcode blindly.
        """
        if len(self.images) < len(image_keys):
            raise ValueError(f"Frame has {len(self.images)} images but {len(image_keys)} keys requested")
        if self.task is None:
            raise ValueError("StationFrame.task is required to build an observation (pass --task)")

        video = {key: self.images[i].decode()[None, None, ...] for i, key in enumerate(image_keys)}
        state = {
            state_key: np.array([[[j.position_norm for j in self.joints]]], dtype=np.float32)
        }
        language = {"task": [[self.task]]}
        return {"video": video, "state": state, "language": language}

    def joint_ranges(self) -> list[tuple[int, int]]:
        return [(j.range_min, j.range_max) for j in self.joints]
