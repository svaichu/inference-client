"""Talk to a running norma-core Station daemon.

This is the only module that imports norma-core code (``station_py`` + the
generated protobuf package under ``target/gen_python``). Both are pure
stdlib on norma-core's side (asyncio/struct/socket/dataclasses — Gremlin is a
custom wire format, not the ``protobuf`` pip package), so importing them adds
no new dependency here. But they live in the norma-core checkout, not in this
package, so the checkout's path must be on ``sys.path`` before this module is
imported -- ``bridge.py`` does that via ``--norma-core-path`` (or the
``NORMA_CORE_PATH`` env var) before importing this module.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def add_norma_core_to_path(norma_core_path: str) -> None:
    root = Path(norma_core_path).resolve()
    station_shared = root / "software" / "station" / "shared"
    for p in (str(root), str(station_shared)):
        if p not in sys.path:
            sys.path.insert(0, p)


# Imported lazily, after add_norma_core_to_path() has run -- see bridge.py.
def _imports():
    from station_py import new_station_client, send_commands  # type: ignore
    from target.gen_python.protobuf.drivers.inferences import normvla  # type: ignore
    from target.gen_python.protobuf.drivers.st3215 import st3215  # type: ignore
    from target.gen_python.protobuf.station import commands, drivers  # type: ignore

    return new_station_client, send_commands, normvla, st3215, commands, drivers


QUEUE_ID = "inference/normvla"
ST3215_TARGET_POS_REGISTER = 0x2A


class StationLink:
    """Fetch frames from / send commands to one Station daemon connection."""

    def __init__(self, client):
        self._client = client
        (
            _,
            self._send_commands,
            self._normvla,
            self._st3215,
            self._commands,
            self._drivers,
        ) = _imports()

    @classmethod
    async def connect(cls, server: str, logger) -> "StationLink":
        new_station_client, *_ = _imports()
        client = await new_station_client(server, logger)
        return cls(client)

    async def fetch_frame(self, timeout: float = 5.0):
        """Return a raw ``normvla.FrameReader`` -- pass to StationFrame.from_live()."""
        qr = self._client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
        if entry is None:
            raise RuntimeError(f"{QUEUE_ID} closed without delivering a frame ({qr.err})")
        return self._normvla.FrameReader(memoryview(bytes(entry.Data)))

    async def fetch_fresh_frame(self, last_id: bytes, timeout: float = 5.0, retry_ms: int = 20):
        """Poll until a frame with a different ``global_frame_id`` than ``last_id`` arrives.

        ``inference/normvla`` updates at ~10 Hz; fetching faster just re-reads
        the same frame.
        """
        while True:
            frame = await self.fetch_frame(timeout)
            fid = bytes(frame.get_global_frame_id())
            if fid != last_id:
                return frame, fid
            await asyncio.sleep(retry_ms / 1000.0)

    def build_sync_write_command(self, bus_serial: str, motor_ids: list[int], raw_goals: list[int]):
        """One atomic sync-write of all joints' target position register."""
        motors = [
            self._st3215.ST3215SyncWriteCommand_MotorWrite(
                motor_id=mid, value=raw.to_bytes(2, byteorder="little")
            )
            for mid, raw in zip(motor_ids, raw_goals)
        ]
        sync = self._st3215.ST3215SyncWriteCommand(address=ST3215_TARGET_POS_REGISTER, motors=motors)
        cmd = self._st3215.Command(target_bus_serial=bus_serial, sync_write=sync)
        return self._commands.DriverCommand(
            type=self._drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode()
        )

    async def send(self, command) -> None:
        await self._send_commands(self._client, [command])


def unnormalize_goal(goal_norm: float, range_min: int, range_max: int) -> int:
    raw = int(round(range_min + float(goal_norm) * (range_max - range_min)))
    return max(range_min, min(range_max, raw))
