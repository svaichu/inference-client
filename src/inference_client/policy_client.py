"""Standalone ZMQ client for a GR00T policy server.

Lifted from ``vendor/Isaac-GR00T/gr00t/policy/server_client.py`` (NVIDIA,
Apache-2.0) — same wire protocol, msgpack_numpy framing, and
``allow_pickle=False`` refusal. The original class can't be imported in
isolation: ``gr00t.policy.__init__`` eagerly imports ``Gr00tPolicy``, which
imports torch and transformers at module scope, just to get a plain ZMQ
client. This file has none of that: only ``pyzmq``, ``msgpack``,
``msgpack-numpy``, and ``numpy``.

Dropped relative to the original: the ``BasePolicy`` ABC plumbing (strict
validation, ``ModalityConfig`` round-tripping) — this client is meant to call
exactly one endpoint (``get_action``), fetch ``get_modality_config`` once at
startup, and otherwise stay out of the way.
"""

from __future__ import annotations

import functools
import io
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class MsgSerializer:
    """msgpack_numpy wire format with a hard ``allow_pickle=False`` boundary.

    msgpack_numpy's encode/decode hooks route object-dtype ndarrays through
    ``pickle`` before any user hook can intervene; the checks below run
    first and refuse that path, matching the server's own guard.
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=mnp.encode)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(MsgSerializer._safe_decode, chain=mnp.decode)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj, chain=None):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                f"convert to a concrete numeric dtype before sending."
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        if isinstance(obj, dict):
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload (pickle-bearing).")
        return mnp.decode(obj, chain=chain)


class PolicyClient:
    """REQ-socket client for a ``gr00t.eval.run_gr00t_server`` instance.

    Usage::

        with PolicyClient(host="<tunnel-host>", port=5555) as policy:
            assert policy.ping()
            modality = policy.get_modality_config()   # call once, to discover key names
            action, info = policy.get_action(observation)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self._closed = False
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_modality_config(self) -> dict:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return tuple(response)  # msgpack returns a list; normalize to (action, info)

    def reset(self, options: dict | None = None) -> dict:
        return self.call_endpoint("reset", {"options": options})

    def kill_server(self) -> None:
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # REQ socket is now stuck waiting for a reply that will never
            # arrive; recreate it so the *next* call can send again.
            self._init_socket()
            raise

        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the correct policy server is running.")
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
