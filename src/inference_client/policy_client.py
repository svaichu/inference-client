"""Standalone ZMQ client for a GR00T policy server.

Lifted from ``vendor/Isaac-GR00T/gr00t/policy/server_client.py`` (NVIDIA,
Apache-2.0) — same wire protocol: msgpack, with ndarrays framed as
``{"__ndarray_class__": True, "as_npy": <np.save bytes>}`` and loaded with
``allow_pickle=False``. (Not msgpack_numpy: the server only decodes its own
``__ndarray_class__`` framing, and passes anything else through as a dict.)
The original class can't be imported in isolation: ``gr00t.policy.__init__``
eagerly imports ``Gr00tPolicy``, which imports torch and transformers at
module scope, just to get a plain ZMQ client. This file has none of that:
only ``pyzmq``, ``msgpack``, and ``numpy``.

Dropped relative to the original: the ``BasePolicy`` ABC plumbing (strict
validation, ``ModalityConfig`` round-tripping — a ``ModalityConfig`` from the
server comes back as its plain JSON dict) — this client is meant to call
exactly one endpoint (``get_action``), fetch ``get_modality_config`` once at
startup, and otherwise stay out of the way.
"""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    """The server's own wire format (``gr00t.policy.server_client.MsgSerializer``).

    ``np.save`` / ``np.load`` run with ``allow_pickle=False``, so object-dtype
    ndarrays are refused in both directions, matching the server's guard.
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


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
        # Close any previous (possibly stuck) socket first: left open with the
        # default infinite linger, it makes context.term() in close() hang forever.
        if getattr(self, "socket", None) is not None:
            self.socket.close(linger=0)
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
