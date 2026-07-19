"""Single-writer, multi-reader shared-memory ring buffer for tensor-parallel
control-plane broadcast.

This replaces per-step gloo/NCCL command broadcasts (which cost a full collective
each step and make continuous batching slow) with a POSIX shared-memory ring
buffer. Rank 0 (the writer) enqueues a serialized command; worker ranks (readers)
dequeue it. The model-forward NCCL all-reduces are unchanged — this only moves the
control plane (which row decodes, which prefills, the token ids) off the
collective path.

Protocol mirrors vLLM's ShmRingBuffer. Each slot has a 1-byte written flag plus
one read flag per reader. The writer waits until a slot is free (unwritten, or
read by all readers), writes the payload, resets reader flags, then sets the
written flag last. Readers spin until the written flag is set and their own read
flag is clear, copy the payload out, then set their read flag.

Memory ordering: these are x86_64 hosts (TSO), so plain stores are not reordered
and no explicit fence is required between the data write and the flag write. The
GIL further serializes Python-level access within a process.
"""

from __future__ import annotations

import os
import pickle
import struct
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Iterator
from unittest.mock import patch

# Per-slot header: 4-byte unsigned length prefix for the payload.
_LEN_HEADER = 4
_LEN_STRUCT = struct.Struct("<I")


class ShmRingBuffer:
    """A fixed-capacity ring buffer in POSIX shared memory.

    One writer (rank 0) and ``n_reader`` readers (the worker ranks). Sizes are
    fixed at creation; payloads larger than ``max_chunk_bytes - 4`` are rejected
    by ``enqueue`` so the caller can fall back to the gloo transport.
    """

    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ) -> None:
        if n_reader < 1:
            raise ValueError("n_reader must be positive")
        if max_chunk_bytes <= _LEN_HEADER:
            raise ValueError("max_chunk_bytes must exceed the length header")
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.data_offset = 0
        self.metadata_offset = max_chunk_bytes * max_chunks
        self.total_bytes = (max_chunk_bytes + self.metadata_size) * max_chunks

        if name is None:
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(create=True, size=self.total_bytes)
            # Zero the metadata region so every slot starts writable/unread.
            view = memoryview(self.shared_memory.buf)[self.metadata_offset :]
            for i in range(len(view)):
                view[i] = 0
            view.release()
        else:
            self.is_creator = False
            # Python's resource_tracker over-eagerly unlinks shm it did not
            # create; suppress its registration when attaching (vLLM workaround).
            with patch("multiprocessing.resource_tracker.register", lambda *a, **k: None):
                self.shared_memory = shared_memory.SharedMemory(name=name)

    @property
    def name(self) -> str:
        return self.shared_memory.name

    def handle(self) -> tuple[int, int, int, str]:
        return (self.n_reader, self.max_chunk_bytes, self.max_chunks, self.shared_memory.name)

    @staticmethod
    def from_handle(handle: tuple[int, int, int, str]) -> "ShmRingBuffer":
        n_reader, max_chunk_bytes, max_chunks, name = handle
        return ShmRingBuffer(n_reader, max_chunk_bytes, max_chunks, name=name)

    def _data_slice(self, idx: int) -> memoryview:
        start = self.data_offset + idx * self.max_chunk_bytes
        return memoryview(self.shared_memory.buf)[start : start + self.max_chunk_bytes]

    def _meta_slice(self, idx: int) -> memoryview:
        start = self.metadata_offset + idx * self.metadata_size
        return memoryview(self.shared_memory.buf)[start : start + self.metadata_size]

    def close(self) -> None:
        try:
            self.shared_memory.close()
        except Exception:
            pass
        if self.is_creator:
            try:
                self.shared_memory.unlink()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()


class ShmWriter:
    """Writer side (rank 0). Not thread-safe; call from a single thread."""

    def __init__(self, buffer: ShmRingBuffer) -> None:
        self.buffer = buffer
        self._idx = 0

    @contextmanager
    def _acquire_write(self, timeout: float | None) -> Iterator[memoryview]:
        start = time.monotonic()
        while True:
            meta = self.buffer._meta_slice(self._idx)
            written = meta[0]
            read_count = sum(meta[1:])
            if written and read_count != self.buffer.n_reader:
                # Slot still being read by some reader; wait.
                meta.release()
                if timeout is not None and time.monotonic() - start > timeout:
                    raise TimeoutError("shm ring buffer write timed out")
                time.sleep(0)
                continue
            # Slot is free (never written, or read by everyone).
            meta[0] = 0  # mark not-written while we fill it
            data = self.buffer._data_slice(self._idx)
            try:
                yield data
            finally:
                data.release()
            # Reset reader flags first, then publish (written flag last).
            for i in range(1, self.buffer.n_reader + 1):
                meta[i] = 0
            meta[0] = 1
            meta.release()
            self._idx = (self._idx + 1) % self.buffer.max_chunks
            return

    def enqueue(self, payload: bytes, timeout: float | None = None) -> bool:
        """Write ``payload`` into the next slot. Returns False if it does not fit
        (caller should fall back to another transport)."""
        if len(payload) + _LEN_HEADER > self.buffer.max_chunk_bytes:
            return False
        with self._acquire_write(timeout) as data:
            data[:_LEN_HEADER] = _LEN_STRUCT.pack(len(payload))
            data[_LEN_HEADER : _LEN_HEADER + len(payload)] = payload
        return True


class ShmReader:
    """Reader side (a worker rank). One reader per rank; ``reader_id`` in
    ``[0, n_reader)`` selects this rank's flag byte."""

    def __init__(self, buffer: ShmRingBuffer, reader_id: int) -> None:
        if not (0 <= reader_id < buffer.n_reader):
            raise ValueError("reader_id out of range")
        self.buffer = buffer
        self.reader_id = reader_id
        self._idx = 0

    def dequeue(self, timeout: float | None = None) -> bytes:
        start = time.monotonic()
        flag_byte = 1 + self.reader_id
        try:
            yield_interval = max(
                1,
                int(os.environ.get("TORCHINFERNO_SHM_POLL_YIELD_INTERVAL", "1")),
            )
        except ValueError:
            yield_interval = 1
        empty_polls = 0
        while True:
            meta = self.buffer._meta_slice(self._idx)
            written = meta[0]
            already_read = meta[flag_byte]
            if (not written) or already_read:
                meta.release()
                empty_polls += 1
                if empty_polls >= yield_interval:
                    if timeout is not None and time.monotonic() - start > timeout:
                        raise TimeoutError("shm ring buffer read timed out")
                    time.sleep(0)
                    empty_polls = 0
                continue
            data = self.buffer._data_slice(self._idx)
            (length,) = _LEN_STRUCT.unpack_from(data, 0)
            payload = bytes(data[_LEN_HEADER : _LEN_HEADER + length])
            data.release()
            meta[flag_byte] = 1  # mark read by this reader
            meta.release()
            self._idx = (self._idx + 1) % self.buffer.max_chunks
            return payload


def pickle_dumps(obj: object) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_loads(data: bytes) -> object:
    return pickle.loads(data)
