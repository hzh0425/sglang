from __future__ import annotations

import json
import mmap
import os
import time
from typing import Any, Optional

import torch


_META_CAPACITY = 256 * 1024


class MmapDumper:
    def __init__(self, dump_dir: Optional[str] = None) -> None:
        self._dump_dir = None
        self._pid = os.getpid()
        self._scalars: dict[str, Any] = {}
        self._tensor_meta: dict[str, Any] = {}
        self._tensor_mmaps: dict[str, Any] = {}
        self._meta_mmap = None
        self._meta_fd = None
        if dump_dir:
            self.set_dir(dump_dir)

    def set_dir(self, dump_dir: str) -> None:
        if self._dump_dir == dump_dir:
            return

        os.makedirs(dump_dir, exist_ok=True)
        self._dump_dir = dump_dir
        path = os.path.join(dump_dir, f"pid{self._pid}_meta.json.mmap")
        self._meta_fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(self._meta_fd, _META_CAPACITY)
        self._meta_mmap = mmap.mmap(
            self._meta_fd, _META_CAPACITY, access=mmap.ACCESS_WRITE
        )

    def is_active(self) -> bool:
        return self._dump_dir is not None and self._meta_mmap is not None

    def dump(self, items: dict[str, Any]) -> None:
        if not self.is_active():
            return

        t0 = time.perf_counter()
        for name, value in items.items():
            if isinstance(value, torch.Tensor):
                self._dump_tensor(name, value)
            else:
                self._scalars[name] = _jsonify(value)
        self._flush_meta()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[MmapDumper pid={self._pid}] dumped {len(items)} items "
            f"in {elapsed_ms:.2f} ms",
            flush=True,
        )

    def _dump_tensor(self, name: str, tensor: torch.Tensor) -> None:
        import numpy as np

        cpu_tensor = tensor.detach().cpu().contiguous()
        if cpu_tensor.dim() == 0:
            raw_tensor = cpu_tensor.reshape(1).view(torch.uint8)
        else:
            raw_tensor = cpu_tensor.view(torch.uint8)
        raw_tensor = raw_tensor.reshape(-1)
        nbytes = raw_tensor.numel()
        alloc_bytes = max(nbytes, 1)

        bin_path = os.path.join(self._dump_dir, f"pid{self._pid}_{name}.bin")
        entry = self._tensor_mmaps.get(name)
        if entry is None or entry["capacity"] < alloc_bytes:
            if entry is not None:
                entry["mmap"].close()
                os.close(entry["fd"])
            capacity = max(alloc_bytes * 2, 4096)
            fd = os.open(bin_path, os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(fd, capacity)
            entry = {
                "fd": fd,
                "mmap": mmap.mmap(fd, capacity, access=mmap.ACCESS_WRITE),
                "capacity": capacity,
            }
            self._tensor_mmaps[name] = entry

        if nbytes:
            src = raw_tensor.numpy()
            dst = np.frombuffer(entry["mmap"], dtype=np.uint8, count=nbytes)
            np.copyto(dst, src)

        self._tensor_meta[name] = {
            "shape": list(cpu_tensor.shape),
            "stride": list(cpu_tensor.stride()),
            "dtype": str(cpu_tensor.dtype),
            "nbytes": nbytes,
            "bin_filename": os.path.basename(bin_path),
        }

    def _flush_meta(self) -> None:
        meta = {
            "pid": self._pid,
            "scalars": self._scalars,
            "tensors": self._tensor_meta,
        }
        payload = json.dumps(meta).encode("utf-8")
        n = len(payload)
        assert n + 4 <= _META_CAPACITY, f"mmap dumper meta too big: {n}"
        self._meta_mmap[0:4] = n.to_bytes(4, "little")
        self._meta_mmap[4 : 4 + n] = payload
        self._meta_mmap.flush()


def _jsonify(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return repr(value)


_TORCH_DTYPE_TO_TORCH = {
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "torch.float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def read_dump(dump_dir: str, pid: int) -> dict[str, Any]:
    meta_path = os.path.join(dump_dir, f"pid{pid}_meta.json.mmap")
    with open(meta_path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        meta = json.loads(f.read(n).decode("utf-8"))

    tensors = {}
    for name, info in meta["tensors"].items():
        torch_dtype = _TORCH_DTYPE_TO_TORCH[info["dtype"]]
        assert torch_dtype is not None, f"unsupported dtype in dump: {info['dtype']}"
        if info["nbytes"] == 0:
            tensors[name] = torch.empty(info["shape"], dtype=torch_dtype)
            continue
        raw_path = os.path.join(dump_dir, info["bin_filename"])
        with open(raw_path, "rb") as f:
            raw = torch.frombuffer(bytearray(f.read(info["nbytes"])), dtype=torch.uint8)
        tensors[name] = raw.view(torch_dtype).reshape(info["shape"])
    return {"scalars": meta["scalars"], "tensors": tensors}


def list_dump_pids(dump_dir: str) -> list[int]:
    pids = []
    for filename in os.listdir(dump_dir):
        if filename.startswith("pid") and filename.endswith("_meta.json.mmap"):
            pids.append(int(filename[len("pid") : -len("_meta.json.mmap")]))
    return sorted(pids)
