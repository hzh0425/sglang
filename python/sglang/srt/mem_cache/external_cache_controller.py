from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

import torch


@dataclasses.dataclass(frozen=True, kw_only=True)
class BackupRequest:
    node_id: int
    write_back: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class BackupResult:
    backed_up_tokens: int = 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadBackRequest:
    node_id: int
    mem_quota: Optional[int] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadBackResult:
    loaded: bool = False
    device_indices: Optional[torch.Tensor] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExternalCacheProgress:
    completed_writes: int = 0
    completed_loads: int = 0
    completed_storage_ops: int = 0


@runtime_checkable
class ExternalCacheTreeOps(Protocol):
    def contains_node(self, node_id: int) -> bool:
        ...

    def get_node_token_count(self, node_id: int) -> int:
        ...


class BaseExternalCacheController(ABC):
    @abstractmethod
    def write_backup(
        self, request: BackupRequest, tree_ops: ExternalCacheTreeOps
    ) -> BackupResult:
        raise NotImplementedError

    @abstractmethod
    def load_back(
        self, request: LoadBackRequest, tree_ops: ExternalCacheTreeOps
    ) -> LoadBackResult:
        raise NotImplementedError

    @abstractmethod
    def begin_pending_loads(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def poll(
        self, tree_ops: ExternalCacheTreeOps, *, wait: bool = False
    ) -> ExternalCacheProgress:
        raise NotImplementedError

    @abstractmethod
    def abort_request(
        self, request_id: str, tree_ops: ExternalCacheTreeOps
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError


class NoopExternalCacheController(BaseExternalCacheController):
    def write_backup(
        self, request: BackupRequest, tree_ops: ExternalCacheTreeOps
    ) -> BackupResult:
        return BackupResult()

    def load_back(
        self, request: LoadBackRequest, tree_ops: ExternalCacheTreeOps
    ) -> LoadBackResult:
        return LoadBackResult()

    def begin_pending_loads(self) -> int:
        return 0

    def poll(
        self, tree_ops: ExternalCacheTreeOps, *, wait: bool = False
    ) -> ExternalCacheProgress:
        return ExternalCacheProgress()

    def abort_request(
        self, request_id: str, tree_ops: ExternalCacheTreeOps
    ) -> None:
        return None

    def reset(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
