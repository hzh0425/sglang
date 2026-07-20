from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolTransfer
    from sglang.srt.mem_cache.radix_cache import RadixKey


@dataclass(frozen=True)
class ExternalCacheHit:
    page_keys: list[str]
    hit_tokens: int


@dataclass(frozen=True)
class ExternalStoreCompletion:
    operation_id: int
    success: bool


class ExternalCacheConnector(ABC):
    @abstractmethod
    def query(
        self,
        key: RadixKey,
        local_tokens: int,
        transfers: Sequence[PoolTransfer],
    ) -> ExternalCacheHit:
        pass

    @abstractmethod
    def load(
        self,
        hit: ExternalCacheHit,
        transfers: Sequence[PoolTransfer],
    ) -> bool:
        pass

    @abstractmethod
    def store_async(
        self,
        key: RadixKey,
        transfers: Sequence[PoolTransfer],
    ) -> int:
        pass

    @abstractmethod
    def poll_completed(self) -> list[ExternalStoreCompletion]:
        pass

    @abstractmethod
    def wait_for_all_stores(self) -> list[ExternalStoreCompletion]:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset connector-local state without deleting remote objects."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass
