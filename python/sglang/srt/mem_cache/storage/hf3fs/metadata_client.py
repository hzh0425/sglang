import requests
from typing import List, Optional, Tuple

class MetadataClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def _post(self, endpoint: str, json_data: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}/{endpoint}", json=json_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to metadata server: {e}") from e

    def _get(self, endpoint: str) -> dict:
        try:
            response = requests.get(f"{self.base_url}/{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to metadata server: {e}") from e

    def initialize(self, rank: int, num_pages: int) -> None:
        self._post(f"{rank}/initialize", {"num_pages": num_pages})

    def reserve_and_get_indices(self, rank: int, keys: List[str]) -> List[Tuple[bool, int]]:
        response = self._post(f"{rank}/reserve_and_get_indices", {"keys": keys})
        return [tuple(item) for item in response.get("indices")]

    def confirm_write(self, rank: int, written_keys: List[Tuple[str, int]]) -> None:
        self._post(f"{rank}/confirm_write", {"written_keys": written_keys})

    def release_pages(self, rank: int, page_indices: List[int]) -> None:
        self._post(f"{rank}/release_pages", {"page_indices": page_indices})

    def delete_keys(self, rank: int, keys: List[str]) -> None:
        self._post(f"{rank}/delete_keys", {"keys": keys})

    def exists(self, rank: int, key: str) -> bool:
        response = self._get(f"{rank}/exists/{key}")
        return response.get("exists", False)

    def clear(self, rank: int) -> None:
        self._post(f"{rank}/clear", {})
