import argparse
import atexit
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

# --- Data Models ---
class RankMetadata:
    """Holds all metadata for a single rank."""
    def __init__(self, num_pages: int):
        self.lock = threading.RLock()
        self.num_pages = num_pages
        self.free_pages: List[int] = list(range(num_pages))
        self.key_to_index: OrderedDict[str, int] = OrderedDict()

class GlobalState:
    """Manages the state for all ranks and persistence."""
    def __init__(self, persistence_path: Optional[str], save_interval: int):
        self.global_lock = threading.RLock()
        self.ranks: Dict[int, RankMetadata] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.save_interval = save_interval
        self.save_timer: Optional[threading.Timer] = None
        self.is_shutting_down = False

    def load_from_disk(self):
        if not self.persistence_path or not self.persistence_path.exists():
            logging.info("Persistence file not found. Starting with a clean state.")
            return

        logging.info(f"Loading state from {self.persistence_path}")
        try:
            with open(self.persistence_path, 'r') as f:
                persisted_data = json.load(f)
            
            with self.global_lock:
                for rank_id_str, data in persisted_data.items():
                    rank_id = int(rank_id_str)
                    num_pages = data['num_pages']
                    rank_meta = RankMetadata(num_pages)
                    rank_meta.free_pages = data['free_pages']
                    # Ensure key_to_index is loaded as OrderedDict
                    rank_meta.key_to_index = OrderedDict(data['key_to_index'])
                    self.ranks[rank_id] = rank_meta
                logging.info(f"Successfully loaded metadata for {len(self.ranks)} ranks.")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Failed to load or parse persistence file: {e}. Starting fresh.", exc_info=True)
            self.ranks.clear()

    def save_to_disk(self):
        if not self.persistence_path:
            return

        logging.info("Persisting metadata to disk...")
        with self.global_lock:
            serializable_state = {}
            for rank_id, rank_meta in self.ranks.items():
                with rank_meta.lock:
                    serializable_state[rank_id] = {
                        'num_pages': rank_meta.num_pages,
                        'free_pages': rank_meta.free_pages,
                        'key_to_index': list(rank_meta.key_to_index.items())
                    }
        
        try:
            temp_path = self.persistence_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(serializable_state, f, indent=4)
            temp_path.rename(self.persistence_path)
            logging.info(f"Metadata successfully persisted to {self.persistence_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata to disk: {e}", exc_info=True)

    def schedule_save(self):
        if self.is_shutting_down or not self.persistence_path:
            return
        self.save_to_disk()
        self.save_timer = threading.Timer(self.save_interval, self.schedule_save)
        self.save_timer.start()

    def shutdown(self):
        logging.info("Shutting down metadata server...")
        self.is_shutting_down = True
        if self.save_timer:
            self.save_timer.cancel()
        self.save_to_disk()
        logging.info("Shutdown complete.")

# --- Pydantic Models for API ---
class InitRequest(BaseModel):
    num_pages: int

class KeysRequest(BaseModel):
    keys: List[str]

class WrittenKeysRequest(BaseModel):
    written_keys: List[tuple[str, int]]

class PageIndicesRequest(BaseModel):
    page_indices: List[int]
    
class RankStatsResponse(BaseModel):
    rank_id: int
    num_pages: int
    free_pages_count: int
    used_pages_count: int
    keys_count: int
    usage_percentage: float
    
class GlobalStatsResponse(BaseModel):
    total_ranks: int
    total_pages: int
    total_keys: int
    ranks: List[RankStatsResponse]
    persistence_enabled: bool
    persistence_path: Optional[str] = None
    save_interval: Optional[int] = None

# --- API Endpoints ---
def get_rank_metadata(rank: int) -> RankMetadata:
    with state.global_lock:
        if rank not in state.ranks:
            raise HTTPException(status_code=404, detail=f"Rank {rank} not initialized. Please call /{{rank}}/initialize first.")
        return state.ranks[rank]

@app.post("/{rank}/initialize")
def initialize(rank: int, request: InitRequest):
    with state.global_lock:
        if rank in state.ranks:
            logging.info(f"Rank {rank} already exists. Initialization request ignored.")
            if state.ranks[rank].num_pages != request.num_pages:
                 logging.warning(f"Rank {rank} initialized with different num_pages. Existing: {state.ranks[rank].num_pages}, New: {request.num_pages}")
        else:
            logging.info(f"Initializing new Rank {rank} with {request.num_pages} pages.")
            state.ranks[rank] = RankMetadata(request.num_pages)
    return {"message": f"Rank {rank} is ready."}

@app.get("/{rank}/exists/{key}")
def exists(rank: int, key: str):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        key_exists = key in metadata.key_to_index
        if key_exists:
            metadata.key_to_index.move_to_end(key)
        return {"exists": key_exists}

@app.post("/{rank}/reserve_and_get_indices")
def reserve_and_get_indices(rank: int, request: KeysRequest):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        keys = request.keys
        results = [None] * len(keys)
        
        new_keys_to_process = []
        for i, key in enumerate(keys):
            if key in metadata.key_to_index:
                results[i] = (True, metadata.key_to_index[key])
                metadata.key_to_index.move_to_end(key)
            else:
                new_keys_to_process.append((i, key))

        for i, key in new_keys_to_process:
            if len(metadata.free_pages) > 0:
                page_idx = metadata.free_pages.pop()
                results[i] = (False, page_idx)
            elif len(metadata.key_to_index) > 0:
                # Get the least recently used key (first item in OrderedDict)
                lru_key, page_idx = next(iter(metadata.key_to_index.items()))
                # Remove the LRU key
                metadata.key_to_index.pop(lru_key)
                logging.info(f"Rank {rank}: Evicting LRU key '{lru_key}' from page {page_idx} for new key '{key}'.")
                results[i] = (False, page_idx)
            else:
                results[i] = (False, -1)
        
        return {"indices": results}

@app.post("/{rank}/confirm_write")
def confirm_write(rank: int, request: WrittenKeysRequest):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        for key, page_index in request.written_keys:
            metadata.key_to_index[key] = page_index
            metadata.key_to_index.move_to_end(key)
    return {"message": f"Rank {rank}: Write confirmed for {len(request.written_keys)} keys."}

@app.post("/{rank}/release_pages")
def release_pages(rank: int, request: PageIndicesRequest):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        for page_index in request.page_indices:
            if page_index not in metadata.free_pages:
                metadata.free_pages.append(page_index)
    return {"message": f"Rank {rank}: {len(request.page_indices)} pages released."}

@app.post("/{rank}/delete_keys")
def delete_keys(rank: int, request: KeysRequest):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        count = 0
        for key in request.keys:
            if key in metadata.key_to_index:
                page_index = metadata.key_to_index.pop(key)
                if page_index not in metadata.free_pages:
                    metadata.free_pages.append(page_index)
                count += 1
    return {"message": f"Rank {rank}: {count} keys deleted."}

@app.post("/{rank}/clear")
def clear(rank: int):
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        metadata.free_pages = list(range(metadata.num_pages))
        metadata.key_to_index.clear()
    return {"message": f"Rank {rank}: Metadata cleared."}

@app.get("/{rank}/stats")
def get_rank_stats(rank: int):
    """Get statistics for a specific rank."""
    metadata = get_rank_metadata(rank)
    with metadata.lock:
        free_pages_count = len(metadata.free_pages)
        used_pages_count = metadata.num_pages - free_pages_count
        keys_count = len(metadata.key_to_index)
        usage_percentage = (used_pages_count / metadata.num_pages) * 100 if metadata.num_pages > 0 else 0
        
        return RankStatsResponse(
            rank_id=rank,
            num_pages=metadata.num_pages,
            free_pages_count=free_pages_count,
            used_pages_count=used_pages_count,
            keys_count=keys_count,
            usage_percentage=usage_percentage
        )

@app.get("/stats")
def get_global_stats():
    """Get global statistics for all ranks."""
    with state.global_lock:
        total_pages = 0
        total_keys = 0
        rank_stats = []
        
        for rank_id, rank_meta in state.ranks.items():
            with rank_meta.lock:
                free_pages_count = len(rank_meta.free_pages)
                used_pages_count = rank_meta.num_pages - free_pages_count
                keys_count = len(rank_meta.key_to_index)
                usage_percentage = (used_pages_count / rank_meta.num_pages) * 100 if rank_meta.num_pages > 0 else 0
                
                total_pages += rank_meta.num_pages
                total_keys += keys_count
                
                rank_stats.append(RankStatsResponse(
                    rank_id=rank_id,
                    num_pages=rank_meta.num_pages,
                    free_pages_count=free_pages_count,
                    used_pages_count=used_pages_count,
                    keys_count=keys_count,
                    usage_percentage=usage_percentage
                ))
        
        return GlobalStatsResponse(
            total_ranks=len(state.ranks),
            total_pages=total_pages,
            total_keys=total_keys,
            ranks=rank_stats,
            persistence_enabled=state.persistence_path is not None,
            persistence_path=str(state.persistence_path) if state.persistence_path else None,
            save_interval=state.save_interval if state.persistence_path else None
        )

@app.post("/force_save", status_code=status.HTTP_200_OK)
def force_save():
    """Force an immediate save of metadata to disk."""
    if not state.persistence_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Persistence is not enabled. Cannot force save."
        )
    
    try:
        state.save_to_disk()
        return {"message": "Metadata successfully saved to disk."}
    except Exception as e:
        logging.error(f"Failed to force save metadata: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save metadata: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "ranks": len(state.ranks)}

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF3FS Metadata Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on."
    )
    parser.add_argument(
        "--persistence-path", type=str, default=None, help="Path to the file for persisting metadata. If not provided, persistence is disabled."
    )
    parser.add_argument(
        "--save-interval", type=int, default=60, help="Interval in seconds for periodically saving metadata to disk."
    )
    args = parser.parse_args()

    state = GlobalState(args.persistence_path, args.save_interval)
    
    state.load_from_disk()
    if state.persistence_path:
        state.schedule_save()
        atexit.register(state.shutdown)

    import uvicorn
    logging.info(f"Starting metadata server on http://{args.host}:{args.port}")
    if state.persistence_path:
        logging.info(f"Persistence is ENABLED. Saving to '{args.persistence_path}' every {args.save_interval} seconds.")
    else:
        logging.info("Persistence is DISABLED.")
    
    uvicorn.run(app, host=args.host, port=args.port)