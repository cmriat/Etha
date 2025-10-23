import lmdb
from contextlib import contextmanager

QUEUE_ROOT = 'prototyping/weight_transfer/dbs'
STORAGE_ROOT = 'prototyping/weight_transfer/dbs/storage.lmdb'

def queue_path(rank: int, direction: str = 'send') -> str:
    return f"{QUEUE_ROOT}/{rank}_{direction}.lmdb"

@contextmanager
def open_storage_env():
    """Open LMDB environment for storing tensor payloads."""
    env = lmdb.open(
        STORAGE_ROOT,
        map_size=1 << 26,  # 64MB
        subdir=False,
        lock=True,
    )
    try:
        yield env
    finally:
        env.close()


def store_tensor_payload(tensor_id: str, payload: bytes):
    """Store pickled tensor payload in LMDB.

    Args:
        tensor_id: Unique tensor identifier
        payload: Pickled tensor bytes (from ForkingPickler.dumps)
    """
    with open_storage_env() as env:
        with env.begin(write=True) as txn:
            txn.put(tensor_id.encode(), payload)


def load_tensor_payload(tensor_id: str) -> bytes | None:
    """Load pickled tensor payload from LMDB.

    Args:
        tensor_id: Unique tensor identifier

    Returns:
        Pickled tensor bytes, or None if not found
    """
    with open_storage_env() as env:
        with env.begin() as txn:
            return txn.get(tensor_id.encode())
