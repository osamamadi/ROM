"""Child-process worker: strip a training checkpoint down to {state_dict, meta}.

    python slim_checkpoint.py <full.pth> <slim.pth>

Runs in its own process so that ALL memory used for the (large) full checkpoint goes back to the OS when it
exits; the API process never holds it. Exit code 0 = slim file written AND verified bit-identical.
"""
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s:rom.slim:%(message)s")
log = logging.getLogger("rom.slim")


def _load(path: str):
    """mmap keeps tensor storages file-backed (paged in lazily), so the optimizer state is never read."""
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except Exception as e:  # older torch / legacy file format
        log.warning("mmap load unavailable (%s); loading normally", e)
        return torch.load(path, map_location="cpu")


def main(full: str, dest: str) -> int:
    ck = _load(full)
    log.info("Full checkpoint keys: %s", sorted(ck.keys()))
    slim = {"state_dict": ck["state_dict"], "meta": ck.get("meta", {})}

    tmp = dest + ".tmp"
    torch.save(slim, tmp)

    back = _load(tmp)
    same = back["state_dict"].keys() == slim["state_dict"].keys() and all(
        torch.equal(back["state_dict"][k], v) for k, v in slim["state_dict"].items()
    )
    if not same:
        os.remove(tmp)
        log.error("Slim checkpoint does not match the original state_dict.")
        return 1

    log.info("Slim state_dict verified bit-identical (%d tensors); dataset_meta kept: %s",
             len(slim["state_dict"]), "dataset_meta" in back.get("meta", {}))
    del back, slim, ck
    os.replace(tmp, dest)  # atomic: a crash never leaves a truncated *_slim.pth
    log.info("Slim checkpoint: %.0f MB -> %.0f MB", os.path.getsize(full) / 2**20, os.path.getsize(dest) / 2**20)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
