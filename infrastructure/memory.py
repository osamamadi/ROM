"""Tiny RSS logger so memory use is visible in the Railway logs."""
import logging
import os

import psutil

log = logging.getLogger("rom.mem")
_proc = psutil.Process(os.getpid())


def rss_mb() -> float:
    return _proc.memory_info().rss / 2**20


def peak_mb() -> float:
    """Peak RSS of this process so far (Linux: ru_maxrss in KB; elsewhere: fall back to current)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        return rss_mb()


def log_rss(tag: str) -> float:
    cur = rss_mb()
    log.info("RSS %-28s %7.0f MB (peak %.0f MB)", tag, cur, peak_mb())
    return cur
