"""Lightweight memory monitoring for diagnosing CPU RAM OOM.

Reads PSS (Proportional Set Size) from /proc — the metric SLURM cgroups track.
Call log_memory() periodically from training loops.
"""

import os
import psutil


def get_pss_mb(pid=None):
    """Get PSS from /proc/[pid]/smaps_rollup (Linux only)."""
    if pid is None:
        pid = os.getpid()
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except (FileNotFoundError, PermissionError):
        return None


def get_rss_mb(pid=None):
    if pid is None:
        pid = os.getpid()
    return psutil.Process(pid).memory_info().rss / 1024**2


def get_memory_snapshot():
    """Return dict with parent + children memory stats."""
    proc = psutil.Process()
    parent_rss = proc.memory_info().rss / 1024**2
    parent_pss = get_pss_mb() or parent_rss

    children = proc.children(recursive=True)
    child_pss_total = 0
    n_children = 0
    for c in children:
        try:
            c_pss = get_pss_mb(c.pid)
            child_pss_total += (c_pss or c.memory_info().rss / 1024**2)
            n_children += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    total_pss = parent_pss + child_pss_total
    return {
        'parent_rss_mb': parent_rss,
        'parent_pss_mb': parent_pss,
        'child_pss_total_mb': child_pss_total,
        'n_children': n_children,
        'total_pss_mb': total_pss,
    }


_SLURM_LIMIT_MB = 40960  # 40G default, override via log_memory(limit_mb=...)


def log_memory(label="", step=0, limit_mb=_SLURM_LIMIT_MB, logger=None):
    """Print memory snapshot and optionally log to wandb via logger."""
    snap = get_memory_snapshot()
    headroom = limit_mb - snap['total_pss_mb']
    print(
        f"[MEM {label} step={step}] "
        f"parent_RSS={snap['parent_rss_mb']:.0f}MB "
        f"parent_PSS={snap['parent_pss_mb']:.0f}MB | "
        f"children_PSS={snap['child_pss_total_mb']:.0f}MB ({snap['n_children']} procs) | "
        f"total_PSS={snap['total_pss_mb']:.0f}MB | "
        f"headroom={headroom:.0f}MB",
        flush=True,
    )
    if logger is not None:
        try:
            logger.log({
                'parent_rss_mb': snap['parent_rss_mb'],
                'parent_pss_mb': snap['parent_pss_mb'],
                'child_pss_total_mb': snap['child_pss_total_mb'],
                'total_pss_mb': snap['total_pss_mb'],
                'headroom_mb': headroom,
            }, "memory", step=step)
        except Exception:
            pass
    return snap
