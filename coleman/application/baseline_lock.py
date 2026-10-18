"""Local attempt serialization for matrix orchestration and its existing native workers."""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path

from coleman.corpus.materialization import validate_source_storage


class BaselineBusyError(RuntimeError):
    """A live local process owns the same matrix or native execution; retry after it finishes."""


@contextmanager
def baseline_lock(path: Path):
    """Hold a nonblocking POSIX advisory lock, released by the kernel on process death.

    Lock files are retained to avoid inode replacement races. The caller must
    use a trusted local filesystem and serialize its own threads; this is not a
    distributed or hostile-filesystem lock. Symlink/nonregular/multiply linked
    lock files reject. Unsupported platforms/filesystems and I/O errors propagate.
    Active contention raises BaselineBusyError without declaring work complete.

    A parent matrix lock alone is insufficient: an existing native child may
    outlive a killed parent. Each native worker separately holds its execution
    lock through result/checkpoint publication, so a new parent cannot race that
    child. Descriptors are not inherited across exec/spawn.
    """
    import fcntl

    validate_source_storage(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Baseline lock requires one regular local file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BaselineBusyError("Baseline execution is already active; retry after its owner finishes") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
