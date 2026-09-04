"""Background jobs with progress reporting and cancellation.

Long operations (tracking, export) run in daemon threads so the web server
stays responsive. The UI polls :meth:`Job.to_dict` and can call
:meth:`JobManager.cancel`; the worker checks :meth:`Job.check_cancel`
between frames.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

JobState = str  # "queued" | "running" | "done" | "error" | "cancelled"


class JobCancelledError(Exception):
    """Raised inside a worker when the user cancelled the job."""


@dataclass
class Job:
    """State of one background operation."""

    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    state: JobState = "queued"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_progress(self, fraction: float, message: str | None = None) -> None:
        """Update progress (``-1`` = unknown) and optionally the message."""
        with self._lock:
            self.progress = float(fraction)
            if message is not None:
                self.message = message

    def check_cancel(self) -> None:
        """Raise :class:`JobCancelledError` if cancellation was requested."""
        if self.cancel_event.is_set():
            raise JobCancelledError()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
        return self.cancel_event.is_set()

    def to_dict(self) -> dict:
        """JSON view for the API."""
        with self._lock:
            elapsed = (self.finished or time.time()) - (self.started or self.created)
            return {
                "id": self.id,
                "name": self.name,
                "state": self.state,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "elapsed_s": round(elapsed, 1),
                "result": self.result
                if isinstance(self.result, (dict, list, str, int, float))
                else None,
            }


class JobManager:
    """Start, look up and cancel jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, name: str, fn: Callable[[Job], Any]) -> Job:
        """Run ``fn(job)`` in a daemon thread; its return value becomes ``job.result``."""
        job = Job(name=name)
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            job.started = time.time()
            job.state = "running"
            try:
                result = fn(job)
                if job.cancelled:
                    job.state = "cancelled"
                    job.message = "Cancelled"
                else:
                    job.result = result
                    job.state = "done"
                    job.progress = 1.0
            except JobCancelledError:
                job.state = "cancelled"
                job.message = "Cancelled"
            except Exception as exc:  # report to the UI instead of dying silently
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = job.error
                traceback.print_exc()
            finally:
                job.finished = time.time()

        threading.Thread(target=_run, name=f"job-{name}-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        """Look a job up by id."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation; returns ``False`` for unknown or finished jobs."""
        job = self.get(job_id)
        if job is None or job.state in ("done", "error", "cancelled"):
            return False
        job.cancel_event.set()
        job.message = "Cancelling"
        return True

    def all(self) -> list[Job]:
        """All jobs, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def wait(self, job_id: str, timeout: float = 600.0, poll: float = 0.05) -> Job:
        """Block until the job finishes (tests and CLI)."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        deadline = time.time() + timeout
        while job.state in ("queued", "running"):
            if time.time() > deadline:
                raise TimeoutError(f"job {job_id} still {job.state} after {timeout}s")
            time.sleep(poll)
        return job
