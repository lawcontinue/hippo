"""Task queue — priority-based task management."""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(order=False)
class Task:
    id: str
    type: str  # chat / embed / image / generic
    payload: Dict[str, Any]
    priority: int = 0  # higher = more urgent
    target_node: Optional[str] = None  # specific node, or None = auto
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    _counter: int = field(default=0, repr=False)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


class TaskQueue:
    def __init__(self):
        self._heap: List[tuple] = []
        self._tasks: Dict[str, Task] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def put(self, task: Task) -> None:
        with self._lock:
            task._counter = self._counter
            self._counter += 1
            self._tasks[task.id] = task
            heapq.heappush(self._heap, (-task.priority, task._counter, task.id))

    def pop(self) -> Optional[Task]:
        with self._lock:
            while self._heap:
                _, _, tid = heapq.heappop(self._heap)
                task = self._tasks.get(tid)
                if task and task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    return task
            return None

    def complete(self, task_id: str, result: Any = None, error: Optional[str] = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED if error else TaskStatus.DONE
                task.result = result
                task.error = error
                task.finished_at = time.time()

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def stats(self) -> Dict[str, int]:
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for t in self._tasks.values():
            counts[t.status.value] += 1
        return counts

    def recent(self, limit: int = 20) -> list:
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]
