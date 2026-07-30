"""Thread-safe teleoperation history buffer.

Stores synchronized-enough records at camera rate (~10 Hz); provides
time-subsampled snapshots for the VLM prompt and motion statistics for the
minimum-motion gate and the visual arrow.
"""
import threading
from collections import deque
from typing import List, Optional

import numpy as np


class HistoryRecord:
    __slots__ = ("stamp", "image", "ee_pos", "gripper_state", "cmd_vel")

    def __init__(self, stamp, image, ee_pos, gripper_state, cmd_vel):
        self.stamp = stamp
        self.image = image
        self.ee_pos = np.asarray(ee_pos, dtype=float)
        self.gripper_state = gripper_state
        self.cmd_vel = np.asarray(cmd_vel, dtype=float)

    def state_line(self, t_rel: float) -> str:
        p = self.ee_pos
        v = self.cmd_vel
        return ("t=%+.1fs | EE(world)=[%.3f, %.3f, %.3f] | gripper=%s | "
                "user_cmd_vel=[%+.3f, %+.3f, %+.3f]"
                % (t_rel, p[0], p[1], p[2], self.gripper_state,
                   v[0], v[1], v[2]))


class HistoryBuffer:
    def __init__(self, max_seconds: float = 12.0, rate_hz: float = 10.0):
        self._lock = threading.Lock()
        self._records = deque(maxlen=int(max_seconds * rate_hz) + 8)

    def append(self, record: HistoryRecord) -> None:
        with self._lock:
            self._records.append(record)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self):
        with self._lock:
            return len(self._records)

    def snapshot(self, n_frames: int, history_sec: float
                 ) -> List[HistoryRecord]:
        """Latest record plus (n_frames - 1) evenly time-spaced earlier
        records covering ~history_sec. Oldest first."""
        with self._lock:
            records = list(self._records)
        if not records:
            return []
        newest = records[-1]
        if n_frames <= 1:
            return [newest]
        targets = [newest.stamp - history_sec * (1 - i / (n_frames - 1))
                   for i in range(n_frames - 1)]
        picked, idx = [], 0
        for t in targets:
            while (idx + 1 < len(records)
                   and abs(records[idx + 1].stamp - t)
                   <= abs(records[idx].stamp - t)):
                idx += 1
            picked.append(records[idx])
        result = []
        for r in picked + [newest]:
            if not result or r is not result[-1]:
                result.append(r)
        return result

    def motion_since(self, window_s: float):
        """(path_length_m, ee_pos_at_window_start, ee_pos_now) over the
        last window_s. Returns (0.0, None, None) when empty."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 0.0, None, None
        cutoff = records[-1].stamp - window_s
        window = [r for r in records if r.stamp >= cutoff] or [records[-1]]
        path = 0.0
        for a, b in zip(window[:-1], window[1:]):
            path += float(np.linalg.norm(b.ee_pos - a.ee_pos))
        return path, window[0].ee_pos, window[-1].ee_pos
