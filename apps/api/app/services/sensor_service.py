"""USB-serial bridge for Arduino vehicle-presence events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from threading import Event, Lock, Thread
from time import sleep

import serial


class SensorService:
    """Read newline-delimited Arduino events without blocking API requests."""

    def __init__(self, port: str | None, baudrate: int = 115200) -> None:
        self.port = port.strip() if port else None
        self.baudrate = baudrate
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._serial: serial.Serial | None = None
        self._connected = False
        self._event_id = 0
        self._last_event = "disabled" if not self.port else "waiting"
        self._distance_cm: float | None = None
        self._last_seen_at: str | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.port or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="arduino-sensor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._serial and self._serial.is_open:
            self._serial.close()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                with serial.Serial(self.port, self.baudrate, timeout=1) as connection:
                    self._serial = connection
                    with self._lock:
                        self._connected = True
                        self._last_error = None
                    while not self._stop_event.is_set():
                        line = connection.readline().decode("utf-8", errors="replace").strip()
                        if line:
                            self.handle_line(line)
            except (OSError, serial.SerialException) as exc:
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
                if not self._stop_event.is_set():
                    sleep(2)
            finally:
                self._serial = None
                with self._lock:
                    self._connected = False

    def handle_line(self, line: str) -> bool:
        """Parse one Arduino JSON line and return whether it was accepted."""
        try:
            payload = json.loads(line)
            event_name = str(payload["event"])
            distance = float(payload["distance_cm"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._last_event = event_name
            self._distance_cm = distance if distance >= 0 else None
            self._last_seen_at = now
            if event_name == "vehicle_detected":
                self._event_id += 1
        return True

    def simulate_detection(self, distance_cm: float = 25.0) -> None:
        self.handle_line(json.dumps({"event": "vehicle_detected", "distance_cm": distance_cm}))

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": bool(self.port),
                "connected": self._connected,
                "port": self.port,
                "event_id": self._event_id,
                "last_event": self._last_event,
                "distance_cm": self._distance_cm,
                "last_seen_at": self._last_seen_at,
                "last_error": self._last_error,
            }
