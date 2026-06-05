"""Industrial PLC bridge over MQTT.

Two roles, both backed by paho-mqtt:

  * ``InspectionPublisher`` — the vision controller. Publishes every verdict to
    ``factory/inspection/results`` and, when a unit fails *and* PLC output is
    enabled, issues a reject command on ``factory/plc/reject``.

  * ``VirtualPLC`` — a simulated programmable logic controller that subscribes
    to the reject topic and "actuates" a pneumatic pusher (tracked in state and
    exposed to the dashboard).

If the broker is unreachable the publisher degrades to a no-op so the rest of
the pipeline keeps running — exactly what you want on a noisy plant network.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Optional

import paho.mqtt.client as mqtt

from .config import settings


class InspectionPublisher:
    def __init__(self) -> None:
        self._client = mqtt.Client(client_id="vision-controller", clean_session=True)
        self.connected = False
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            self._client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=30)
            self._client.loop_start()
            self.connected = True
            print(f"[mqtt] publisher connected to {settings.mqtt_host}:{settings.mqtt_port}")
        except Exception as exc:  # pragma: no cover - network dependent
            self.connected = False
            print(f"[mqtt] publisher offline ({exc}); running in degraded mode")

    def publish_result(self, payload: dict) -> None:
        if not self.connected:
            return
        with self._lock:
            self._client.publish(settings.mqtt_topic_results, json.dumps(payload, default=str), qos=0)

    def publish_reject(self, unit_id: str, defect_types: list[str]) -> None:
        if not self.connected:
            return
        cmd = {
            "command": "REJECT",
            "unit_id": unit_id,
            "defects": defect_types,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with self._lock:
            self._client.publish(settings.mqtt_topic_reject, json.dumps(cmd), qos=1)

    def disconnect(self) -> None:
        if self.connected:
            self._client.loop_stop()
            self._client.disconnect()
            self.connected = False


class VirtualPLC:
    """Simulated reject actuator listening on the reject topic."""

    def __init__(self) -> None:
        self._client = mqtt.Client(client_id="virtual-plc", clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self.online = False
        # Actuator state surfaced to the dashboard.
        self.last_command: Optional[str] = None
        self.last_reject_unit: Optional[str] = None
        self.last_reject_at: Optional[datetime] = None
        self.total_rejects = 0

    def connect(self) -> None:
        try:
            self._client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=30)
            self._client.loop_start()
            self.online = True
            print("[plc] virtual PLC subscribed to reject topic")
        except Exception as exc:  # pragma: no cover
            self.online = False
            print(f"[plc] virtual PLC offline ({exc})")

    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe(settings.mqtt_topic_reject, qos=1)

    def _on_message(self, client, userdata, msg):
        try:
            cmd = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if cmd.get("command") == "REJECT":
            self.last_command = "REJECT"
            self.last_reject_unit = cmd.get("unit_id")
            self.last_reject_at = datetime.utcnow()
            self.total_rejects += 1
            # In a real plant this is where the digital output toggles.
            print(f"[plc] >>> ACTUATE pusher — rejecting {self.last_reject_unit} "
                  f"(defects: {cmd.get('defects')})")

    def status(self) -> dict:
        return {
            "online": self.online,
            "last_command": self.last_command,
            "last_reject_unit": self.last_reject_unit,
            "last_reject_at": self.last_reject_at,
            "total_rejects": self.total_rejects,
        }

    def disconnect(self) -> None:
        if self.online:
            self._client.loop_stop()
            self._client.disconnect()
            self.online = False
