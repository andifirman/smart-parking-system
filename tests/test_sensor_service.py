from apps.api.app.services.sensor_service import SensorService


def test_sensor_service_increments_event_for_vehicle_detection() -> None:
    sensor = SensorService(port=None)

    accepted = sensor.handle_line('{"event":"vehicle_detected","distance_cm":24.5}')

    assert accepted
    assert sensor.status()["event_id"] == 1
    assert sensor.status()["distance_cm"] == 24.5


def test_sensor_service_does_not_increment_event_for_telemetry() -> None:
    sensor = SensorService(port=None)

    sensor.handle_line('{"event":"distance","distance_cm":75.0}')

    assert sensor.status()["event_id"] == 0
    assert sensor.status()["last_event"] == "distance"


def test_sensor_service_rejects_malformed_serial_data() -> None:
    sensor = SensorService(port=None)

    assert not sensor.handle_line("not-json")
