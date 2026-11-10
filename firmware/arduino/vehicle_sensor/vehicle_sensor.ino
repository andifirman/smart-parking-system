// Smart Parking vehicle-presence trigger for Arduino Nano + HC-SR04.
// Wiring: VCC -> 5V, GND -> GND, TRIG -> D9, ECHO -> D10.

const uint8_t TRIG_PIN = 9;
const uint8_t ECHO_PIN = 10;
const uint8_t STATUS_LED_PIN = LED_BUILTIN;

const float DETECT_DISTANCE_CM = 35.0;
const float CLEAR_DISTANCE_CM = 50.0;
const uint8_t REQUIRED_STABLE_READINGS = 3;
const unsigned long SAMPLE_INTERVAL_MS = 100;
const unsigned long TELEMETRY_INTERVAL_MS = 1000;

bool vehiclePresent = false;
uint8_t detectedReadings = 0;
uint8_t clearReadings = 0;
unsigned long lastSampleAt = 0;
unsigned long lastTelemetryAt = 0;

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  const unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0) {
    return -1.0;
  }
  return duration * 0.0343 / 2.0;
}

void publishEvent(const char* eventName, float distanceCm) {
  Serial.print(F("{\"event\":\""));
  Serial.print(eventName);
  Serial.print(F("\",\"distance_cm\":"));
  Serial.print(distanceCm, 1);
  Serial.println(F("}"));
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.begin(115200);
  delay(500);
  publishEvent("sensor_ready", -1.0);
}

void loop() {
  const unsigned long now = millis();
  if (now - lastSampleAt < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleAt = now;

  const float distanceCm = readDistanceCm();
  const bool detected = distanceCm >= 2.0 && distanceCm <= DETECT_DISTANCE_CM;
  const bool cleared = distanceCm < 0.0 || distanceCm >= CLEAR_DISTANCE_CM;

  if (!vehiclePresent) {
    detectedReadings = detected ? detectedReadings + 1 : 0;
    if (detectedReadings >= REQUIRED_STABLE_READINGS) {
      vehiclePresent = true;
      detectedReadings = 0;
      digitalWrite(STATUS_LED_PIN, HIGH);
      publishEvent("vehicle_detected", distanceCm);
    }
  } else {
    clearReadings = cleared ? clearReadings + 1 : 0;
    if (clearReadings >= REQUIRED_STABLE_READINGS) {
      vehiclePresent = false;
      clearReadings = 0;
      digitalWrite(STATUS_LED_PIN, LOW);
      publishEvent("vehicle_cleared", distanceCm);
    }
  }

  if (now - lastTelemetryAt >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryAt = now;
    publishEvent("distance", distanceCm);
  }
}
