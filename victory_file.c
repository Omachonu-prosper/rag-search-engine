#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecureBearSSL.h>
#include <Adafruit_PN532.h>
#include <LiquidCrystal_I2C.h>

// === CONFIG ===
const char* ssid     = "Mularboss";
const char* password = "mularboss";         

const String apiUrl    = "https://school.ufuon.com/api/v1/attendance/gate/scan";
const String schoolKey = "Ufuom-2030";             // Shared secret for all devices

String deviceId;  // Will be auto-set to unique MAC address

// PN532 (SPI mode - CS/SS pin)
#define PN532_SS 2
Adafruit_PN532 nfc(PN532_SS);

// LCD (I2C - common address 0x27; change if yours is different, e.g. 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  Serial.begin(115200);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Booting...");

  // Connect to WiFi
  WiFi.begin(ssid, password);
  int attempts = 0;
  lcd.clear();
  lcd.print("WiFi Connecting");

  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    lcd.setCursor(attempts % 16, 1);
    lcd.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    // Get unique MAC address as deviceId
    String mac = WiFi.macAddress();                    // e.g. "5C:CF:7F:12:34:56"
    deviceId = mac;                                    // With colons
    // Optional cleaner version without colons: deviceId.replace(":", "");

    // Optional: prefix + last part for readability
    // deviceId = "gate-" + mac.substring(mac.length() - 8);

    Serial.println("Unique Device ID: " + deviceId);

    lcd.clear();
    lcd.print("Device ID:");
    lcd.setCursor(0, 1);
    lcd.print(deviceId.substring(deviceId.length() - 8));  // Show last 8 chars
    delay(2500);
  } else {
    lcd.clear();
    lcd.print("WiFi Failed!");
    Serial.println("WiFi connection failed!");
    while (true) delay(1000);
  }

  // Initialize PN532
  nfc.begin();
  uint32_t versiondata = nfc.getFirmwareVersion();
  if (!versiondata) {
    lcd.clear();
    lcd.print("PN532 Not Found");
    Serial.println("PN532 not detected!");
    while (true) delay(1000);
  }
  nfc.SAMConfig();  // Configure for reading cards

  lcd.clear();
  lcd.print("Ready - Scan Tag");
  Serial.println("Device ready. Waiting for RFID tag...");
}

void loop() {
  uint8_t uid[7] = {0};
  uint8_t uidLength = 0;

  // Poll for a card (timeout 300ms)
  bool cardDetected = nfc.readPassiveTargetID(PN532_MIFARE_ISO14443A, uid, &uidLength, 300);

  if (cardDetected && uidLength > 0) {
    // Format UID exactly like Postman: "0xEE 0xE2 0xD5 0x5"
    String rfidTagId = "";
    for (uint8_t i = 0; i < uidLength; i++) {
      if (i > 0) rfidTagId += " ";
      rfidTagId += "0x";
      if (uid[i] < 0x10) rfidTagId += "0";
      rfidTagId += String(uid[i], HEX);
    }
    rfidTagId.toUpperCase();

    Serial.println("Detected RFID: " + rfidTagId);

    bool success = sendScan(rfidTagId);

    lcd.clear();
    if (success) {
      lcd.print("Attendance Sent!");
      lcd.setCursor(0, 1);
      lcd.print(rfidTagId.substring(0, 11));  // Show part of tag for feedback
      // Optional buzzer: tone(14, 1200, 300);  // GPIO14 = D5
    } else {
      lcd.print("Send Failed");
      lcd.setCursor(0, 1);
      lcd.print("Check Network");
    }
    delay(3000);
    lcd.clear();
    lcd.print("Ready - Scan Tag");
  }
}

bool sendScan(String tagId) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost - reconnecting...");
    WiFi.reconnect();
    delay(3000);
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("Reconnect failed");
      return false;
    }
  }

  std::unique_ptr<BearSSL::WiFiClientSecure> client(new BearSSL::WiFiClientSecure);
  client->setInsecure();  // For dev/testing (remove in final prod with proper cert)

  HTTPClient http;
  if (!http.begin(*client, apiUrl)) {
    Serial.println("HTTP begin failed");
    return false;
  }

  // Required headers
  http.addHeader("x-device-id", deviceId);
  http.addHeader("x-school-key", schoolKey);
  http.addHeader("Content-Type", "application/json");

  String jsonBody = "{\"rfidTagId\":\"" + tagId + "\"}";

  Serial.println("POSTing to: " + apiUrl);
  Serial.println("Body: " + jsonBody);
  Serial.println("Device ID: " + deviceId);

  int httpCode = http.POST(jsonBody);

  String response = (httpCode > 0) ? http.getString() : "No response";
  Serial.printf("Response code: %d | Body: %s\n", httpCode, response.c_str());

  http.end();

  return (httpCode == 200 || httpCode == 201);
}