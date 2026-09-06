#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>

// ==========================================
// CONFIGURACIÓN WI-FI ACCESS POINT (AP)
// ==========================================
const char *ssid = "Plataforma_Diagnostico";
const char *password = "admin_plataforma";  // Mínimo 8 caracteres
WiFiServer server(8080);                    // Servidor TCP en el puerto 8080

// ==========================================
// CONFIGURACIÓN HX711 (Celdas de Carga)
// ==========================================
const int SCK_PIN = 4;
// Índices 0 al 4: LADO IZQUIERDO | Índices 5 al 9: LADO DERECHO
// FPI(0), SPI(1), AI(2), SFI(3), FFI(4), FPD(5), SPD(6), AD(7), SFD(8), FFD(9)
const int DT_PINS[10] = { 13, 34, 32, 25, 27, 14, 35, 33, 26, 19 };
long lecturasFuerza[10] = { 0 };

// ==========================================
// CONFIGURACIÓN ADXL345 (Acelerómetros)
// ==========================================
const uint8_t ADDR_A = 0x53;
const uint8_t ADDR_B = 0x1D;

// Variables para control de 80 Hz (12500 microsegundos por ciclo)
const unsigned long intervaloMuestreo = 12500;
unsigned long tiempoAnterior = 0;

void activarSensor(TwoWire &bus, uint8_t direccion) {
  bus.beginTransmission(direccion);
  bus.write(0x2D);  // Registro POWER_CTL
  bus.write(0x08);  // Activar modo de medición
  bus.endTransmission();

  // Rango +-4G (Registro DATA_FORMAT 0x31) -> Valor 0x01
  bus.beginTransmission(direccion);
  bus.write(0x31);
  bus.write(0x01);
  bus.endTransmission();
}

void setup() {
  Serial.begin(115200);

  // 1. Inicializar Pines HX711
  pinMode(SCK_PIN, OUTPUT);
  digitalWrite(SCK_PIN, LOW);
  for (int i = 0; i < 10; i++) {
    pinMode(DT_PINS[i], INPUT);
  }

  // 2. Inicializar Buses I2C (ADXL345)
  Wire.begin(17, 16);   // Bus 0: Eje Posterior
  Wire1.begin(21, 22);  // Bus 1: Eje Frontal
  delay(100);

  activarSensor(Wire, ADDR_A);   // API
  activarSensor(Wire, ADDR_B);   // APD
  activarSensor(Wire1, ADDR_A);  // AFD
  activarSensor(Wire1, ADDR_B);  // AFI

  // 3. Crear Access Point Wi-Fi
  Serial.println("Configurando Access Point Wi-Fi...");
  WiFi.softAP(ssid, password);
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  // 4. Iniciar Servidor
  server.begin();
  Serial.println("Servidor iniciado. Esperando conexión de Python...");
}

bool leerCeldasCarga() {
  // Verificar si todos los HX711 están listos
  for (int i = 0; i < 10; i++) {
    if (digitalRead(DT_PINS[i]) == HIGH) return false;
  }

  for (int i = 0; i < 10; i++) lecturasFuerza[i] = 0;

  noInterrupts();  // Desactivar interrupciones para evitar desincronización
  for (int i = 0; i < 24; i++) {
    digitalWrite(SCK_PIN, HIGH);
    delayMicroseconds(1);
    for (int j = 0; j < 10; j++) {
      lecturasFuerza[j] = (lecturasFuerza[j] << 1) | digitalRead(DT_PINS[j]);
    }
    digitalWrite(SCK_PIN, LOW);
    delayMicroseconds(1);
  }

  // Pulso 25
  digitalWrite(SCK_PIN, HIGH);
  delayMicroseconds(1);
  digitalWrite(SCK_PIN, LOW);
  interrupts();

  // Complemento a 2 e inversión espacial (FPI[0], AI[2], FFI[4])
  for (int i = 0; i < 10; i++) {
    if (lecturasFuerza[i] & 0x800000) lecturasFuerza[i] |= 0xFF000000;
    if (i == 0 || i == 2 || i == 4) lecturasFuerza[i] *= -1;
  }
  return true;
}

void obtenerAceleraciones(TwoWire &bus, uint8_t direccion, int multiplicador, String &trama) {
  bus.beginTransmission(direccion);
  bus.write(0x32);
  bus.endTransmission(false);
  bus.requestFrom((uint16_t)direccion, (uint8_t)6, (uint8_t) true);

  if (bus.available() >= 6) {
    int16_t x = (bus.read() | (bus.read() << 8)) * multiplicador;
    int16_t y = (bus.read() | (bus.read() << 8)) * multiplicador;
    int16_t z = (bus.read() | (bus.read() << 8));
    trama += String(x) + "," + String(y) + "," + String(z) + ",";
  } else {
    trama += "0,0,0,";  // Fallback por si hay error en I2C
  }
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("Cliente (Python) conectado.");
    String comando = "";

    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        comando += c;

        // Si Python envía 'E' (Encerado) o 'A' (Adquisición de 5 segundos)
        if (c == '\n') {
          comando.trim();

          if (comando == "E" || comando == "A") {
            int muestrasRequeridas = (comando == "E") ? 80 : 400;  // 1 seg para encerar, 5 seg para ensayo
            int muestrasActuales = 0;
            tiempoAnterior = micros();

            while (muestrasActuales < muestrasRequeridas && client.connected()) {
              unsigned long tiempoActual = micros();

              // Bucle determinista a 80 Hz
              if (tiempoActual - tiempoAnterior >= intervaloMuestreo) {
                tiempoAnterior = tiempoActual;

                if (leerCeldasCarga()) {
                  String tramaDatos = "";

                  // 1. Concatenar Fuerzas
                  for (int i = 0; i < 10; i++) {
                    tramaDatos += String(lecturasFuerza[i]) + ",";
                  }

                  // 2. Concatenar Aceleraciones
                  obtenerAceleraciones(Wire, ADDR_A, 1, tramaDatos);    // API
                  obtenerAceleraciones(Wire, ADDR_B, 1, tramaDatos);    // APD
                  obtenerAceleraciones(Wire1, ADDR_A, -1, tramaDatos);  // AFD (Espejo X,Y)
                  obtenerAceleraciones(Wire1, ADDR_B, -1, tramaDatos);  // AFI (Espejo X,Y)

                  // 3. Enviar por Wi-Fi
                  tramaDatos += "\n";
                  client.print(tramaDatos);

                  muestrasActuales++;
                }
              }
            }
            client.println("FIN");  // Indica a Python que terminó el lote de datos
          }
          comando = "";
        }
      }
    }
    Serial.println("Cliente desconectado.");
  }
}