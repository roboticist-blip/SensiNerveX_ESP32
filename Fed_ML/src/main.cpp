// main.cpp
// XIAO ESP32-S3 — On-Device Vibration Classifier (TinyML)
//
// Pipeline (mirroring the paper's methodology):
//   1. Sample MPU-6050 at 100 Hz
//   2. Accumulate 100 samples → 500-element feature vector
//      (pitch×100, roll×100, gx×100, gy×100, gz×100)
//   3. One forward pass through the NN
//   4. One backpropagation step (online learning)
//   5. Log MSE loss to Serial
//   6. Repeat
//
// Label assignment (hardcoded for now):
//   CLASS 0 — idle / stationary
//   CLASS 1 — low-frequency vibration
//   CLASS 2 — high-frequency vibration
//
// Change CURRENT_LABEL at compile-time or via Serial command
// (see handleSerialCommand()) to collect different classes.

#include <Arduino.h>
#include "MPU6050_driver.h"
#include "FeatureExtractor.h"
#include "NeuralNetwork.h"

#define I2C_SDA_PIN 5
#define I2C_SCL_PIN 6

#define SAMPLE_INTERVAL_US 10000UL   
#define LEARNING_RATE   0.01f
#define CURRENT_LABEL   0 // 0, 1, or 2 — change to collect other classes
#define DEBUG_ANGLES    1
#define DEBUG_LOSS      1

static MPU6050Driver  imu;
static FeatureExtractor fex;
static NeuralNetwork  nn;
static float featureVec[FEATURE_VECTOR_SIZE];
static float nnOutput[NN_OUTPUT_SIZE];
static float target[NN_OUTPUT_SIZE];
static uint32_t lastSampleUs = 0;
static uint32_t trainStep = 0;

static void buildTarget(uint8_t label)
{
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) target[i] = 0.0f;
    if (label < NN_OUTPUT_SIZE) target[label] = 1.0f;
}

// ────────────────────────────────────────────────────────────
// handleSerialCommand() — runtime label switching
// Send '0', '1', or '2' over Serial to switch the active label.
// ────────────────────────────────────────────────────────────
static uint8_t activeLabel = CURRENT_LABEL;

static void handleSerialCommand()
{
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c >= '0' && c <= '2') {
            activeLabel = (uint8_t)(c - '0');
            Serial.printf("[CMD] Label set to %u\n", activeLabel);
        } else if (c == 'w' || c == 'W') {
            // Dump serialized weights to Serial (hex-encoded words)
            size_t count = nn.serializeToSerial();
            Serial.printf("[CMD] Weights dumped: %u floats\n", (unsigned)count);
        }
    }
}

static void printRamInfo()
{
    Serial.printf("[SYS] Free heap : %6lu B\n",
                  (unsigned long)ESP.getFreeHeap());
    Serial.printf("[SYS] Min heap  : %6lu B\n",
                  (unsigned long)ESP.getMinFreeHeap());
    Serial.printf("[SYS] Static alloc — W1:%u B  featureVec:%u B\n",
                  (unsigned)sizeof(float) * NN_W1_SIZE,
                  (unsigned)sizeof(featureVec));
}

void setup()
{
    Serial.begin(115200);
    delay(500);   // allow USB CDC to enumerate

    Serial.println("==============================================");
    Serial.println(" XIAO ESP32-S3 — TinyML Vibration Classifier");
    Serial.println("==============================================");
    printRamInfo();

    if (!imu.begin(I2C_SDA_PIN, I2C_SCL_PIN)) {
        Serial.println("[FATAL] MPU-6050 not found — halting.");
        while (true) { delay(1000); }
    }
    nn.init();
    buildTarget(activeLabel);
    Serial.printf("[MAIN] Training label: %u  LR: %.4f\n",
                  activeLabel, LEARNING_RATE);
    Serial.println("[MAIN] Sampling started — send '0'/'1'/'2' to switch label");
    lastSampleUs = micros();
}

void loop()
{
    handleSerialCommand();
    uint32_t now = micros();
    if ((uint32_t)(now - lastSampleUs) < SAMPLE_INTERVAL_US) return;
    lastSampleUs = now;
    CalibratedIMU cal;
    Angles        angles;

    if (!imu.sample(cal, angles)) {
        Serial.println("[WARN] IMU read failed — skipping sample");
        return;
    }

#if DEBUG_ANGLES
    Serial.printf("[ANG] pitch=%7.3f  roll=%7.3f  "
                  "gx=%7.3f  gy=%7.3f  gz=%7.3f\n",
                  angles.pitch, angles.roll,
                  cal.gx_dps, cal.gy_dps, cal.gz_dps);
#endif

    bool windowFull = fex.pushSample(cal, angles);
    if (!windowFull) return;  
    fex.getFeatureVector(featureVec);
    fex.reset();   
    nn.forward(featureVec, nnOutput);
    buildTarget(activeLabel);
    float loss = nn.backward(target, LEARNING_RATE);
    nn.updateW1(featureVec, LEARNING_RATE);

    trainStep++;

#if DEBUG_LOSS
    Serial.printf("[TRAIN] step=%4lu  label=%u  "
                  "out=[%.4f %.4f %.4f]  loss=%.6f\n",
                  (unsigned long)trainStep, activeLabel,
                  nnOutput[0], nnOutput[1], nnOutput[2], loss);
#endif

    if (trainStep % 50 == 0) {
        nn.debugWeights();
        printRamInfo();
    }
}
