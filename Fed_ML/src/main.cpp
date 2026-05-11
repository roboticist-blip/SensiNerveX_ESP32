// main.cpp
// XIAO ESP32-S3 — On-Device Vibration Classifier (TinyML)
//
// Pipeline (mirroring the paper's methodology):
//   1. Sample MPU-6050 at 100 Hz
//   2. Accumulate 100 samples → 500-element feature vector
//      (pitch×100, roll×100, gx×100, gy×100, gz×100)
//   3. One forward pass through the NN
//   4. [TRAIN mode] One backpropagation step (online learning) + log MSE loss
//      [INFER mode] argmax over outputs → predicted class + confidence log
//   5. Repeat
//
// Label assignment:
//   CLASS 0 — idle / stationary
//   CLASS 1 — low-frequency vibration
//   CLASS 2 — high-frequency vibration
//
// Serial commands (115200 baud):
//   '0'/'1'/'2'  — set active training label
//   'w'/'W'      — dump weights as IEEE-754 hex stream
//   't'/'T'      — switch to TRAINING mode
//   'i'/'I'      — switch to INFERENCE mode
//   's'/'S'      — save weights to LittleFS  (/snx_model.bin)
//   'l'/'L'      — load weights from LittleFS (/snx_model.bin)
//
// Recommended workflow:
//   1. Boot with no saved model → auto-starts in TRAINING mode
//   2. Train each class (≥50 steps each), switch labels with '0'/'1'/'2'
//   3. Send 's' to persist weights to flash
//   4. Send 'i' to enter INFERENCE mode
//   5. On next boot, weights are loaded automatically → boots into INFER

#include <Arduino.h>
#include "MPU6050_driver.h"
#include "FeatureExtractor.h"
#include "NeuralNetwork.h"

// Hardware & tuning constants 
#define I2C_SDA_PIN        5
#define I2C_SCL_PIN        6
#define SAMPLE_INTERVAL_US 10000UL   // 100 Hz
#define LEARNING_RATE      0.01f
#define CURRENT_LABEL      0         // 0 / 1 / 2 — compile-time default
#define DEBUG_ANGLES       1         // 1 = print angles every sample
#define DEBUG_LOSS         1         // 1 = print loss / prediction every window

// Global objects — all static RAM, no heap 
static MPU6050Driver    imu;
static FeatureExtractor fex;
static NeuralNetwork    nn;
static float featureVec[FEATURE_VECTOR_SIZE];
static float nnOutput[NN_OUTPUT_SIZE];
static float target[NN_OUTPUT_SIZE];

static uint32_t lastSampleUs = 0;
static uint32_t trainStep    = 0;
static uint32_t inferStep    = 0;
static uint8_t  activeLabel  = CURRENT_LABEL;

// Operating mode
enum OpMode { MODE_TRAIN, MODE_INFER };
static OpMode opMode = MODE_TRAIN;   // overridden in setup() if model loads

// Human-readable class names — shared by both modes
static const char *CLASS_NAMES[NN_OUTPUT_SIZE] = {
    "IDLE",
    "LOW_VIB",
    "HIGH_VIB"
};

// Helpers
static void buildTarget(uint8_t label)
{
    for (int i = 0; i < NN_OUTPUT_SIZE; i++) target[i] = 0.0f;
    if (label < NN_OUTPUT_SIZE) target[label] = 1.0f;
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

// handleSerialCommand() 
// Processes one character at a time from the serial buffer.
// Non-blocking — called every loop() iteration.
static void handleSerialCommand()
{
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c >= '0' && c <= '2') {
            // Label switch — only meaningful in TRAINING mode
            activeLabel = (uint8_t)(c - '0');
            Serial.printf("[CMD] Label set to %u (%s)\n",
                          activeLabel, CLASS_NAMES[activeLabel]);

        } else if (c == 'w' || c == 'W') {
            // Hex weight dump
            size_t count = nn.serializeToSerial();
            Serial.printf("[CMD] Weights dumped: %u floats\n", (unsigned)count);

        } else if (c == 'i' || c == 'I') {
            // Enter inference mode — no weight updates from here
            opMode    = MODE_INFER;
            inferStep = 0;   // reset per-session counter
            Serial.println("[CMD] Switched to INFERENCE mode");
            Serial.println("[CMD] Send 't'/'T' to return to training");

        } else if (c == 't' || c == 'T') {
            // Return to training mode
            opMode = MODE_TRAIN;
            Serial.println("[CMD] Switched to TRAINING mode");
            Serial.printf("[CMD] Active label: %u (%s)  LR: %.4f\n",
                          activeLabel, CLASS_NAMES[activeLabel], LEARNING_RATE);

        } else if (c == 's' || c == 'S') {
            // Persist current weights to LittleFS
            if (nn.saveToLittleFS()) {
                Serial.println("[CMD] Model saved — will auto-load on next boot");
            } else {
                Serial.println("[CMD] Save FAILED — check LittleFS mount");
            }

        } else if (c == 'l' || c == 'L') {
            // Load weights from LittleFS into live model
            if (nn.loadFromLittleFS()) {
                opMode    = MODE_INFER;
                inferStep = 0;
                Serial.println("[CMD] Model loaded — switched to INFERENCE mode");
            } else {
                Serial.println("[CMD] Load FAILED — staying in current mode");
            }
        }
    }
}

// runInference() 
// Forward pass only — weights are NOT modified.
// argmax() reads _a2[] populated by forward(), no second forward call.
static void runInference()
{
    nn.forward(featureVec, nnOutput);

    // argmax() reads _a2[] set by the forward() call above — no re-run
    uint8_t pred = nn.argmax();

    inferStep++;

#if DEBUG_LOSS
    Serial.printf("[INFER] step=%4lu  pred=%-8s (%u)  "
                  "out=[%.4f  %.4f  %.4f]\n",
                  (unsigned long)inferStep,
                  CLASS_NAMES[pred], pred,
                  nnOutput[0], nnOutput[1], nnOutput[2]);
#endif

    if (inferStep % 50 == 0) {
        printRamInfo();
    }
}

// runTraining() 
// Full online learning step: forward() → backward() → updateW1()
//
// Two-call W1 update pattern (mandatory):
//   backward() stashes delta1 into _z1.
//   updateW1() reads it from _z1 using the still-valid featureVec.
//   Do NOT modify featureVec between these two calls.
static void runTraining()
{
    nn.forward(featureVec, nnOutput);
    buildTarget(activeLabel);

    float loss = nn.backward(target, LEARNING_RATE);
    nn.updateW1(featureVec, LEARNING_RATE);  // MUST follow backward() immediately

    trainStep++;

#if DEBUG_LOSS
    Serial.printf("[TRAIN] step=%4lu  label=%u  "
                  "out=[%.4f  %.4f  %.4f]  loss=%.6f\n",
                  (unsigned long)trainStep, activeLabel,
                  nnOutput[0], nnOutput[1], nnOutput[2], loss);
#endif

    if (trainStep % 50 == 0) {
        nn.debugWeights();
        printRamInfo();
    }
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

    // Attempt to restore a previously saved model from flash.
    // Success  → boot into INFERENCE mode (model is ready).
    // Failure  → fresh Xavier init, stay in TRAINING mode.
    if (nn.loadFromLittleFS()) {
        opMode = MODE_INFER;
        Serial.println("[MAIN] Saved model loaded — starting in INFERENCE mode");
    } else {
        nn.init();
        opMode = MODE_TRAIN;
        Serial.println("[MAIN] No saved model — starting in TRAINING mode");
    }

    buildTarget(activeLabel);

    Serial.printf("[MAIN] Mode: %s  Label: %u (%s)  LR: %.4f\n",
                  opMode == MODE_TRAIN ? "TRAINING" : "INFERENCE",
                  activeLabel, CLASS_NAMES[activeLabel], LEARNING_RATE);
    Serial.println("[MAIN] Commands: '0'/'1'/'2'=label  'i'=infer  't'=train"
                   "  'w'=dump  's'=save  'l'=load");

    lastSampleUs = micros();
}

void loop()
{
    handleSerialCommand();

    // Enforce 100 Hz sample rate via microsecond timer
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

    // Accumulate sample into the 100-slot ring buffer
    bool windowFull = fex.pushSample(cal, angles);
    if (!windowFull) return;

    // Extract feature vector and immediately reset for the next window.
    // featureVec must stay unmodified until after updateW1() in runTraining().
    fex.getFeatureVector(featureVec);
    fex.reset();

    // Dispatch to the active operating mode
    if (opMode == MODE_INFER) {
        runInference();
    } else {
        runTraining();
    }
}
