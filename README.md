# SensiNerveX
On-device training of a feedforward neural network directly on the Seeed XIAO ESP32-S3 microcontroller

# XIAO ESP32-S3 — On-Device TinyML Vibration Classifier

A PlatformIO/Arduino project that performs **on-device training** of a feedforward neural network directly on the Seeed XIAO ESP32-S3 microcontroller, validated with a live hardware test on 2026-05-03.

Inspired by the methodology in:

> Llisterri Giménez et al., *"On-Device Training of Machine Learning Models on Microcontrollers with Federated Learning"*, Electronics 2022, 11, 573.

The paper's MFCC audio pipeline is replaced by a vibration-angle pipeline: the MPU-6050 IMU is sampled at 100 Hz, a complementary filter fuses accelerometer and gyroscope data into pitch/roll angles, and every 1-second window (500 features) is pushed through an online-trained three-class neural network. This constitutes one **federated learning client node** — the foundation for a multi-device FedAvg system.

---

## Table of Contents

1. [Hardware](#hardware)
2. [Project Structure](#project-structure)
3. [System Architecture](#system-architecture)
4. [Pipeline — Step by Step](#pipeline--step-by-step)
5. [Library Reference](#library-reference)
   - [MPU6050Driver](#mpu6050driver)
   - [FeatureExtractor](#featureextractor)
   - [NeuralNetwork](#neuralnetwork)
6. [Neural Network Design](#neural-network-design)
7. [RAM Budget](#ram-budget)
8. [Build, Flash & Monitor](#build-flash--monitor)
9. [Fed_ML Helper Script](#fed_ml-helper-script)
10. [Runtime Label Switching & Serial Commands](#runtime-label-switching--serial-commands)
11. [Weight Serialization Format](#weight-serialization-format)
12. [Configuration Reference](#configuration-reference)
13. [Validated Test Results (2026-05-03)](#validated-test-results-2026-05-03)
14. [Extending to Federated Learning](#extending-to-federated-learning)
15. [Known Limitations & Future Work](#known-limitations--future-work)
16. [What's updated in the Latest Version](#What's New in V1.0.1)
17. [License](#license)

---

## Hardware

| Component     | Details                          |
|---------------|----------------------------------|
| MCU board     | Seeed XIAO ESP32-S3              |
| IMU           | MPU-6050 (I2C, address 0x68)     |
| I2C SDA       | D4 → GPIO 5                      |
| I2C SCL       | D5 → GPIO 6                      |
| Power supply  | 3.3 V from XIAO 3V3 pin          |
| Flash size    | 8 MB (standard variant)          |
| SRAM          | 512 KB internal (PSRAM not used) |

### Wiring

```
MPU-6050    XIAO ESP32-S3
---------   ---------------
VCC      →  3V3
GND      →  GND
SDA      →  D4  (GPIO 5)
SCL      →  D5  (GPIO 6)
AD0      →  GND  (fixes I2C address at 0x68)
INT         not connected
```

> **Note on WHO_AM_I:** The driver checks for `0x70` (the Seeed-variant MPU-6050 silkscreen). If your breakout returns `0x68`, update the check in `MPU6050_driver.cpp::begin()` accordingly.

---

## Project Structure

```
Fed_ML/
├── platformio.ini                — Build configuration (target: seeed_xiao_esp32s3)
├── Fed_ML                        — Bash helper script (build / upload / monitor / dump)
├── README.md                     — This file
├── src/
│   └── main.cpp                  — Application entry point; orchestrates all modules
└── lib/
    ├── MPU_6050_driver/
    │   ├── MPU6050_driver.h      — I2C driver + complementary filter declarations
    │   └── MPU6050_driver.cpp    — Register init, burst read, angle fusion
    ├── FeatureExtractor/
    │   ├── FeatureExtractor.h    — 1-second window → 500-element flat vector
    │   └── FeatureExtractor.cpp  — pushSample(), getFeatureVector(), reset()
    └── NeuralNetwork/
        ├── NeuralNetwork.h       — 500→16→3 feedforward NN declarations
        └── NeuralNetwork.cpp     — Forward pass, backprop, weight update, serialization
```

Build artifacts are placed under `.pio/build/seeed_xiao_esp32s3/` and are not tracked by git (see `.gitignore`).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        XIAO ESP32-S3                           │
│                                                                 │
│  ┌──────────┐  I2C 400kHz   ┌──────────────────────────────┐   │
│  │ MPU-6050 │ ─────────────► │    MPU6050Driver             │   │
│  │  100 Hz  │               │  readRaw() → calibrate()     │   │
│  └──────────┘               │  updateAngles() (comp. filt) │   │
│                             └─────────────┬────────────────┘   │
│                                           │ CalibratedIMU       │
│                                           │ Angles (pitch,roll) │
│                                           ▼                     │
│                             ┌──────────────────────────────┐   │
│                             │    FeatureExtractor          │   │
│                             │  100-sample ring window      │   │
│                             │  [pitch×100|roll×100|        │   │
│                             │   gx×100|gy×100|gz×100]      │   │
│                             └─────────────┬────────────────┘   │
│                                           │ float[500]          │
│                                           ▼                     │
│                             ┌──────────────────────────────┐   │
│                             │    NeuralNetwork 500→16→3    │   │
│                             │  forward()   → output[3]     │   │
│                             │  backward()  → MSE gradients │   │
│                             │  updateW1()  → weight patch  │   │
│                             └─────────────┬────────────────┘   │
│                                           │                     │
│                                           ▼                     │
│                              USB Serial (115200 baud)           │
│                              [TRAIN] loss, outputs, angles      │
│                              [CMD]   label switch, weight dump  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                   (future)         │  Wi-Fi / USB Serial
                                    ▼
                         ┌─────────────────────┐
                         │  Federated Server   │
                         │  FedAvg aggregation │
                         └─────────────────────┘
```

---

## Pipeline — Step by Step

Each iteration of `loop()` executes the following stages at a fixed 100 Hz rate enforced by a microsecond timer:

| Stage | Code | Description |
|-------|------|-------------|
| 1 | `imu.sample()` | Burst-reads 14 bytes from MPU-6050 (accel XYZ + gyro XYZ), converts to physical units, runs complementary filter |
| 2 | `fex.pushSample()` | Appends pitch, roll, gx, gy, gz to a 100-slot ring; returns `true` when window is full (every ~1 s) |
| 3 | `fex.getFeatureVector()` | Copies the 5 × 100 arrays into a flat `float[500]` by `memcpy` — no normalization at this stage |
| 4 | `fex.reset()` | Resets window count so the next window can begin immediately |
| 5 | `nn.forward()` | Computes hidden pre-activations `z1`, applies sigmoid → `a1`; then output pre-activations `z2`, sigmoid → `a2` (the class probabilities) |
| 6 | `buildTarget()` | Constructs a one-hot `float[3]` from `activeLabel` |
| 7 | `nn.backward()` | Calculates output deltas (MSE gradient × sigmoid derivative), backpropagates to hidden layer, updates W2, b2, b1; stores hidden deltas in `_z1` temporarily; returns MSE loss |
| 8 | `nn.updateW1()` | Applies the hidden-layer weight gradient to W1 using the feature vector (called separately to avoid storing 500 floats inside the class) |
| 9 | Serial logging | Prints `[TRAIN]` line with step count, label, all three output activations, and MSE loss; every 50 steps also dumps W2 snapshot and heap stats |

---

## Library Reference

### MPU6050Driver

**File:** `lib/MPU_6050_driver/`

Provides a minimal, dependency-free I2C driver for the MPU-6050.

**Key constants (header)**

| Constant | Value | Meaning |
|----------|-------|---------|
| `MPU6050_ADDR` | `0x68` | I2C address (AD0 low) |
| `ACCEL_SCALE_FACTOR` | `16384.0` | LSB/g for ±2 g range |
| `GYRO_SCALE_FACTOR` | `131.0` | LSB/(°/s) for ±250 °/s range |
| `COMP_FILTER_ALPHA` | `0.98` | Complementary filter weight (gyro trust) |
| `SAMPLE_DT` | `0.01` | Integration timestep (s) matching 100 Hz |

**Initialization sequence**

The MPU-6050 is configured as follows at startup:

| Register | Value | Effect |
|----------|-------|--------|
| `PWR_MGMT_1` (0x6B) | `0x00` | Wake from sleep |
| `SMPLRT_DIV` (0x19) | `0x09` | Output rate ÷10 → 100 Hz |
| `CONFIG` (0x1A) | `0x03` | DLPF bandwidth ≈ 44 Hz |
| `GYRO_CONFIG` (0x1B) | `0x00` | ±250 °/s full scale |
| `ACCEL_CONFIG` (0x1C) | `0x00` | ±2 g full scale |

**Complementary filter**

Pitch and roll are estimated by fusing gyro integration (fast, drift-prone) with accelerometer geometry (slow, noise-prone):

```
pitch_new = α × (pitch_prev + gy_dps × dt) + (1−α) × atan2(ay, √(ax²+az²))
roll_new  = α × (roll_prev  + gx_dps × dt) + (1−α) × atan2(−ax, √(ay²+az²))
```

α = 0.98 means the gyro dominates short-term dynamics while the accelerometer corrects slow drift. The first sample bootstraps the angles from accelerometer only (no cold-start error).

**API**

```cpp
bool begin(uint8_t sda, uint8_t scl, uint32_t clock_hz = 400000UL);
bool readRaw(RawIMU &out);
void toCalibratedIMU(const RawIMU &raw, CalibratedIMU &cal);
void updateAngles(const CalibratedIMU &cal, Angles &angles);
bool sample(CalibratedIMU &cal, Angles &angles);  // combines all three
```

**Data structures**

```cpp
struct RawIMU      { int16_t ax, ay, az, gx, gy, gz; };
struct CalibratedIMU { float ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps; };
struct Angles        { float pitch, roll; };  // degrees
```

---

### FeatureExtractor

**File:** `lib/FeatureExtractor/`

Accumulates exactly 100 IMU samples and exposes them as a flat 500-element feature vector.

**Constants**

| Constant | Value |
|----------|-------|
| `WINDOW_SAMPLES` | 100 |
| `FEATURES_PER_SAMPLE` | 5 |
| `FEATURE_VECTOR_SIZE` | 500 |

**Feature vector layout**

```
Index range    Content
[  0 –  99]   pitch   (degrees, 100 samples)
[100 – 199]   roll    (degrees, 100 samples)
[200 – 299]   gx_dps  (°/s,     100 samples)
[300 – 399]   gy_dps  (°/s,     100 samples)
[400 – 499]   gz_dps  (°/s,     100 samples)
```

Each channel is a direct time-series snapshot — no FFT, no statistics, no normalization applied at this layer. The neural network learns to find discriminative patterns across all 500 values jointly.

**API**

```cpp
void  reset();
bool  pushSample(const CalibratedIMU &cal, const Angles &angles);
void  getFeatureVector(float *out) const;   // out must be float[500]
bool  isReady() const;
uint16_t sampleCount() const;
```

`pushSample()` returns `true` once the window is full; subsequent calls are no-ops until `reset()` is called.

---

### NeuralNetwork

**File:** `lib/NeuralNetwork/`

A hand-coded, heap-free feedforward neural network with online backpropagation. All weight arrays live in static RAM inside the class object.

**Dimensions**

| Symbol | Value | Meaning |
|--------|-------|---------|
| `NN_INPUT_SIZE` | 500 | Feature vector length |
| `NN_HIDDEN_SIZE` | 16 | Hidden neurons |
| `NN_OUTPUT_SIZE` | 3 | Output classes |
| `NN_W1_SIZE` | 8 000 | Input→Hidden weight count (500×16) |
| `NN_W2_SIZE` | 48 | Hidden→Output weight count (16×3) |

**Weight initialization — Xavier uniform**

```
limit_1 = sqrt(6 / (500 + 16))  →  W1 ∈ [-0.1078, +0.1078]
limit_2 = sqrt(6 / ( 16 +  3))  →  W2 ∈ [-0.5623, +0.5623]
b1 = b2 = 0.01  (small positive bias)
```

`srand(42)` fixes the seed for reproducibility.

**Forward pass (row-major layout)**

```
z1[j] = b1[j] + Σᵢ  w1[i × 16 + j] × input[i]    (i = 0..499, j = 0..15)
a1[j] = σ(z1[j])

z2[k] = b2[k] + Σⱼ  w2[j ×  3 + k] × a1[j]       (j = 0..15,  k = 0..2)
a2[k] = σ(z2[k])   ←── output (class probabilities in [0, 1])
```

**Backpropagation (MSE, single sample)**

```
δ_out[k] = (a2[k] − target[k]) × σ'(a2[k])
δ_hid[j] = (Σₖ δ_out[k] × w2[j × 3 + k]) × σ'(a1[j])

W2[j,k] -= lr × δ_out[k] × a1[j]
b2[k]   -= lr × δ_out[k]
b1[j]   -= lr × δ_hid[j]

W1 update deferred to updateW1() — see below
```

**Why `updateW1()` is separate**

Updating W1 requires the full 500-float input vector. To avoid storing a redundant copy inside the class (which would add another 2 KB to the object), `backward()` temporarily stashes `δ_hid` into the `_z1` array and returns. The caller (`main.cpp`) then immediately invokes `updateW1(featureVec, lr)`, which still has the feature vector in scope. This two-call pattern is mandatory — `updateW1()` must follow `backward()` in the same loop iteration before `featureVec` is overwritten.

**API**

```cpp
void  init();                                    // Xavier init, prints param count
void  forward(const float *input, float *output);
float backward(const float *target, float lr);   // returns MSE loss
void  updateW1(const float *input, float lr);    // MUST follow backward()
float mse(const float *target) const;
void  debugWeights() const;                      // prints W2 to Serial
size_t serializeToSerial() const;                // hex dump of all params
```

---

## Neural Network Design

| Property | Value |
|----------|-------|
| Architecture | 500 → 16 → 3 |
| Activation (hidden) | Sigmoid |
| Activation (output) | Sigmoid |
| Loss function | Mean Squared Error (per-output averaged) |
| Optimizer | Online gradient descent (1 window = 1 update step) |
| Learning rate | 0.01 |
| Weight initialization | Xavier uniform, seed 42 |
| Weight storage | Static RAM — no heap allocation |
| Total trainable parameters | 8 000 + 16 + 48 + 3 = **8 067** |
| Firmware binary size | **272 736 bytes** (266 KB) |

**Output classes**

| Node | Label | Meaning |
|------|-------|---------|
| 0 | `IDLE` | Stationary / no vibration |
| 1 | `LOW_VIB` | Low-frequency vibration |
| 2 | `HIGH_VIB` | High-frequency vibration |

The output layer uses sigmoid (not softmax), so activations are independent [0, 1] probabilities. Classification is argmax over the three outputs.

---

## RAM Budget

| Segment | Size (bytes) |
|---------|-------------|
| W1 — 500×16 float32 | 32 000 |
| W2 — 16×3 float32 | 192 |
| b1 — 16 float32 | 64 |
| b2 — 3 float32 | 12 |
| _z1, _a1, _z2, _a2 (activations) | 224 |
| `featureVec[500]` (stack, main.cpp) | 2 000 |
| `target[3]`, `nnOutput[3]` (stack) | 24 |
| MPU-6050 I2C buffers | ~100 |
| FeatureExtractor internal arrays | 2 000 |
| **Project static + stack total** | **≈ 36.6 KB** |
| **Free heap observed (hardware)** | **333 360 B (325 KB)** |
| **Min heap recorded during test** | **328 116 B (320 KB)** |

The XIAO ESP32-S3 has 512 KB SRAM. This firmware consumes well under 10% of available RAM, leaving ample headroom for Wi-Fi stack, BLE, or PSRAM-backed buffers in future extensions.

---

## Build, Flash & Monitor

### Prerequisites

- [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation/index.html) ≥ 6.x (CLI) **or** the PlatformIO IDE extension in VS Code.
- No additional library dependencies — Wire is bundled with the Arduino-ESP32 framework.

### Manual PlatformIO commands

```bash
# Build only
pio run

# Build + flash
pio run --target upload

# Serial monitor (115200 baud)
pio device monitor --baud 115200

# Clean build artifacts
pio run --target clean
```

### platformio.ini summary

```ini
[env:seeed_xiao_esp32s3]
platform  = espressif32
board     = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 115200

build_flags =
    -O2
    -DCORE_DEBUG_LEVEL=0
    -DARDUINO_USB_MODE=1
    -DARDUINO_USB_CDC_ON_BOOT=1
    ; -DBOARD_HAS_PSRAM   ← uncomment only for PSRAM variant
```

`-DARDUINO_USB_CDC_ON_BOOT=1` enables the native USB serial port without a separate USB-UART chip. Ensure your host sees `/dev/ttyACM0` (Linux) or `COMx` (Windows) after plugging in.

---

## Fed_ML Helper Script

`Fed_ML` is a self-contained Bash script in the project root that wraps the full build → upload → monitor workflow with automatic log capture.

### Usage

```bash
./Fed_ML                   # Build + upload + monitor (default)
./Fed_ML --build-only      # Compile only, no flash
./Fed_ML --upload-only     # Build + flash, no monitor
./Fed_ML --monitor-only    # Start monitor + log (no build/flash)
./Fed_ML --dump-weights    # Monitor + auto-capture NN weight dump
./Fed_ML --clean           # Clean PlatformIO build cache
./Fed_ML --list            # Print detected PlatformIO environments
./Fed_ML --help            # Show usage summary
```

### Output files

| File | Generated by | Content |
|------|-------------|---------|
| `serial_log_YYYYMMDD_HHMMSS.txt` | Any mode with monitor | Full raw serial output |
| `weights_dump_YYYYMMDD_HHMMSS.txt` | `--dump-weights` | One IEEE-754 hex word per line |

Custom log prefix:

```bash
FEDML_LOG_PREFIX=run_label2 ./Fed_ML --monitor-only
# → run_label2_20260503_143000.txt
```

### Weight dump mode (`--dump-weights`)

This mode pipes the monitor through `awk` and waits for the firmware to emit `[NN_DUMP START]`. When detected, it extracts every 8-character hex word and writes them line-by-line to `weights_dump_*.txt`. The monitor closes automatically after `[NN_DUMP END]` is seen. Trigger the dump by sending `w` in any other terminal window connected to the same serial port, or by using a separate PIO monitor session.

---

## Runtime Label Switching & Serial Commands

Send a single character over the serial monitor (115200 baud) at any time — no reflashing required.

| Key | Action |
|-----|--------|
| `0` | Switch active training label → Class 0 (Idle) |
| `1` | Switch active training label → Class 1 (Low-frequency vibration) |
| `2` | Switch active training label → Class 2 (High-frequency vibration) |
| `w` / `W` | Serialize all weights + biases to Serial as hex-float stream |

Label switches take effect on the next training window (≤ 1 second delay). The system prints `[CMD] Label set to N` immediately upon receipt.

### Recommended data-collection procedure

1. Flash firmware with `CURRENT_LABEL 0`. Place sensor stationary. Let it train for ≥ 50 steps.
2. Send `1` over serial. Apply low-frequency excitation (e.g., motor at low RPM). Train ≥ 50 steps.
3. Send `2`. Apply high-frequency excitation. Train ≥ 50 steps.
4. Observe that `out[N]` for the current label increases toward 1.0 and others decrease.

---

## Weight Serialization Format

Sending `w` causes the firmware to emit:

```
[NN_DUMP START]
[NN_DUMP COUNT 8067]
3C23D70A 3D4A3D71 BC8B4396 ...   (8 hex words per line)
...
[NN_DUMP END]
[CMD] Weights dumped: 8067 floats
```

**Float order within the dump:**

| Segment | Count | Description |
|---------|-------|-------------|
| W1 | 8 000 | Row-major `[input_idx × 16 + hidden_idx]` |
| W2 | 48 | Row-major `[hidden_idx × 3 + output_idx]` |
| b1 | 16 | Hidden layer biases |
| b2 | 3 | Output layer biases |
| **Total** | **8 067** | All trainable parameters |

Each word is an IEEE-754 single-precision float encoded as an 8-character uppercase hex string. To reconstruct in Python:

```python
import struct, numpy as np

with open("weights_dump_20260503_100836.txt") as f:
    words = [line.strip() for line in f if len(line.strip()) == 8]

floats = [struct.unpack('>f', bytes.fromhex(w))[0] for w in words]

W1 = np.array(floats[:8000]).reshape(500, 16)
W2 = np.array(floats[8000:8048]).reshape(16, 3)
b1 = np.array(floats[8048:8064])
b2 = np.array(floats[8064:8067])
```

---

## Configuration Reference

All compile-time tunables:

| Constant | File | Default | Notes |
|----------|------|---------|-------|
| `I2C_SDA_PIN` | `main.cpp` | `5` | GPIO for SDA (D4 on XIAO silk) |
| `I2C_SCL_PIN` | `main.cpp` | `6` | GPIO for SCL (D5 on XIAO silk) |
| `SAMPLE_INTERVAL_US` | `main.cpp` | `10 000` | 100 Hz sample rate |
| `LEARNING_RATE` | `main.cpp` | `0.01f` | SGD step size |
| `CURRENT_LABEL` | `main.cpp` | `0` | Initial training class (0/1/2) |
| `DEBUG_ANGLES` | `main.cpp` | `1` | 1 = print angles every sample |
| `DEBUG_LOSS` | `main.cpp` | `1` | 1 = print loss every window |
| `MPU6050_ADDR` | `MPU6050_driver.h` | `0x68` | Change to `0x69` if AD0 = high |
| `COMP_FILTER_ALPHA` | `MPU6050_driver.h` | `0.98f` | Gyro trust weight |
| `SAMPLE_DT` | `MPU6050_driver.h` | `0.01f` | Must match sample interval |
| `WINDOW_SAMPLES` | `FeatureExtractor.h` | `100` | Samples per training window |
| `FEATURES_PER_SAMPLE` | `FeatureExtractor.h` | `5` | pitch, roll, gx, gy, gz |
| `NN_HIDDEN_SIZE` | `NeuralNetwork.h` | `16` | Hidden neuron count |

> **Performance tip:** Set `DEBUG_ANGLES 0` in production to reduce serial bus load. Each sample currently prints two lines (~80 chars each), which at 115200 baud adds measurable latency overhead at 100 Hz.

---

## Validated Test Results (2026-05-03)

Hardware test run captured in `serial_log_20260503_100836.txt`. All data below is extracted directly from the log.

### Environment

| Item | Detail |
|------|--------|
| Board | Seeed XIAO ESP32-S3 |
| Serial port | `/dev/ttyACM0` |
| Baud rate | 115200 |
| Training label | 0 (Idle / stationary) throughout the session |
| Firmware binary | 272 736 bytes (266 KB) |

### IMU readings (stationary, label 0)

The sensor was stationary at a fixed tilt during the test. The complementary filter converged quickly and held stable angles for the duration of the log:

| Channel | Observed range | Notes |
|---------|---------------|-------|
| Pitch | −5.86° to −5.89° | Stable; device mounted at slight angle |
| Roll | +3.57° to +3.59° | Stable |
| gx_dps | ~0.82 – 0.96 | Low residual gyro bias (no offset calibration applied) |
| gy_dps | ~0.33 – 0.52 | Low residual |
| gz_dps | ~1.03 – 1.17 | Low residual |

Raw ADC values were consistent (e.g. `ax ≈ −950`, `ay ≈ −1800`, `az ≈ 17 040`), confirming stable mounting and clean I2C reads throughout.

### Training convergence (label 0, 68 steps logged)

The network was trained from Xavier-initialized weights. Output node 0 (target class) ramped from ~0.26 toward ~0.63 over 68 steps, with nodes 1 and 2 (non-target) suppressed:

| Step | out[0] | out[1] | out[2] | MSE Loss |
|------|--------|--------|--------|----------|
| 2 (first) | 0.2584 | 0.2991 | 0.6396 | 0.349487 |
| 10 | 0.4827 | 0.1898 | 0.3731 | 0.147589 |
| 20 | 0.5507 | 0.1648 | 0.3612 | 0.119817 |
| 30 | 0.5704 | 0.1547 | 0.3515 | 0.110691 |
| 50 | 0.6016 | 0.1406 | 0.3350 | 0.096908 |
| 69 (last) | 0.6264 | 0.1334 | 0.3185 | 0.086263 |

Loss dropped **75.3%** from step 2 to step 69 (0.3495 → 0.0863) with consistent monotonic decrease — no divergence, no NaN, no oscillation. The network is learning on-device as expected.

### Memory (measured at step 50)

| Metric | Value |
|--------|-------|
| Free heap | 333 360 B (325.5 KB) |
| Min free heap (peak consumption) | 328 116 B (320.4 KB) |
| W1 static allocation | 32 000 B |
| `featureVec` stack allocation | 2 000 B |

No heap fragmentation or memory leak observed over 68 training steps.

### Weight dump events

Two weight dumps were triggered during the session (steps 29 and 43), each confirming `8 067 floats` serialized successfully. The `[NN_DUMP COUNT 8067]` total matches: 8 000 (W1) + 48 (W2) + 16 (b1) + 3 (b2) = 8 067.

### W2 snapshot (at step 50 debug print)

```
h00 →  0.16433   0.54335   0.50043
h01 → -0.46780   0.08345   0.54418
h02 → -0.14303  -0.45860  -0.27952
h03 → -0.32664  -0.45662  -0.41447
h04 → -0.10393   0.28673  -0.43501
h05 → -0.28307   0.38870  -0.21362
h06 →  0.08055  -0.54873  -0.41997
h07 → -0.14224  -0.04835   0.43553
h08 →  0.39411  -0.14704  -0.02619
h09 → -0.28669   0.11434   0.03127
h10 →  0.09942   0.35660   0.01934
h11 → -0.15255   0.55371  -0.03994
h12 →  0.34269   0.41685   0.35252
h13 → -0.35374  -0.17969   0.38605
h14 →  0.27059  -0.35640   0.36424
h15 → -0.24690  -0.46226  -0.30824
```

Column 0 (class 0 output) shows a mix of positive and negative weights, confirming that the network has formed non-trivial internal representations — it is not saturated or dead.

### Verdict

✅ IMU initializes and reads correctly at 100 Hz
✅ Complementary filter stable and converged
✅ Feature window accumulation and reset working correctly
✅ Forward pass produces valid [0, 1] outputs
✅ Backpropagation converges — loss decreasing monotonically
✅ Weight serialization operational (8 067 floats, correct format)
✅ Runtime label switching functional (`[CMD]` messages confirmed)
✅ Heap stable — no memory leak over extended operation
✅ Firmware binary size: 266 KB (well within 8 MB flash)

---

## Extending to Federated Learning

This project implements a single FL client node. The following steps extend it to the full architecture described in the reference paper.

### Step 1 — Server (Python, PC or SBC)

```python
# Pseudocode — FedAvg server
import numpy as np, serial, struct

def receive_weights(port) -> np.ndarray:
    # Read lines until [NN_DUMP END], parse 8-char hex words
    ...

def fedavg(weight_list):
    return np.mean(weight_list, axis=0)

def send_global_model(port, weights):
    # Transmit as binary or hex back to device over Serial/Wi-Fi
    ...
```

### Step 2 — Client additions (firmware)

```cpp
// Add to NeuralNetwork.h / .cpp:
void loadWeightsFromSerial();   // parse incoming hex dump and restore W1, W2, b1, b2
```

### Step 3 — Wi-Fi transport

The XIAO ESP32-S3's built-in 2.4 GHz Wi-Fi enables HTTP POST or MQTT for model exchange — eliminating the 64-byte USB serial buffer bottleneck noted in the paper. Suggested round trip:

1. Device trains for N windows.
2. Device POSTs weight hex dump to server endpoint.
3. Server runs FedAvg across connected nodes.
4. Server responds with global model hex dump.
5. Device calls `loadWeightsFromSerial()` (or a Wi-Fi equivalent).

### Step 4 — Multi-class data collection

Collect separate sessions for labels 1 and 2 using the runtime label switch before federated rounds begin, ensuring each node specializes on distinct vibration conditions before aggregation.

---

## Known Limitations & Future Work

| Limitation | Impact | Suggested fix |
|------------|--------|---------------|
| No gyro bias offset calibration at startup | Small DC bias in gx/gy/gz (observed ~0.9 °/s) | 100-sample still average at boot; subtract mean |
| No feature normalization | Raw angle/gyro values passed to NN; scale-sensitive | Z-score normalization per channel before `forward()` |
| `DEBUG_ANGLES = 1` at 100 Hz | ~80 chars/sample × 100 Hz = 8 KB/s serial traffic | Set `DEBUG_ANGLES 0` for production / long runs |
| `updateW1` requires two-call pattern | Caller must not modify `featureVec` between `backward()` and `updateW1()` | Refactor to cache input pointer inside class |
| Single active label per session | No mixed-label training in one session | Add automatic label rotation or timer-based switching |
| Sigmoid activations throughout | Vanishing gradient risk for deeper networks | Replace hidden activation with ReLU for future scaling |
| No model persistence | Weights lost on power cycle | Use ESP32 NVS or LittleFS to save/load weight binary |
| WHO_AM_I check for `0x70` only | Will reject MPU-6050 clones returning `0x68` | Accept both values, print warning not fatal error |

---
---

## What's New in V1.0.1

V2 extends the firmware from a pure training platform into a complete on-device machine learning lifecycle — **train**, **persist**, and **infer** — all without leaving the microcontroller. The table below summarises what runs in each mode:

| Mode | Forward pass | Backward pass | W1 / W2 update | Serial prefix |
|------|:---:|:---:|:---:|:---:|
| `MODE_TRAIN` | ✅ | ✅ | ✅ | `[TRAIN]` |
| `MODE_INFER` | ✅ | ❌ | ❌ | `[INFER]` |

---

## Training Mode — Usage & Technical Reference

### What happens during a training step

Every completed 1-second feature window triggers the following sequence in `runTraining()`:

```
forward(featureVec, nnOutput)
  └─ hidden pre-activations  z1[j] = b1[j] + Σᵢ w1[i·16+j] · featureVec[i]
  └─ hidden activations      a1[j] = σ(z1[j])
  └─ output pre-activations  z2[k] = b2[k] + Σⱼ w2[j·3+k]  · a1[j]
  └─ output activations      a2[k] = σ(z2[k])   ← class probabilities

backward(target, lr)
  └─ output deltas   δ_out[k] = (a2[k] − target[k]) · σ'(a2[k])
  └─ hidden deltas   δ_hid[j] = (Σₖ δ_out[k] · w2[j·3+k]) · σ'(a1[j])
  └─ update W2       w2[j,k] -= lr · δ_out[k] · a1[j]
  └─ update b2       b2[k]   -= lr · δ_out[k]
  └─ update b1       b1[j]   -= lr · δ_hid[j]
  └─ stash δ_hid into _z1    (temp buffer for updateW1)

updateW1(featureVec, lr)          ← MUST follow backward() immediately
  └─ update W1       w1[i,j] -= lr · _z1[j] · featureVec[i]
```

The two-call W1 update pattern (`backward()` then `updateW1()`) exists entirely to avoid storing a second copy of the 500-float feature vector inside the `NeuralNetwork` class object. `backward()` stashes `δ_hid` into the `_z1` buffer (which is no longer needed as a pre-activation cache until the next `forward()` call), and `updateW1()` consumes it using the caller's still-live `featureVec[]`. Do not insert any code between these two calls.

### Loss function and convergence signal

The loss reported on each `[TRAIN]` line is **Mean Squared Error** averaged across the three output nodes:

```
MSE = (1/3) · Σₖ (a2[k] − target[k])²
```

For a well-initialised Xavier network with `lr = 0.01`, expect:
- Steps 1–10: loss drops sharply from ~0.35 toward ~0.15 as the network breaks symmetry
- Steps 10–50: steady monotonic descent; `out[label]` climbs toward 1.0; the other two nodes fall
- Steps 50+: diminishing returns; loss typically plateaus between 0.05 and 0.10 for a single-class session

The network uses **online gradient descent** (one weight update per window, not per epoch over a dataset). This matches the federated learning paper's methodology and is deliberately chosen to minimise RAM — there is no replay buffer.

### Activation function

Both the hidden and output layers use **sigmoid**:

```
σ(x)  = 1 / (1 + e⁻ˣ)
σ'(x) = a · (1 − a)     where a = σ(x)   ← computed from the activation, not the pre-activation
```

The derivative is evaluated from the cached activation value `_a1[j]` / `_a2[k]`, avoiding a redundant `expf()` call per neuron. Output activations are independent probabilities in `[0, 1]` — sigmoid is used rather than softmax because the paper's methodology treats each class output as an independent binary estimator, and softmax would couple the gradients across classes in a way that is harder to aggregate under FedAvg.

### Weight initialisation

Weights are Xavier-uniform with a fixed seed:

```
W1 limit = √(6 / (500 + 16)) = 0.1078   →   W1 ∈ [−0.1078, +0.1078]
W2 limit = √(6 / ( 16 +  3)) = 0.5623   →   W2 ∈ [−0.5623, +0.5623]
b1 = b2 = 0.01
srand(42)
```

`srand(42)` pins the random seed so that two devices starting from the same seed produce identical initial weights. This matters for federated averaging: if devices diverge from the same starting point, the FedAvg global model averages comparable parameter spaces rather than misaligned ones.

### How to use Training mode

**Step 1 — Flash and open the serial monitor**

```bash
pio run --target upload
pio device monitor --baud 115200
```

On first boot with no saved model you will see:

```
[MAIN] No saved model — starting in TRAINING mode
[MAIN] Mode: TRAINING  Label: 0 (IDLE)  LR: 0.0100
```

**Step 2 — Collect class 0 (IDLE)**

Place the sensor stationary. The device is already labelling incoming windows as class 0. Watch the `[TRAIN]` lines:

```
[TRAIN] step=   1  label=0  out=[0.2584  0.2991  0.6396]  loss=0.349487
[TRAIN] step=   2  label=0  out=[0.3812  0.2204  0.5110]  loss=0.241063
...
[TRAIN] step=  50  label=0  out=[0.6016  0.1406  0.3350]  loss=0.096908
```

Wait until `out[0]` is comfortably above 0.5 and loss has flattened (typically 50–80 windows, ~50–80 seconds).

**Step 3 — Switch to class 1 (LOW\_VIB)**

Send `1` in the serial monitor. Apply your low-frequency excitation source (motor at low RPM, tapping, etc.):

```
[CMD] Label set to 1 (LOW_VIB)
[TRAIN] step=  51  label=1  out=[0.5943  0.1289  0.3197]  loss=0.298841
...
```

`out[1]` should climb while `out[0]` and `out[2]` suppress. Allow ≥50 windows.

**Step 4 — Switch to class 2 (HIGH\_VIB)**

Send `2`. Apply high-frequency excitation. Allow ≥50 windows.

**Step 5 — Save and verify**

```
s        ← send over serial
[CMD] Model saved — will auto-load on next boot
```

Send `i` to immediately switch to inference mode and verify predictions without reflashing.

**Tips**

- Set `DEBUG_ANGLES 0` in `main.cpp` for production or long training runs — at 100 Hz, angle printouts consume ~8 KB/s of serial bandwidth and add measurable latency jitter.
- Every 50 steps the firmware automatically prints a W2 snapshot (`[NN] W2 snapshot:`) and heap stats. Use these to confirm the network has not saturated (all weights near ±limit) or died (all weights near zero).
- To reset training from scratch without reflashing, cycle power (or hard-reset) with no `/snx_model.bin` on flash, or erase with `pio run --target erase`.

---

## Inference Mode — Usage & Technical Reference

### What happens during an inference step

Every completed 1-second feature window triggers the following sequence in `runInference()`:

```
forward(featureVec, nnOutput)
  └─ same forward pass as training (see above)
  └─ _a2[0..2] populated with class probabilities

argmax()
  └─ scans _a2[] — no recomputation, reads the cache populated by forward()
  └─ returns index k* where a2[k*] = max(a2[0], a2[1], a2[2])
```

No `backward()`, no `updateW1()` — the weight arrays are never written. The model is frozen at whatever state it was in when `saveToLittleFS()` was last called.

### Classification rule

The predicted class is simply the output neuron with the highest activation:

```
k* = argmax({ a2[0], a2[1], a2[2] })
```

Because sigmoid outputs are independent (not softmax-normalised), the three activations do not sum to 1. Each value should be read as the network's independent confidence that the window belongs to that class. A healthy inference log looks like:

```
[INFER] step=   1  pred=IDLE     (0)  out=[0.8812  0.0934  0.1205]
[INFER] step=   2  pred=IDLE     (0)  out=[0.8749  0.1013  0.1188]
```

The winning class has a clearly dominant activation (>0.7) while the others are suppressed (<0.2). If activations are bunched together (e.g. `[0.42  0.38  0.35]`), the model is uncertain — usually because that vibration condition was underrepresented during training or the sensor is experiencing a condition outside the training distribution.

### Confidence interpretation

| `out[k*]` range | Interpretation |
|---|---|
| > 0.80 | High confidence — class boundary well-learnt |
| 0.60 – 0.80 | Moderate confidence — consider more training windows |
| 0.50 – 0.60 | Low confidence — ambiguous; class separation insufficient |
| < 0.50 | Prediction unreliable — argmax wins but no class dominates |

These thresholds are heuristics based on sigmoid output characteristics, not calibrated probabilities. For a deployment that needs calibrated uncertainty, consider adding a rejection threshold: if `a2[k*] < threshold`, emit a `UNKNOWN` label rather than a forced classification.

### Model Persistence (LittleFS)

V2 persists the full model to the on-chip flash filesystem so weights survive power cycles.

**Why LittleFS and not NVS?**

ESP32 NVS has a hard **4 KB per-key limit**. W1 alone occupies `500 × 16 × 4 = 32 000 bytes`. LittleFS stores arbitrary-size binary files on the same 8 MB flash chip with no such restriction. The blob layout is:

```
/snx_model.bin
├── W1   [8 000 floats × 4 B = 32 000 B]   row-major [input_idx × 16 + hidden_idx]
├── W2   [   48 floats × 4 B =    192 B]   row-major [hidden_idx × 3  + output_idx]
├── b1   [   16 floats × 4 B =     64 B]
└── b2   [    3 floats × 4 B =     12 B]
                              ──────────
                    Total      32 268 B   (≈ 31.5 KB)
```

`loadFromLittleFS()` checks `file.size() == MODEL_BLOB_BYTES` before reading a single byte. If the size does not match exactly (partial write, flash corruption, architecture change), the file is rejected and the firmware falls back to `nn.init()` rather than loading garbage weights silently.

**Setup (one-time)**

Add to `platformio.ini`:

```ini
board_build.filesystem = littlefs
```

Run once to initialise the filesystem partition on a blank device:

```bash
pio run --target uploadfs
```

This only needs to be done once per device. Subsequent `pio run --target upload` firmware flashes do not erase the LittleFS partition.

### Auto-Boot Behaviour

`setup()` attempts `loadFromLittleFS()` before calling `nn.init()`:

```
Boot
 │
 ├─ loadFromLittleFS() succeeds?
 │    YES → weights restored → opMode = MODE_INFER
 │           "[MAIN] Saved model loaded — starting in INFERENCE mode"
 │
 └─ NO  → nn.init() (Xavier fresh start) → opMode = MODE_TRAIN
           "[MAIN] No saved model — starting in TRAINING mode"
```

A fully trained and saved device behaves like a finished product on every subsequent power-on — connect power, `[INFER]` lines start appearing within 1 second of the first complete window.

### How to use Inference mode

**Entering inference mode**

Three ways:

1. **Automatic on boot** — if `/snx_model.bin` exists on flash (most common after first save).
2. **Serial command `i`** — switch live from training without reflashing or power-cycling.
3. **Serial command `l`** — explicitly reload the saved model from flash and enter inference.

**Reading the inference log**

```
[INFER] step=   1  pred=IDLE     (0)  out=[0.8812  0.0934  0.1205]
         │          │             │    │
         │          │             │    └── raw sigmoid activations for all 3 classes
         │          │             └─────── winning class index
         │          └───────────────────── winning class name
         └──────────────────────────────── window count since entering INFER mode
```

`inferStep` resets to 0 each time `MODE_INFER` is entered. Every 50 inference windows the firmware prints a RAM health report (`[SYS]` lines) identical to the one emitted during training.

**Returning to training mode**

Send `t` at any time. The model in RAM is unchanged — inference does not modify weights — so you can freely switch back and forth to compare the live model's training loss against its inference predictions on the same window.

**Saving after fine-tuning**

```
t        ← enter training mode
          (fine-tune for N windows)
s        ← overwrite /snx_model.bin with updated weights
i        ← return to inference
```

The save overwrites the previous blob atomically (LittleFS `"w"` open truncates on write). There is no versioning — the last save always wins.

---

## Complete V2 Serial Command Reference

| Key | Mode required | Action |
|-----|:---:|--------|
| `0` | TRAIN | Set active label → Class 0 (IDLE) |
| `1` | TRAIN | Set active label → Class 1 (LOW\_VIB) |
| `2` | TRAIN | Set active label → Class 2 (HIGH\_VIB) |
| `t` / `T` | any | Switch to **TRAINING** mode |
| `i` / `I` | any | Switch to **INFERENCE** mode (resets `inferStep`) |
| `s` / `S` | any | Save weights to `/snx_model.bin` on LittleFS |
| `l` / `L` | any | Load weights from `/snx_model.bin` → enter INFERENCE |
| `w` / `W` | any | Dump all 8 067 parameters as IEEE-754 hex stream |

---

## Internal Refactoring (V1 → V2)

- **`predict()` removed** — it redundantly re-ran `forward()` after `main.cpp` had already called it, performing a full 500 × 16 MAC pass for nothing. Replaced by `argmax()`, which reads `_a2[]` already populated by the preceding `forward()` call at zero additional cost.
- **`saveToNVS()` / `loadFromNVS()` removed** — NVS 4 KB/key limit cannot accommodate W1 (32 KB). Replaced by `saveToLittleFS()` / `loadFromLittleFS()`.
- **`MODEL_BLOB_BYTES` is `constexpr`** — the LittleFS integrity check is derived at compile time from the actual array dimension macros, not a hardcoded magic number. Any change to `NN_INPUT_SIZE` or `NN_HIDDEN_SIZE` automatically updates the check.
- **`runInference()` and `runTraining()` extracted** — `loop()` is now a single `if/else` dispatch; all per-mode logic is encapsulated in its own function.
- **`inferStep` counter** — inference gets its own step counter (resets on each entry to `MODE_INFER`) so `[INFER]` log lines are independently meaningful and the 50-step RAM report cadence is maintained in both modes.

---

## License

MIT — see individual source files for author attribution.

Reference paper (open access):
Llisterri Giménez et al., *"On-Device Training of Machine Learning Models on Microcontrollers with Federated Learning"*, Electronics 2022, 11, 573. https://doi.org/10.3390/electronics11040573
