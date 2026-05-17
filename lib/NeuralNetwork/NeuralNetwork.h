#pragma once
// NeuralNetwork.h
// Feedforward neural network: 500 → 16 → 3
// Implements forward propagation and single-sample online
// backpropagation (gradient descent), analogous to the
// on-device training described in the paper.
//
// All weights live in static RAM arrays — no heap allocation.
//
// Layer notation
//   L0 (input)  : INPUT_SIZE  = 500  neurons
//   L1 (hidden) : HIDDEN_SIZE = 16   neurons  (sigmoid)
//   L2 (output) : OUTPUT_SIZE = 3    neurons  (sigmoid)
//
// Serial commands (handled externally in main.cpp):
//   'i'/'I' → inference mode  — forward() only, no weight update
//   't'/'T' → training mode   — forward() + backward() + updateW1()
//   's'/'S' → save weights to LittleFS (/snx_model.bin)
//   'l'/'L' → load weights from LittleFS (/snx_model.bin)
//   'w'/'W' → dump weights as IEEE-754 hex stream

#include <Arduino.h>
#include <string.h>

//Network dimensions 
#define NN_INPUT_SIZE  500
#define NN_HIDDEN_SIZE 16
#define NN_OUTPUT_SIZE 3

// Flat weight / bias array sizes
#define NN_W1_SIZE (NN_INPUT_SIZE  * NN_HIDDEN_SIZE)  // 8 000
#define NN_W2_SIZE (NN_HIDDEN_SIZE * NN_OUTPUT_SIZE)  //    48
#define NN_B1_SIZE  NN_HIDDEN_SIZE                     //    16
#define NN_B2_SIZE  NN_OUTPUT_SIZE                     //     3

// Total parameters: 8 067
// W1 occupies 32 000 B of static RAM — exceeds ESP32 NVS 4 KB/key limit,
// so persistence uses LittleFS (binary blob, /snx_model.bin).

class NeuralNetwork {
public:
    NeuralNetwork();

    // Xavier-uniform weight initialisation (seed 42 for reproducibility).
    // Must be called once in setup() before any forward/backward calls.
    void init();

    //Core inference
    // Forward pass. Populates internal _a1, _a2 activation caches.
    //   input  : float[NN_INPUT_SIZE]
    //   output : float[NN_OUTPUT_SIZE]  (written by this call; == _a2)
    void forward(const float *input, float *output);

    // Returns the argmax class index (0/1/2) of the last forward() call.
    // Does NOT re-run forward() — call forward() first.
    uint8_t argmax() const;

    // Returns _a2[classIdx] from the last forward() call.
    // Valid for any classIdx in [0, NN_OUTPUT_SIZE).
    float getOutput(uint8_t classIdx) const;

    //Training
    // Backpropagation + bias/W2 update (MSE loss, gradient descent).
    // Stashes delta1 into _z1 so updateW1() can consume it.
    //   target : one-hot float[NN_OUTPUT_SIZE]
    //   Returns MSE loss for this sample.
    // MUST be followed immediately by updateW1() before featureVec changes.
    float backward(const float *target, float lr);

    // Applies the W1 gradient using delta1 stashed in _z1 by backward().
    // MUST be called immediately after backward(), while `input` (the same
    // float[NN_INPUT_SIZE] passed to forward()) is still valid in the caller.
    void updateW1(const float *input, float lr);

    //Diagnostics
    // Compute MSE between _a2 and target for the last forward() result.
    float mse(const float *target) const;

    // Print a compact W2 snapshot to Serial.
    void debugWeights() const;

    // Serialize all parameters to Serial as hex-encoded IEEE-754 words.
    // Format:
    //   [NN_DUMP START]
    //   [NN_DUMP COUNT <N>]
    //   <HEX32> <HEX32> ...   (8 words per line)
    //   [NN_DUMP END]
    // Order: W1 | W2 | b1 | b2
    size_t serializeToSerial() const;

    // Persistence (LittleFS)
    // Save all weights + biases to LittleFS as a raw binary blob.
    // W1 is 32 KB; NVS cannot hold it — LittleFS is used instead.
    // platformio.ini must include: board_build.filesystem = littlefs
    // Returns true on success.
    bool saveToLittleFS(const char *path = "/snx_model.bin");

    // Load weights + biases from a previously saved LittleFS blob.
    // Returns true if the file exists, has the correct size, and is loaded.
    // Call in setup() before nn.init() to resume a trained model;
    // if it returns false, fall back to init() for fresh Xavier weights.
    bool loadFromLittleFS(const char *path = "/snx_model.bin");

private:
    //Weight / bias arrays (static RAM, no heap)
    float _w1[NN_W1_SIZE];   // L0→L1  row-major [input_idx * 16 + hidden_idx]
    float _w2[NN_W2_SIZE];   // L1→L2  row-major [hidden_idx * 3 + output_idx]
    float _b1[NN_B1_SIZE];
    float _b2[NN_B2_SIZE];

    //Activation caches (reused across calls)
    float _z1[NN_HIDDEN_SIZE];   // pre-activation hidden  (also: delta1 buffer after backward())
    float _a1[NN_HIDDEN_SIZE];   // post-activation hidden
    float _z2[NN_OUTPUT_SIZE];   // pre-activation output
    float _a2[NN_OUTPUT_SIZE];   // post-activation output  (== forward output)

    //Core math helpers
    static float sigmoid(float x);
    static float sigmoidDerivative(float activated);   // σ'(z) = a(1 − a)
    static float xavierRand(float limit);

    // Expected binary blob size for LittleFS integrity check
    static constexpr size_t MODEL_BLOB_BYTES =
        (NN_W1_SIZE + NN_W2_SIZE + NN_B1_SIZE + NN_B2_SIZE) * sizeof(float);
};