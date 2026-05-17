// NeuralNetwork.cpp
// On-device feedforward NN with online backpropagation.
// See NeuralNetwork.h for full API documentation.

#include "NeuralNetwork.h"
#include <math.h>
#include <stdlib.h>
#include <LittleFS.h>

// Constructor — zero-initialise all arrays so no stale data
// is ever used if forward() is somehow called before init().
NeuralNetwork::NeuralNetwork()
{
    memset(_w1, 0, sizeof(_w1));
    memset(_w2, 0, sizeof(_w2));
    memset(_b1, 0, sizeof(_b1));
    memset(_b2, 0, sizeof(_b2));
    memset(_z1, 0, sizeof(_z1));
    memset(_a1, 0, sizeof(_a1));
    memset(_z2, 0, sizeof(_z2));
    memset(_a2, 0, sizeof(_a2));
}

// init() — Xavier-uniform initialisation
// Xavier limit for a layer: sqrt(6 / (fan_in + fan_out))
// srand(42) fixes the random seed for reproducibility.
void NeuralNetwork::init()
{
    srand(42);
    float lim1 = sqrtf(6.0f / (float)(NN_INPUT_SIZE  + NN_HIDDEN_SIZE));
    float lim2 = sqrtf(6.0f / (float)(NN_HIDDEN_SIZE + NN_OUTPUT_SIZE));

    for (int i = 0; i < NN_W1_SIZE; i++) _w1[i] = xavierRand(lim1);
    for (int i = 0; i < NN_W2_SIZE; i++) _w2[i] = xavierRand(lim2);
    for (int i = 0; i < NN_B1_SIZE; i++) _b1[i] = 0.01f;
    for (int i = 0; i < NN_B2_SIZE; i++) _b2[i] = 0.01f;

    Serial.printf("[NN] Weights initialised — W1:%d  W2:%d  params total:%d\n",
                  NN_W1_SIZE, NN_W2_SIZE,
                  NN_W1_SIZE + NN_W2_SIZE + NN_B1_SIZE + NN_B2_SIZE);
}

// forward()
//
// Hidden layer:
//   z1[j] = b1[j] + Σ_i( w1[i*16 + j] * input[i] )
//   a1[j] = σ(z1[j])
//
// Output layer:
//   z2[k] = b2[k] + Σ_j( w2[j*3 + k] * a1[j] )
//   a2[k] = σ(z2[k])          ← written to output[]
void NeuralNetwork::forward(const float *input, float *output)
{
    // Hidden layer
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        float sum = _b1[j];
        for (int i = 0; i < NN_INPUT_SIZE; i++) {
            sum += _w1[i * NN_HIDDEN_SIZE + j] * input[i];
        }
        _z1[j] = sum;
        _a1[j] = sigmoid(sum);
    }

    // Output layer
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        float sum = _b2[k];
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            sum += _w2[j * NN_OUTPUT_SIZE + k] * _a1[j];
        }
        _z2[k] = sum;
        _a2[k] = sigmoid(sum);
        output[k] = _a2[k];
    }
}

// argmax()
// Returns the index of the highest activation in _a2.
// Does NOT re-run forward() — call forward() first.
// Used by main.cpp's runInference() to classify without weight update.
uint8_t NeuralNetwork::argmax() const
{
    uint8_t best = 0;
    for (uint8_t k = 1; k < NN_OUTPUT_SIZE; k++) {
        if (_a2[k] > _a2[best]) best = k;
    }
    return best;
}

// getOutput()
// Returns _a2[classIdx] from the last forward() call.
// Bounds-checked: returns 0.0 for out-of-range index.
float NeuralNetwork::getOutput(uint8_t classIdx) const
{
    if (classIdx >= NN_OUTPUT_SIZE) return 0.0f;
    return _a2[classIdx];
}

// backward() — single-sample MSE gradient descent
//
// Output deltas:
//   δ_out[k] = (a2[k] − target[k]) · σ'(a2[k])
//
// Hidden deltas:
//   δ_hid[j] = (Σ_k δ_out[k] · w2[j*3+k]) · σ'(a1[j])
//
// W2 + bias updates happen here.
// W1 update is deferred to updateW1() — see header for rationale.
// delta1 is stashed into _z1 so updateW1() can read it.
//
// IMPORTANT: updateW1() MUST be called before featureVec changes.
float NeuralNetwork::backward(const float *target, float lr)
{
    // Output layer deltas + MSE loss
    float delta2[NN_OUTPUT_SIZE];
    float loss = 0.0f;
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        float err  = _a2[k] - target[k];
        loss      += err * err;
        delta2[k]  = err * sigmoidDerivative(_a2[k]);
    }
    loss /= (float)NN_OUTPUT_SIZE;

    // Hidden layer deltas
    float delta1[NN_HIDDEN_SIZE];
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        float grad = 0.0f;
        for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
            grad += delta2[k] * _w2[j * NN_OUTPUT_SIZE + k];
        }
        delta1[j] = grad * sigmoidDerivative(_a1[j]);
    }

    // Update W2
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
            _w2[j * NN_OUTPUT_SIZE + k] -= lr * delta2[k] * _a1[j];
        }
    }

    // Update b2
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        _b2[k] -= lr * delta2[k];
    }

    // Update b1
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        _b1[j] -= lr * delta1[j];
    }

    // Stash delta1 into _z1 so updateW1() can consume it.
    // _z1 is no longer needed as a pre-activation cache until the
    // next forward() call, which overwrites it anyway.
    memcpy(_z1, delta1, NN_HIDDEN_SIZE * sizeof(float));

    return loss;
}

// updateW1()
// Applies the W1 gradient: ΔW1[i,j] = -lr · delta1[j] · input[i]
// _z1 holds delta1, written there by the preceding backward() call.
// Must be called in the same loop iteration as backward(), before
// the caller's featureVec[] is overwritten by the next window.
void NeuralNetwork::updateW1(const float *input, float lr)
{
    // _z1 now holds delta1 (stashed by backward())
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            _w1[i * NN_HIDDEN_SIZE + j] -= lr * _z1[j] * input[i];
        }
    }
}

// mse()
// Compute MSE between _a2 (last forward result) and target[].
float NeuralNetwork::mse(const float *target) const
{
    float loss = 0.0f;
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        float e = _a2[k] - target[k];
        loss += e * e;
    }
    return loss / (float)NN_OUTPUT_SIZE;
}

void NeuralNetwork::debugWeights() const
{
    Serial.println("[NN] W2 snapshot:");
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        Serial.printf("  h%02d → ", j);
        for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
            Serial.printf("%8.5f ", _w2[j * NN_OUTPUT_SIZE + k]);
        }
        Serial.println();
    }
}

// serializeToSerial()
// Dumps all 8 067 parameters as 8-char uppercase hex words,
// 8 per line, in order: W1 | W2 | b1 | b2.
// Trigger with serial command 'w'/'W'.
size_t NeuralNetwork::serializeToSerial() const
{
    const uint32_t total = NN_W1_SIZE + NN_W2_SIZE + NN_B1_SIZE + NN_B2_SIZE;
    Serial.println("[NN_DUMP START]");
    Serial.printf("[NN_DUMP COUNT %u]\n", (unsigned)total);

    auto printHex = [](float v) {
        uint32_t u;
        memcpy(&u, &v, sizeof(u));
        Serial.printf("%08X", (unsigned)u);
    };

    const float *segments[] = { _w1, _w2, _b1, _b2 };
    const uint32_t sizes[]  = { NN_W1_SIZE, NN_W2_SIZE, NN_B1_SIZE, NN_B2_SIZE };

    uint32_t printed = 0;
    for (int s = 0; s < 4; s++) {
        for (uint32_t i = 0; i < sizes[s]; i++) {
            printHex(segments[s][i]);
            printed++;
            if ((printed & 7) == 0) Serial.println();
            else                    Serial.print(' ');
        }
    }
    if ((printed & 7) != 0) Serial.println();

    Serial.println("[NN_DUMP END]");
    return (size_t)printed;
}

// saveToLittleFS()
// Writes W1|W2|b1|b2 as a raw binary blob.
// W1 alone is 32 KB — NVS has a 4 KB/key limit so LittleFS is used.
//
// platformio.ini must include:
//   board_build.filesystem = littlefs
//
// Usage:  nn.saveToLittleFS();   // triggered by serial command 's'
bool NeuralNetwork::saveToLittleFS(const char *path)
{
    if (!LittleFS.begin(true)) {
        Serial.println("[NN] saveToLittleFS: LittleFS mount failed");
        return false;
    }

    File f = LittleFS.open(path, "w");
    if (!f) {
        Serial.printf("[NN] saveToLittleFS: cannot open %s for writing\n", path);
        return false;
    }

    f.write((const uint8_t *)_w1, sizeof(_w1));
    f.write((const uint8_t *)_w2, sizeof(_w2));
    f.write((const uint8_t *)_b1, sizeof(_b1));
    f.write((const uint8_t *)_b2, sizeof(_b2));
    f.close();

    Serial.printf("[NN] Weights saved to %s  (%u bytes)\n",
                  path, (unsigned)MODEL_BLOB_BYTES);
    return true;
}

// loadFromLittleFS()
// Reads the binary blob written by saveToLittleFS().
// Performs a size check before loading — rejects corrupt/partial files.
//
// Recommended call pattern in setup():
//   if (!nn.loadFromLittleFS()) {
//       nn.init();   // no saved model — start fresh
//       opMode = MODE_TRAIN;
//   } else {
//       opMode = MODE_INFER;   // model ready, go straight to inference
//   }
bool NeuralNetwork::loadFromLittleFS(const char *path)
{
    if (!LittleFS.begin(true)) {
        Serial.println("[NN] loadFromLittleFS: LittleFS mount failed");
        return false;
    }

    if (!LittleFS.exists(path)) {
        Serial.printf("[NN] loadFromLittleFS: %s not found\n", path);
        return false;
    }

    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("[NN] loadFromLittleFS: cannot open %s\n", path);
        return false;
    }

    if (f.size() != MODEL_BLOB_BYTES) {
        Serial.printf("[NN] loadFromLittleFS: size mismatch (%u vs expected %u) — rejecting\n",
                      (unsigned)f.size(), (unsigned)MODEL_BLOB_BYTES);
        f.close();
        return false;
    }

    f.read((uint8_t *)_w1, sizeof(_w1));
    f.read((uint8_t *)_w2, sizeof(_w2));
    f.read((uint8_t *)_b1, sizeof(_b1));
    f.read((uint8_t *)_b2, sizeof(_b2));
    f.close();

    Serial.printf("[NN] Weights loaded from %s  (%u bytes)\n",
                  path, (unsigned)MODEL_BLOB_BYTES);
    return true;
}

float NeuralNetwork::sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// σ'(z) computed from already-activated value a = σ(z): a·(1−a)
float NeuralNetwork::sigmoidDerivative(float activated)
{
    return activated * (1.0f - activated);
}

// Uniform sample from [-limit, +limit]
float NeuralNetwork::xavierRand(float limit)
{
    return ((float)rand() / (float)RAND_MAX) * 2.0f * limit - limit;
}