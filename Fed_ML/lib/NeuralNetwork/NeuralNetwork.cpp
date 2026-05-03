// NeuralNetwork.cpp
// On-device feedforward NN with online backpropagation.

#include "NeuralNetwork.h"
#include <math.h>
#include <stdlib.h>   

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

// Xavier limit for a layer: sqrt(6 / (fan_in + fan_out))
void NeuralNetwork::init()
{
    srand(42);  
    float lim1 = sqrtf(6.0f / (float)(NN_INPUT_SIZE + NN_HIDDEN_SIZE));
    float lim2 = sqrtf(6.0f / (float)(NN_HIDDEN_SIZE + NN_OUTPUT_SIZE));

    for (int i = 0; i < NN_W1_SIZE; i++) _w1[i] = xavierRand(lim1);
    for (int i = 0; i < NN_W2_SIZE; i++) _w2[i] = xavierRand(lim2);
    for (int i = 0; i < NN_B1_SIZE; i++) _b1[i] = 0.01f;
    for (int i = 0; i < NN_B2_SIZE; i++) _b2[i] = 0.01f;

    Serial.printf("[NN] Weights initialised — W1:%d W2:%d params\n",
                  NN_W1_SIZE, NN_W2_SIZE);
}

// forward()
// z1[j] = Σ_i( w1[i][j] * input[i] ) + b1[j]
// a1[j] = σ(z1[j])
// z2[k] = Σ_j( w2[j][k] * a1[j] ) + b2[k]
// a2[k] = σ(z2[k])
void NeuralNetwork::forward(const float *input, float *output)
{
    //  Hidden layer 
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        float sum = _b1[j];
        // w1 is row-major [input][hidden]
        for (int i = 0; i < NN_INPUT_SIZE; i++) {
            sum += _w1[i * NN_HIDDEN_SIZE + j] * input[i];
        }
        _z1[j] = sum;
        _a1[j] = sigmoid(sum);
    }

    //  Output layer 
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

//  backward() — single-sample MSE gradient descent 
// dL/dw = δ · a_prev
// δ_out[k] = (a2[k] - target[k]) · σ'(a2[k])
// δ_hid[j] = (Σ_k δ_out[k] · w2[j][k]) · σ'(a1[j])
float NeuralNetwork::backward(const float *target, float lr)
{
    //  Output layer deltas 
    float delta2[NN_OUTPUT_SIZE];
    float loss = 0.0f;
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        float err   = _a2[k] - target[k];
        loss       += err * err;
        delta2[k]   = err * sigmoidDerivative(_a2[k]);
    }
    loss /= (float)NN_OUTPUT_SIZE;  

    //  Hidden layer deltas 
    float delta1[NN_HIDDEN_SIZE];
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        float grad = 0.0f;
        for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
            grad += delta2[k] * _w2[j * NN_OUTPUT_SIZE + k];
        }
        delta1[j] = grad * sigmoidDerivative(_a1[j]);
    }

    //  Update W2, b2 
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
            _w2[j * NN_OUTPUT_SIZE + k] -= lr * delta2[k] * _a1[j];
        }
    }
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        _b2[k] -= lr * delta2[k];
    }

    //  Update W1, b1 
    // NOTE: The input array is not cached here to save RAM.
    // The caller must pass the same `input` used in forward().
    // We use a separate pointer passed by the train() wrapper
    // in main.cpp, which stores it.  Here we use _a1 derivation:
    // This implementation expects the caller (main.cpp trainStep)
    // to call backward() immediately after forward() with the
    // same feature vector still valid in its buffer.
    // W1 update is done via the public trainStep() in main.cpp
    // which has access to the feature vector.
    // See NeuralNetwork::updateW1() below.

    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        _b1[j] -= lr * delta1[j];
    }

    memcpy(_z1, delta1, NN_HIDDEN_SIZE * sizeof(float));  // reuse _z1 as temp

    return loss;
}

// updateW1() — must be called right after backward() 
// Separated because we must avoid storing the input (500 floats)
// inside the class to stay under the RAM budget.
void NeuralNetwork::updateW1(const float *input, float lr)
{
    // _z1 now holds delta1 (stored there by backward())
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
            _w1[i * NN_HIDDEN_SIZE + j] -= lr * _z1[j] * input[i];
        }
    }
}

//  mse() 
float NeuralNetwork::mse(const float *target) const
{
    float loss = 0.0f;
    for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
        float e = _a2[k] - target[k];
        loss += e * e;
    }
    return loss / (float)NN_OUTPUT_SIZE;
}

//  debugWeights() 
void NeuralNetwork::debugWeights() const
{
    Serial.println("[NN] W2 snapshot:");
    for (int j = 0; j < NN_HIDDEN_SIZE; j++) {
        Serial.printf("  h%02d → ", j);
        for (int k = 0; k < NN_OUTPUT_SIZE; k++) {
            Serial.printf("%8.5f ", _w2[j * NN_OUTPUT_SIZE + k]);
        }
        Serial.println("");
    }
}

// serializeToSerial() — dump all weights/biases as hex-encoded 32-bit words
size_t NeuralNetwork::serializeToSerial() const
{
    const uint32_t total_floats = NN_W1_SIZE + NN_W2_SIZE + NN_B1_SIZE + NN_B2_SIZE;
    Serial.println("[NN_DUMP START]");
    Serial.printf("[NN_DUMP COUNT %u]\n", (unsigned)total_floats);

    auto printHexFloat = [](float v) {
        uint32_t u;
        memcpy(&u, &v, sizeof(u));
        Serial.printf("%08X", (unsigned)u);
    };

    uint32_t printed = 0;
    // W1
    for (uint32_t i = 0; i < NN_W1_SIZE; i++) {
        printHexFloat(_w1[i]);
        printed++;
        if ((printed & 7) == 0) Serial.println(); else Serial.print(' ');
    }
    // W2
    for (uint32_t i = 0; i < NN_W2_SIZE; i++) {
        printHexFloat(_w2[i]);
        printed++;
        if ((printed & 7) == 0) Serial.println(); else Serial.print(' ');
    }
    // b1
    for (uint32_t i = 0; i < NN_B1_SIZE; i++) {
        printHexFloat(_b1[i]);
        printed++;
        if ((printed & 7) == 0) Serial.println(); else Serial.print(' ');
    }
    // b2
    for (uint32_t i = 0; i < NN_B2_SIZE; i++) {
        printHexFloat(_b2[i]);
        printed++;
        if ((printed & 7) == 0) Serial.println(); else Serial.print(' ');
    }

    if ((printed & 7) != 0) Serial.println();
    Serial.println("[NN_DUMP END]");
    return (size_t)printed;
}

//  Private helpers 
float NeuralNetwork::sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float NeuralNetwork::sigmoidDerivative(float activated)
{
    return activated * (1.0f - activated);
}

float NeuralNetwork::xavierRand(float limit)
{
    // Uniform in [-limit, +limit]
    return ((float)rand() / (float)RAND_MAX) * 2.0f * limit - limit;
}
