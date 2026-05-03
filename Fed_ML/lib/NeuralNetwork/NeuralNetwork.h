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

#include <Arduino.h>
#include <string.h>

//  Network dimensions 
#define NN_INPUT_SIZE  500
#define NN_HIDDEN_SIZE 16
#define NN_OUTPUT_SIZE 3

// Flat weight / bias array sizes
#define NN_W1_SIZE (NN_INPUT_SIZE  * NN_HIDDEN_SIZE)  // 8000
#define NN_W2_SIZE (NN_HIDDEN_SIZE * NN_OUTPUT_SIZE)  // 48
#define NN_B1_SIZE  NN_HIDDEN_SIZE                     // 16
#define NN_B2_SIZE  NN_OUTPUT_SIZE                     // 3

class NeuralNetwork {
public:
    NeuralNetwork();
    void init();

    // Forward pass. Stores activations internally.
    // input  : float[NN_INPUT_SIZE]
    // output : float[NN_OUTPUT_SIZE]  (written by this call)
    void forward(const float *input, float *output);

    // Backpropagation + bias update (MSE loss, gradient descent).
    // Stores delta1 in _z1 for subsequent updateW1() call.
    // target : one-hot float[NN_OUTPUT_SIZE]
    // Returns MSE loss for the sample.
    float backward(const float *target, float lr);

    // Update W1 weights. MUST be called immediately after backward(),
    // while the input buffer that was passed to forward() is still valid.
    void updateW1(const float *input, float lr);

    // Compute MSE loss for the last forward pass result.
    float mse(const float *target) const;

    // Print a compact weight summary to Serial.
    void debugWeights() const;
    // Serialize weights to Serial as hex-encoded IEEE754 words.
    // Format:
    //   [NN_DUMP START]\n
    //   [NN_DUMP COUNT <num_floats>]\n
    //   <hex32> <hex32> ...  (space separated, lines for readability)\n
    //   [NN_DUMP END]\n
    size_t serializeToSerial() const;

private:
    float _w1[NN_W1_SIZE];   // L0→L1  row-major [input][hidden]
    float _w2[NN_W2_SIZE];   // L1→L2  row-major [hidden][output]
    float _b1[NN_B1_SIZE];
    float _b2[NN_B2_SIZE];

    float _z1[NN_HIDDEN_SIZE];   // pre-activation hidden
    float _a1[NN_HIDDEN_SIZE];   // post-activation hidden
    float _z2[NN_OUTPUT_SIZE];   // pre-activation output
    float _a2[NN_OUTPUT_SIZE];   // post-activation output (== forward output)

    static float sigmoid(float x);
    static float sigmoidDerivative(float activated);  // σ'(z) = a(1-a)

    static float xavierRand(float limit);
};
