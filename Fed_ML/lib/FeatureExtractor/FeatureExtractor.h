#pragma once
// FeatureExtractor.h
// Accumulates 100 IMU samples (1 s window @ 100 Hz) and
// produces a flat 500-element feature vector:
//   [ pitch×100, roll×100, gx×100, gy×100, gz×100 ]
// Analogous to the MFCC spectrogram pipeline in the paper,
// but adapted for vibration / angle data.

#include <Arduino.h>
#include "MPU6050_driver.h"

#define WINDOW_SAMPLES   100
#define FEATURES_PER_SAMPLE 5
#define FEATURE_VECTOR_SIZE (WINDOW_SAMPLES * FEATURES_PER_SAMPLE)

class FeatureExtractor {
public:
    FeatureExtractor();

    void reset();

    bool pushSample(const CalibratedIMU &cal, const Angles &angles);

    void getFeatureVector(float *out) const;

    bool isReady() const { return _ready; }

    uint16_t sampleCount() const { return _count; }

private:
    float _pitch[WINDOW_SAMPLES];
    float _roll [WINDOW_SAMPLES];
    float _gx   [WINDOW_SAMPLES];
    float _gy   [WINDOW_SAMPLES];
    float _gz   [WINDOW_SAMPLES];

    uint16_t _count;
    bool     _ready;
};
