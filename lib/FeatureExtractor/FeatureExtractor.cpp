// FeatureExtractor.cpp

#include "FeatureExtractor.h"
#include <string.h>

FeatureExtractor::FeatureExtractor()
    : _count(0), _ready(false)
{}

void FeatureExtractor::reset()
{
    _count = 0;
    _ready = false;
}

bool FeatureExtractor::pushSample(const CalibratedIMU &cal,
                                  const Angles &angles)
{
    if (_ready) {
        return true;
    }

    _pitch[_count] = angles.pitch;
    _roll [_count] = angles.roll;
    _gx   [_count] = cal.gx_dps;
    _gy   [_count] = cal.gy_dps;
    _gz   [_count] = cal.gz_dps;

    _count++;

    if (_count >= WINDOW_SAMPLES) {
        _ready = true;
        Serial.printf("[FEX] Window full: %u samples captured\n", _count);
        return true;
    }
    return false;
}

void FeatureExtractor::getFeatureVector(float *out) const
{
    memcpy(out + 0 * WINDOW_SAMPLES, _pitch, WINDOW_SAMPLES * sizeof(float));
    memcpy(out + 1 * WINDOW_SAMPLES, _roll,  WINDOW_SAMPLES * sizeof(float));
    memcpy(out + 2 * WINDOW_SAMPLES, _gx,    WINDOW_SAMPLES * sizeof(float));
    memcpy(out + 3 * WINDOW_SAMPLES, _gy,    WINDOW_SAMPLES * sizeof(float));
    memcpy(out + 4 * WINDOW_SAMPLES, _gz,    WINDOW_SAMPLES * sizeof(float));
}
