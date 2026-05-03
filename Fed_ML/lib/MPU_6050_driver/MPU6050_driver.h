#pragma once
// MPU6050_driver.h
// Lightweight I2C driver for the MPU-6050 IMU.
// Provides raw accelerometer / gyroscope readings and a
// complementary-filter angle estimator (pitch, roll).

#include <Arduino.h>
#include <Wire.h>

#ifndef MPU6050_ADDR
#define MPU6050_ADDR 0x68
#endif

#define MPU6050_REG_SMPLRT_DIV   0x19
#define MPU6050_REG_CONFIG       0x1A
#define MPU6050_REG_GYRO_CONFIG  0x1B
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_ACCEL_XOUT_H 0x3B
#define MPU6050_REG_TEMP_OUT_H   0x41
#define MPU6050_REG_GYRO_XOUT_H  0x43
#define MPU6050_REG_PWR_MGMT_1   0x6B
#define MPU6050_REG_WHO_AM_I     0x75

// Accel ±2 g   → 16384 LSB/g
// Gyro  ±250°/s → 131.0 LSB/(°/s)
#define ACCEL_SCALE_FACTOR 16384.0f
#define GYRO_SCALE_FACTOR  131.0f

// α=0.98 → trusts gyro 98 % for fast motion,
//           accel  2 % corrects slow drift
#define COMP_FILTER_ALPHA 0.98f
#define SAMPLE_DT 0.01f  


struct RawIMU {
    int16_t ax, ay, az;   
    int16_t gx, gy, gz;   
};

struct CalibratedIMU {
    float ax_g, ay_g, az_g;        
    float gx_dps, gy_dps, gz_dps;  
};

struct Angles {
    float pitch;  
    float roll;   
};

class MPU6050Driver {
public:
    MPU6050Driver();
    bool begin(uint8_t sda_pin, uint8_t scl_pin,
               uint32_t clock_hz = 400000UL);
    bool readRaw(RawIMU &out);
    void toCalibratedIMU(const RawIMU &raw, CalibratedIMU &cal);
    void updateAngles(const CalibratedIMU &cal, Angles &angles);
    bool sample(CalibratedIMU &cal, Angles &angles);

private:
    void     writeReg(uint8_t reg, uint8_t val);
    uint8_t  readReg(uint8_t reg);
    bool     readBurst(uint8_t reg, uint8_t *buf, uint8_t len);

    float    _pitch;
    float    _roll;
    bool     _firstSample;
};
