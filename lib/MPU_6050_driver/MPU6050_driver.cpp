// MPU6050_driver.cpp

#include "MPU6050_driver.h"
#include <math.h>

MPU6050Driver::MPU6050Driver()
    : _pitch(0.0f), _roll(0.0f), _firstSample(true)
{}

bool MPU6050Driver::begin(uint8_t sda_pin, uint8_t scl_pin,
                          uint32_t clock_hz)
{
    Wire.begin(sda_pin, scl_pin);
    Wire.setClock(clock_hz);
    uint8_t who = readReg(MPU6050_REG_WHO_AM_I);
    if (who != 0x70) {
        Serial.printf("[MPU6050] WHO_AM_I mismatch: 0x%02X (expected 0x70)\n", who);
        return false;
    }

    writeReg(MPU6050_REG_PWR_MGMT_1, 0x00);
    delay(100);
    writeReg(MPU6050_REG_SMPLRT_DIV, 0x09);
    writeReg(MPU6050_REG_CONFIG, 0x03);
    writeReg(MPU6050_REG_GYRO_CONFIG, 0x00);
    writeReg(MPU6050_REG_ACCEL_CONFIG, 0x00);

    Serial.println("[MPU6050] Initialised OK — 100 Hz, ±2g, ±250 dps");
    return true;
}

bool MPU6050Driver::readRaw(RawIMU &out)
{
    uint8_t buf[14];
    if (!readBurst(MPU6050_REG_ACCEL_XOUT_H, buf, 14)) return false;

    out.ax = (int16_t)((buf[0]  << 8) | buf[1]);
    out.ay = (int16_t)((buf[2]  << 8) | buf[3]);
    out.az = (int16_t)((buf[4]  << 8) | buf[5]);
    out.gx = (int16_t)((buf[8]  << 8) | buf[9]);
    out.gy = (int16_t)((buf[10] << 8) | buf[11]);
    out.gz = (int16_t)((buf[12] << 8) | buf[13]);
    return true;
}

void MPU6050Driver::toCalibratedIMU(const RawIMU &raw,
                                    CalibratedIMU &cal)
{
    cal.ax_g   = raw.ax / ACCEL_SCALE_FACTOR;
    cal.ay_g   = raw.ay / ACCEL_SCALE_FACTOR;
    cal.az_g   = raw.az / ACCEL_SCALE_FACTOR;
    cal.gx_dps = raw.gx / GYRO_SCALE_FACTOR;
    cal.gy_dps = raw.gy / GYRO_SCALE_FACTOR;
    cal.gz_dps = raw.gz / GYRO_SCALE_FACTOR;
}

void MPU6050Driver::updateAngles(const CalibratedIMU &cal,
                                 Angles &angles)
{
    float accel_pitch = atan2f(cal.ay_g,
                               sqrtf(cal.ax_g * cal.ax_g +
                                     cal.az_g * cal.az_g))
                        * (180.0f / (float)M_PI);

    float accel_roll  = atan2f(-cal.ax_g,
                               sqrtf(cal.ay_g * cal.ay_g +
                                     cal.az_g * cal.az_g))
                        * (180.0f / (float)M_PI);

    if (_firstSample) {
        _pitch       = accel_pitch;
        _roll        = accel_roll;
        _firstSample = false;
    } else {
        _pitch = COMP_FILTER_ALPHA * (_pitch + cal.gy_dps * SAMPLE_DT)
               + (1.0f - COMP_FILTER_ALPHA) * accel_pitch;

        _roll  = COMP_FILTER_ALPHA * (_roll  + cal.gx_dps * SAMPLE_DT)
               + (1.0f - COMP_FILTER_ALPHA) * accel_roll;
    }

    angles.pitch = _pitch;
    angles.roll  = _roll;
}

bool MPU6050Driver::sample(CalibratedIMU &cal, Angles &angles)
{
    RawIMU raw;
    if (!readRaw(raw)) return false;

    Serial.printf("[MPU6050][RAW] ax=%6d ay=%6d az=%6d | gx=%6d gy=%6d gz=%6d\n",
                  raw.ax, raw.ay, raw.az, raw.gx, raw.gy, raw.gz);

    toCalibratedIMU(raw, cal);
    updateAngles(cal, angles);

    Serial.printf("[MPU6050][ANG] pitch=%7.3f°  roll=%7.3f°\n",
                  angles.pitch, angles.roll);
    return true;
}

void MPU6050Driver::writeReg(uint8_t reg, uint8_t val)
{
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

uint8_t MPU6050Driver::readReg(uint8_t reg)
{
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)1, (uint8_t)true);
    return Wire.available() ? Wire.read() : 0xFF;
}

bool MPU6050Driver::readBurst(uint8_t reg, uint8_t *buf, uint8_t len)
{
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;

    uint8_t received = Wire.requestFrom((uint8_t)MPU6050_ADDR,
                                        len, (uint8_t)true);
    if (received != len) return false;

    for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
    return true;
}
