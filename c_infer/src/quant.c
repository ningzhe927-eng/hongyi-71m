#include "quant.h"
#include <math.h>

float mt_quant_u8(const float *x, uint8_t *q, int n) {
  float amax = 0.f;
  for (int i = 0; i < n; i++) {
    float v = fabsf(x[i]);
    if (v > amax) amax = v;
  }
  if (amax <= 1e-12f) {
    for (int i = 0; i < n; i++) q[i] = 128;
    return 1e-8f;
  }
  float scale = amax / 127.0f;
  float inv = 1.0f / scale;
  for (int i = 0; i < n; i++) {
    int iv = (int)lrintf(x[i] * inv) + 128;
    if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
    q[i] = (uint8_t)iv;
  }
  return scale;
}
