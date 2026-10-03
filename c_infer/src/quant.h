#ifndef MT_QUANT_H
#define MT_QUANT_H

#include <stdint.h>

/* x[0..n) → 非对称 uint8（zero-point = 128），返回 per-token scale。
 * q = clamp(round(x/scale)+128, 0, 255)，scale = amax(|x|)/127 */
float mt_quant_u8(const float *x, uint8_t *q, int n);

#endif /* MT_QUANT_H */
