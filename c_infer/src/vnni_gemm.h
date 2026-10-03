#ifndef MT_VNNI_GEMM_H
#define MT_VNNI_GEMM_H

#include <stdint.h>

/* y[M,N] (fp32) = x[M,K] (fp32) @ W[N,K]^T (int8, per-row scale)
 *   W[n,k] 为对称 int8；scale_w[n] = amax(row)/127；rowsum[n] = Σ_k W[n,k]
 *   x 每行（每个 token）走非对称 uint8（zero-point 128）
 *   y[m,n] = (Σ_k a_q[m,k]·W[n,k] − 128·rowsum[n]) · a_scale[m] · scale_w[n]
 */
void mt_gemm_q8(float *y, const float *x, const int8_t *W, const float *wscale,
                const int32_t *rowsum, int M, int N, int K);

/* 是否启用了 AVX512-VNNI 快路径 */
int mt_vnni_available(void);

#endif /* MT_VNNI_GEMM_H */
