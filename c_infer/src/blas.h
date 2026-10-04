#ifndef MT_BLAS_H
#define MT_BLAS_H

#include <stdint.h>
#include <stddef.h>

/* y[M,N] = x[M,K] @ W[N,K]^T + (b? b[n] : 0) */
void mt_linear(float *y, const float *x, const float *W, const float *b, int M, int N, int K);
void mt_linear_nobias(float *y, const float *x, const float *W, int M, int N, int K);

void mt_blas_set_threads(int n);
int  mt_blas_get_threads(void);

/* 统一 Linear 描述符：fp32 权重 或 int8 权重（含 per-row scale/rowsum） */
typedef struct {
  const float   *w32;
  const int8_t  *w8;
  const float   *wscale;
  const int32_t *wrowsum;
  int N, K;          /* out_features, in_features */
} mt_lin_t;

void mt_lin_apply(float *y, const float *x, const mt_lin_t *W, int M);
/* 与「M 次 mt_lin_apply(y+m*N, x+m*K, W, 1)」逐位等价（int8 档）；fp32 档保持逐行 sgemv。 */
void mt_lin_apply_batched(float *y, const float *x, const mt_lin_t *W, int M);
void mt_lin_apply_bias(float *y, const float *x, const mt_lin_t *W, const float *b, int M);

#endif /* MT_BLAS_H */
