#include "blas.h"
#include "vnni_gemm.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stddef.h>

#if !defined(MT_QUANT8)
/* ---------------- fp32 档：走 OpenBLAS ---------------- */
#include <cblas.h>
#include <stdlib.h>

extern void openblas_set_num_threads(int);

/* ⚠ 实测（2026-10-01）：把 OpenBLAS 设为单线程、自己用 OpenMP 按 N 分块，
 *   总时长从 24.8s 退化到 43.3s ⇒ **OpenBLAS 自带的线程化更好**，别自己分块。 */
/* ⚠ 实测（2026-10-01）：试过「OpenBLAS 单线程 + 自己 OpenMP 按 N/M 分块」，
 *   profiler 总时长 25.2s → 34.4s（更慢）⇒ **用 OpenBLAS 自带的线程化**，别自己分块。 */
void mt_blas_set_threads(int n) { if (n > 0) openblas_set_num_threads(n); }

int mt_blas_get_threads(void) {
  extern int openblas_get_num_threads(void);
  return openblas_get_num_threads();
}

void mt_linear_nobias(float *y, const float *x, const float *W, int M, int N, int K) {
  if (M == 1) {
    /* 🔴 M=1 走 sgemv（y = W @ x）：实测比 M=1 的 sgemm 快 2~3×，是本次最大的一笔优化。
     *   ⚠ 试过 (ColMajor, Trans) 的等价写法：无改善（25.3s vs 24.8s），保持 RowMajor+NoTrans。 */
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, W, K, x, 1, 0.0f, y, 1);
    return;
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              M, N, K, 1.0f, x, K, W, K, 0.0f, y, N);
}

void mt_linear(float *y, const float *x, const float *W, const float *b,
               int M, int N, int K) {
  if (b == NULL) { mt_linear_nobias(y, x, W, M, N, K); return; }
  if (M == 1) {                 /* M=1：sgemv 再逐元素加 bias */
    mt_linear_nobias(y, x, W, 1, N, K);
    for (int n = 0; n < N; n++) y[n] += b[n];
    return;
  }
  /* 复刻 PyTorch addmm：C 先广播 bias，再用 beta=1 累积 */
  for (int m = 0; m < M; m++) {
    float *yr = y + (size_t)m * N;
    for (int n = 0; n < N; n++) yr[n] = b[n];
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              M, N, K, 1.0f, x, K, W, K, 1.0f, y, N);
}

#else
/* ---------------- int8 档：不链接 OpenBLAS ----------------
 * 🔴 教训（2026-10-01）：量化路径从不调用 BLAS，但只要链了 libopenblas，
 *    其构造器就会按“逻辑 CPU 数”（本机 64，无视 8 核 cgroup 配额）起满线程池，
 *    每个进程白烧 ~1.4s 内核 CPU（≈150ms 墙钟）——曾是“int8 每进程反而慢 1.8×”的唯一原因。 */
#include <stdlib.h>

void mt_blas_set_threads(int n) { (void)n; }
int  mt_blas_get_threads(void) { return 0; }

/* 量化构建里不应走到 fp32 权重路径；保留桩以满足链接 */
void mt_linear_nobias(float *y, const float *x, const float *W, int M, int N, int K) {
  (void)y; (void)x; (void)W; (void)M; (void)N; (void)K; abort();
}
void mt_linear(float *y, const float *x, const float *W, const float *b,
               int M, int N, int K) {
  (void)y; (void)x; (void)W; (void)b; (void)M; (void)N; (void)K; abort();
}
#endif

void mt_lin_apply(float *y, const float *x, const mt_lin_t *W, int M) {
  if (W->w32) mt_linear_nobias(y, x, W->w32, M, W->N, W->K);
  else        mt_gemm_q8(y, x, W->w8, W->wscale, W->wrowsum, M, W->N, W->K);
}

void mt_lin_apply_batched(float *y, const float *x, const mt_lin_t *W, int M) {
  if (W->w32) {  /* fp32：sgemv(M=1) 与 sgemm(M>1) 累加序不同，不保证逐位一致 ⇒ 保持逐行 */
    for (int m = 0; m < M; m++)
      mt_linear_nobias(y + (size_t)m * W->N, x + (size_t)m * W->K, W->w32, 1, W->N, W->K);
    return;
  }
  /* 🔴 逐位等价论证（beam k×batch b 与 batch1 逐位 diff=0 门槛依赖它）：
   * ① int32 累加精确（单点积 ≤ K·127·255 ≪ 2^31）⇒ 结果与 M、分块、OMP 线程划分无关；
   * ② 浮点只在 epilogue (float)corr * as[m] * wscale[n]，每个 (m,n) 独立、操作数与
   *    运算顺序同 M=1 完全一致；③ 激活量化按行独立（每行单独 amax/scale）。
   * ⇒ 一次 M=R 与 R 次 M=1 逐位相同（vnni_gemm.c 顶部注释，memcmp 已验证 M=1/16/48）。 */
  mt_gemm_q8(y, x, W->w8, W->wscale, W->wrowsum, M, W->N, W->K);
}

void mt_lin_apply_bias(float *y, const float *x, const mt_lin_t *W, const float *b, int M) {
  if (W->w32) { mt_linear(y, x, W->w32, b, M, W->N, W->K); return; }  /* 保持已对齐的 fp32 行为 */
  mt_lin_apply(y, x, W, M);
  if (b)
    for (int m = 0; m < M; m++) {
      float *r = y + (size_t)m * W->N;
      for (int n = 0; n < W->N; n++) r[n] += b[n];
    }
}
