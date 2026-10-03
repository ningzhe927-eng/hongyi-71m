/* 内核微基准：同一 build（fp32 或 int8）下测 lm_head 形状 GEMM 的单次耗时 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "blas.h"
#include "vnni_gemm.h"
#include "ops.h"
#include "weights.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(void) {
  const int N = VOCAB_SIZE, K = D_MODEL;
  float *x = malloc((size_t)K * sizeof(float));
  float *y = malloc((size_t)N * sizeof(float));
  for (int i = 0; i < K; i++) x[i] = 0.01f * (float)((i % 17) - 8);
  mt_blas_set_threads(8);
  /* 🔴 必须显式限线程：容器 nproc=64 但 cgroup 只给 8 核，
   *    OpenMP 默认按 64 起 ⇒ 8× 过订阅，M>1 的 GEMM 会慢 ~10×（假象）。 */
#ifdef _OPENMP
  omp_set_num_threads(8);
#endif

  const int WARM = 20, REP = 200;
#if MT_WEIGHT_QUANT8
  mt_lin_t W = (mt_lin_t) LINIT_tok_embed;
  for (int r = 0; r < WARM; r++) mt_lin_apply(y, x, &W, 1);
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int r = 0; r < REP; r++) mt_lin_apply(y, x, &W, 1);
  clock_gettime(CLOCK_MONOTONIC, &b);
  const char *tag = "int8";
#else
  for (int r = 0; r < WARM; r++) mt_linear_nobias(y, x, g_W_tok_embed, 1, N, K);
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int r = 0; r < REP; r++) mt_linear_nobias(y, x, g_W_tok_embed, 1, N, K);
  clock_gettime(CLOCK_MONOTONIC, &b);
  const char *tag = "fp32";
#endif
  double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
  double per = ms / REP;
  /* 字节数：fp32 = N*K*4；int8 = N*K*1 */
  double bytes = (double)N * K * (MT_WEIGHT_QUANT8 ? 1.0 : 4.0);
  printf("%s  lm_head(%d x %d): %.3f ms/call  %.1f GB/s  (sink=%.3f)\n",
         tag, N, K, per, bytes / (per * 1e-3) / 1e9, y[123]);
  free(x); free(y);

  /* ---- swiglu 的两颗 GEMM + 逐元素 silu：扫 M=1/16/48 ----
   * 判据：M 变大 ⇒ 每字节权重的 MAC 变多 ⇒ GB/s 应**上升**（算术强度上升）。
   *       若 GB/s 反而掉 ⇒ M>1 路径有问题（batch=16 反常的根因候选）。 */
  static mt_lin_t WGU = LINIT_enc0_mlp_wgu;
  static mt_lin_t WD  = LINIT_enc0_mlp_w_down;
  const int N_WGU = 2 * D_FF, K_WGU = D_MODEL;      /* (3242, 608) */
  const int N_WD  = D_MODEL,  K_WD  = D_FF;         /* (608, 1621) */
  float *xg = malloc((size_t)48 * K_WGU * sizeof(float));
  float *yg = malloc((size_t)48 * (N_WGU > N_WD ? N_WGU : N_WD) * sizeof(float));
  float *up = malloc((size_t)48 * D_FF * sizeof(float));
  float *gt = malloc((size_t)48 * D_FF * sizeof(float));
  for (int i = 0; i < 48 * K_WGU; i++) xg[i] = 0.01f * (float)((i % 17) - 8);
  for (int i = 0; i < 48 * D_FF; i++) { gt[i] = 0.02f * (float)((i % 11) - 5); up[i] = 0.03f * (float)((i % 7) - 3); }
  printf("\n-- swiglu 部件（Rep=%d，8 线程）--\n", REP);
  for (int m = 0; m < 3; m++) {
    const int M = (m == 0 ? 1 : (m == 1 ? 16 : 48));
    struct timespec c1, c2; double ms, gb;
#define BENCH(LBL, EXPR, NBYTES)                                            \
    for (int r = 0; r < WARM; r++) { EXPR; }                                \
    clock_gettime(CLOCK_MONOTONIC, &c1);                                    \
    for (int r = 0; r < REP; r++) { EXPR; }                                 \
    clock_gettime(CLOCK_MONOTONIC, &c2);                                    \
    ms = (c2.tv_sec - c1.tv_sec) * 1e3 + (c2.tv_nsec - c1.tv_nsec) / 1e6;   \
    gb = (double)(NBYTES) / (ms / REP * 1e-3) / 1e9;                        \
    printf("  M=%-3d %-14s %8.4f ms/call  %7.1f GB/s\n", M, LBL, ms / REP, gb)

    BENCH("wgu GEMM", mt_lin_apply(yg, xg, &WGU, M), (double)N_WGU * K_WGU);
    BENCH("wd  GEMM", mt_lin_apply(yg, gt, &WD, M), (double)N_WD * K_WD);
    BENCH("silu(逐元素)", mt_silu_mul(gt, up, M * D_FF), (double)M * D_FF * 8);
    BENCH("memcpy(gate)", memcpy(gt, up, (size_t)M * D_FF * sizeof(float)), (double)M * D_FF * 8);
#undef BENCH
  }
  free(xg); free(yg); free(up); free(gt); 
  return 0;
}
