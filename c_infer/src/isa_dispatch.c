/* ISA 运行时分派（仅在 MT_ISA=portable 构建下编译；native 构建不参与）。
 *
 * 背景：CMakeLists 原来用 `-march=native`，本机 = Intel Ice Lake ⇒ 发布的二进制
 * 只编进 AVX-512 路径，在无 AVX-512 的 CPU（AMD Zen1/2/3、老 Xeon、部分云 SKU）上 **SIGILL**。
 *
 * 做法：把两个"含 SIMD 的 TU"各编译两遍
 *   · `src/vnni_gemm.c`（int8 GEMM；符号 mt_gemm_q8 / mt_vnni_available）
 *   · `src/ops.c`（逐元素/归一化；10 个对外符号）
 *   一遍 baseline（-march=x86-64-v3）⇒ 后缀 `_scalar`（走文件内标量回退 / 标量 expf）
 *   一遍 AVX-512（+ -mavx512f/vl/bw/dq/vnni）⇒ 后缀 `_avx512`（走 vpdpbusd / 位精确 SIMD expf）
 * 本文件在启动时按 CPU 选路，定义**所有对外符号** ⇒ 调用点零改动。
 *
 * ⚠ 数值口径：两条路都声称与标量**逐位一致**（int32 精确累加 + mt_expf8 复刻 glibc expf），
 *   因此「MT_ISA_FORCE=scalar」与「MT_ISA_FORCE=avx512」的输出应当完全相同 —— 这是验证判据。
 * 测试旋钮：环境变量 `MT_ISA_FORCE=scalar|avx512`（只改选路，不改数值）。
 */
#include "vnni_gemm.h"
#include "ops.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 两个变体的符号声明（由 -DMT_ISA_SUFFIX 改名生成）---------- */
void mt_gemm_q8_scalar(float *y, const float *x, const int8_t *W, const float *wscale,
                       const int32_t *rowsum, int M, int N, int K);
void mt_gemm_q8_avx2(float *y, const float *x, const int8_t *W, const float *wscale,
                     const int32_t *rowsum, int M, int N, int K);
void mt_gemm_q8_avx512(float *y, const float *x, const int8_t *W, const float *wscale,
                       const int32_t *rowsum, int M, int N, int K);
int  mt_vnni_available_scalar(void);
int  mt_vnni_available_avx512(void);

void mt_rmsnorm_scalar(float *, const float *, const float *, int, int, float);
void mt_silu_mul_scalar(float *, const float *, int);
void mt_silu_mul2_scalar(float *, const float *, const float *, int);
void mt_sigmoid2_scalar(float *, int);
void mt_add_inplace_scalar(float *, const float *, size_t);
void mt_softmax_rows_scalar(float *, int, int);
void mt_log_softmax_scalar(float *, int);
void mt_log_softmax_rows_scalar(float *, int, const int *, int);
int  mt_argmax_scalar(const float *, int);
void mt_topk_scalar(const float *, int, int, float *, int *);

void mt_rmsnorm_avx512(float *, const float *, const float *, int, int, float);
void mt_silu_mul_avx512(float *, const float *, int);
void mt_silu_mul2_avx512(float *, const float *, const float *, int);
void mt_sigmoid2_avx512(float *, int);
void mt_add_inplace_avx512(float *, const float *, size_t);
void mt_softmax_rows_avx512(float *, int, int);
void mt_log_softmax_avx512(float *, int);
void mt_log_softmax_rows_avx512(float *, int, const int *, int);
int  mt_argmax_avx512(const float *, int);
void mt_topk_avx512(const float *, int, int, float *, int *);

void mt_rmsnorm_avx2(float *, const float *, const float *, int, int, float);
void mt_silu_mul_avx2(float *, const float *, int);
void mt_silu_mul2_avx2(float *, const float *, const float *, int);
void mt_sigmoid2_avx2(float *, int);
void mt_add_inplace_avx2(float *, const float *, size_t);
void mt_softmax_rows_avx2(float *, int, int);
void mt_log_softmax_avx2(float *, int);
void mt_log_softmax_rows_avx2(float *, int, const int *, int);
int  mt_argmax_avx2(const float *, int);
void mt_topk_avx2(const float *, int, int, float *, int *);

typedef void (*gemm_q8_fn)(float *, const float *, const int8_t *, const float *,
                           const int32_t *, int, int, int);

/* ---------- 选路：2 = avx512vnni, 1 = avx2, 0 = scalar ---------- */
static int g_isa = 0;
static int g_init_done = 0;

static int mt_isa_pick(void) {
  const char *force = getenv("MT_ISA_FORCE");
  if (force && force[0]) {
    if (strcmp(force, "scalar") == 0) return 0;
    if (strcmp(force, "avx2") == 0) return 1;
    if (strcmp(force, "avx512") == 0) return 2;
    fprintf(stderr, "[isa] ⚠ MT_ISA_FORCE=%s 无法识别（应为 scalar|avx2|avx512），按自动检测\n", force);
  }
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512vl") &&
      __builtin_cpu_supports("avx512vnni")) return 2;
  if (__builtin_cpu_supports("avx2")) return 1;
  return 0;
}

static const char *mt_isa_name(int v) {
  return v == 2 ? "avx512vnni" : (v == 1 ? "avx2" : "scalar");
}

static void mt_isa_setup(void) {
  if (g_init_done) return;
  g_isa = mt_isa_pick();
  g_init_done = 1;
  fprintf(stderr, "[isa] int8 GEMM = %s ; ops = %s%s\n",
          mt_isa_name(g_isa), mt_isa_name(g_isa),
          getenv("MT_ISA_FORCE") ? " (MT_ISA_FORCE)" : "");
}

__attribute__((constructor)) static void mt_isa_ctor(void) { mt_isa_setup(); }

#define MT_ISA_READY() do { if (!g_init_done) mt_isa_setup(); } while (0)

static gemm_q8_fn mt_gemm_q8_impl(void) {
  return g_isa == 2 ? mt_gemm_q8_avx512 : (g_isa == 1 ? mt_gemm_q8_avx2 : mt_gemm_q8_scalar);
}

/* ---------- 对外符号（调用点零改动）---------- */
void mt_gemm_q8(float *y, const float *x, const int8_t *W, const float *wscale,
                const int32_t *rowsum, int M, int N, int K) {
  MT_ISA_READY();
  mt_gemm_q8_impl()(y, x, W, wscale, rowsum, M, N, K);
}

int mt_vnni_available(void) { MT_ISA_READY(); return g_isa; }

/* ops：scalar / avx2（expf8/logf8 走 256 位 + gather）/ avx512 三路 */
#define MT_OPS3(fn, args) do { MT_ISA_READY(); \
  if (g_isa == 2) fn##_avx512 args; else if (g_isa == 1) fn##_avx2 args; else fn##_scalar args; } while (0)

void mt_rmsnorm(float *out, const float *x, const float *w, int rows, int dim, float eps) {
  MT_OPS3(mt_rmsnorm, (out, x, w, rows, dim, eps));
}
void mt_silu_mul(float *gate, const float *up, int n) { MT_OPS3(mt_silu_mul, (gate, up, n)); }
void mt_silu_mul2(float *dst, const float *gate, const float *up, int n) {
  MT_OPS3(mt_silu_mul2, (dst, gate, up, n));
}
void mt_sigmoid2(float *x, int n) { MT_OPS3(mt_sigmoid2, (x, n)); }
void mt_add_inplace(float *dst, const float *src, size_t n) { MT_OPS3(mt_add_inplace, (dst, src, n)); }
void mt_softmax_rows(float *x, int rows, int n) { MT_OPS3(mt_softmax_rows, (x, rows, n)); }
void mt_log_softmax(float *x, int n) { MT_OPS3(mt_log_softmax, (x, n)); }
void mt_log_softmax_rows(float *x, int n, const int *ridx, int nrows) {
  MT_OPS3(mt_log_softmax_rows, (x, n, ridx, nrows));
}
void mt_topk(const float *x, int n, int k, float *val, int *idx) {
  MT_OPS3(mt_topk, (x, n, k, val, idx));
}
int mt_argmax(const float *x, int n) {
  MT_ISA_READY();
  return (g_isa == 2 ? mt_argmax_avx512
                     : (g_isa == 1 ? mt_argmax_avx2 : mt_argmax_scalar))(x, n);
}
