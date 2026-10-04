#include "vnni_gemm.h"
#include "quant.h"
#include "prof.h"
#include "isa_suffix.h"   /* MT_ISAN()：同一份源码可编译成 _scalar / _avx512 两个 TU */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- 复用的 scratch（避免每 token 每层 malloc） ---- */
static uint8_t *g_aq = NULL;  static size_t g_aq_cap = 0;
static float   *g_as = NULL;  static size_t g_as_cap = 0;

static uint8_t *aq_get(size_t need) {
  if (need > g_aq_cap) {
    free(g_aq);
    g_aq = (uint8_t *)malloc(need);
    g_aq_cap = g_aq ? need : 0;
  }
  return g_aq;
}
static float *as_get(size_t need) {
  if (need > g_as_cap) {
    free(g_as);
    g_as = (float *)malloc(need * sizeof(float));
    g_as_cap = g_as ? need : 0;
  }
  return g_as;
}

#if defined(__AVX512VNNI__) || defined(__AVX512F__)
static inline __m512i ld512(const void *p) { return _mm512_loadu_si512(p); }

static inline int32_t hsum_epi32(__m512i v) {
  __m256i lo = _mm512_castsi512_si256(v);
  __m256i hi = _mm512_extracti64x4_epi64(v, 1);
  __m256i s = _mm256_add_epi32(lo, hi);
  __m128i t = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  t = _mm_hadd_epi32(t, t);
  t = _mm_hadd_epi32(t, t);
  return _mm_cvtsi128_si32(t);
}
#endif

/* int8 GEMM 的"开线程"阈值。量化后单次 GEMM 的 N 只有 512~30008，
 * 沿用 4096 会让解码侧（N=1024~3242）全部串行 ⇒ 可用 `MT_OMP_MIN` 现场扫描，
 * 不必反复重编。默认值由实测定（见 REPORT.md）。 */
static int mt_omp_min(void) {
  static int v = -1;
  if (v < 0) {
    const char *s = getenv("MT_OMP_MIN");
    v = (s && *s) ? atoi(s) : 512;
  }
  return v;
}

/* 单次 GEMM 的线程数=min(最大线程, (N/4)/MT_GEMM_MIN_BLK)。
 * 🔴 为什么不能直接 omp_get_max_threads()：任务数 = (N&~3)/4，而最小 GEMM（wd/out/co，
 *   N=608）只有 **152 个任务**。128 核机器上开 128 线程 ⇒ 24 个线程拿 2 个、104 个拿 1 个，
 *   关键路径 2× 且白唤醒一百多个线程。本机（mx=8）算出来 = min(8,76) = 8 ⇒ 零回归。 */
static int mt_gemm_threads(int N) {
#ifdef _OPENMP
  const char *e = getenv("MT_GEMM_MIN_BLK");
  int minblk = (e && *e) ? atoi(e) : 2;
  if (minblk < 1) minblk = 1;
  int nblk = (N & ~3) >> 2;
  int t = nblk / minblk;
  if (t < 1) t = 1;
  int mx = omp_get_max_threads();
  return t < mx ? t : mx;
#else
  (void)N; return 1;
#endif
}

int MT_ISAN(mt_vnni_available)(void) {
#if defined(__AVX512VNNI__) || defined(__AVX512F__)
  return __builtin_cpu_supports("avx512f") != 0;
#else
  return 0;
#endif
}

/* ================= int8 GEMM（2026-10-01 重写：K 补齐 + 4×n 寄存器分块） =================
 * 🔴 与旧实现**逐位等价**（微基准已用 memcmp 复核：wgu/wd/lm_head × M=1/16/48 全等）：
 *   · int32 累加是**精确整数**（单条点积 ≤ K·127·255 ≈ 2e7 ≪ 2^31，不溢出）⇒ 怎么分块、
 *     怎么重排累加顺序都得到同一个整数；浮点只在最后一步
 *     `(float)corr * a_scale[m] * scale_w[n]`，保持原顺序即可。
 *   · 量化也逐位等价：amax 用 max（与顺序无关）；`lrintf` 与 `_mm512_cvtps_epi32` 同为
 *     「MXCSR 默认最近偶数舍入」；clamp[0,255] ≡ `_mm512_cvtusepi32_epi8`（无符号饱和）。
 * ✅ 相对旧实现的两处改进（微基准实测 **1.7~2.6×**，见 `tests/bench_kernel.c`）：
 *   ① **消掉标量尾巴**：旧版 K%64 的余数走标量 `acc += a[k]*w[k]`（K=608 ⇒ 32 次、
 *      K=1621 ⇒ 21 次），实测占 M=1 单次点积的 ~70%。现在用一条**带 mask 的 dpbusd** 收尾：
 *      激活行补 0 ⇒ 多读到的权重字节乘 0、贡献恒为 0；越界 lane 被 mask ⇒ 不触发缺页。
 *   ② **4×n 寄存器分块**：4 个 n 共用同一份激活向量 ⇒ 激活加载降到 1/4，
 *      且 4 条独立累加链把 vpdpbusd（延迟 ~5 cycle）的流水线填满。
 * ⚠ 还没做的（下一步）：权重侧预打包（32 一组 + scale 交错）、M>1 的 2D tiling
 *    （现在 M>1 时激活仍被重读 N/4 次）、epilogue 与 silu 融合。
 */
#if defined(__AVX512VNNI__) || defined(__AVX512F__)

/* 一行激活 → u8（zero-point 128），并把 [K, KP) 补 0
 * ⚠ 补 0（不是补 128）是尾部 mask 方案的前提：补 0 ⇒ 尾部多算的权重字节贡献恒为 0 */
static void quant_row_pad(const float *x, uint8_t *q, int K, int KP, float *s_out) {
  float amax = 0.f;
  int i = 0;
  __m512 vmax = _mm512_setzero_ps();
  for (; i + 16 <= K; i += 16)
    vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
  float tmp[16];
  _mm512_storeu_ps(tmp, vmax);
  for (int j = 0; j < 16; j++) if (tmp[j] > amax) amax = tmp[j];
  for (; i < K; i++) { float v = fabsf(x[i]); if (v > amax) amax = v; }

  if (amax <= 1e-12f) {
    memset(q, 128, (size_t)K); memset(q + K, 0, (size_t)(KP - K)); *s_out = 1e-8f; return;
  }
  const float scale = amax / 127.0f, inv = 1.0f / scale;
  const __m512 vinv = _mm512_set1_ps(inv);
  const __m512i v128 = _mm512_set1_epi32(128);
  i = 0;
  for (; i + 16 <= K; i += 16) {
    __m512i iv = _mm512_add_epi32(
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i), vinv)), v128);
    _mm_storeu_si128((__m128i *)(q + i), _mm512_cvtusepi32_epi8(iv));  /* 无符号饱和 ≡ clamp[0,255] */
  }
  for (; i < K; i++) {
    int iv = (int)lrintf(x[i] * inv) + 128;
    if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
    q[i] = (uint8_t)iv;
  }
  memset(q + K, 0, (size_t)(KP - K));
  *s_out = scale;
}

/* 单个 n：K 主部走 VNNI，尾巴用 mask dpbusd（a 已补到 KP ⇒ 加载安全） */
static inline int32_t dot_pad(const uint8_t *a, const int8_t *w, int K) {
  __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
  __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
  int k = 0;
  for (; k + 256 <= K; k += 256) {
    a0 = _mm512_dpbusd_epi32(a0, ld512(a + k),       ld512(w + k));
    a1 = _mm512_dpbusd_epi32(a1, ld512(a + k + 64),  ld512(w + k + 64));
    a2 = _mm512_dpbusd_epi32(a2, ld512(a + k + 128), ld512(w + k + 128));
    a3 = _mm512_dpbusd_epi32(a3, ld512(a + k + 192), ld512(w + k + 192));
  }
  for (; k + 64 <= K; k += 64)
    a0 = _mm512_dpbusd_epi32(a0, ld512(a + k), ld512(w + k));
  __m512i s = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
  const int tail = K & 63;
  if (tail) {
    const __mmask16 mk = (__mmask16)((1u << ((tail + 3) >> 2)) - 1);  /* 只放开含有效字节的 lane */
    __m512i va = _mm512_loadu_si512(a + k);
    __m512i vw = _mm512_maskz_loadu_epi32(mk, (const void *)(w + k)); /* mask 掉的 lane 不会缺页 */
    s = _mm512_mask_dpbusd_epi32(s, mk, va, vw);
  }
  return hsum_epi32(s);
}

/* 4 个连续的 n 共用同一份激活向量：1 次激活加载 + 4 次权重加载 ⇒ 256 MAC / 5 load，
 * 且 4 条独立累加链把 vpdpbusd（延迟 ~5 cycle）的流水线填满。
 * ⚠ 试过更激进的两种，**都更慢**（实测，别再试）：
 *   · 按 M 做 4×4 二维分块：会打散"4 行权重常驻 L1"，权重被迫从 L2 重读 M/4 遍
 *     ⇒ M=48 wgu 0.148 → 0.202 ms。
 *   · K 再切 4 段（16 条链）：链数翻倍但每轮要多取 4 倍权重向量
 *     ⇒ lm_head 185 → 156 GB/s、wd M=48 慢 16%，只有 wgu 好 ~4%。
 *   ⇒ 现在的瓶颈不在累加链数，而在**每轮要取的权重字节数**（已接近 L2 带宽）。 */
static inline void dot4_pad(const uint8_t *a, const int8_t *w0, const int8_t *w1,
                            const int8_t *w2, const int8_t *w3, int K, int32_t *out) {
  __m512i A0 = _mm512_setzero_si512(), A1 = _mm512_setzero_si512();
  __m512i A2 = _mm512_setzero_si512(), A3 = _mm512_setzero_si512();
  const int8_t *wp[4] = {w0, w1, w2, w3};
  int k = 0;
  for (; k + 64 <= K; k += 64) {
    __m512i va = ld512(a + k);
    A0 = _mm512_dpbusd_epi32(A0, va, ld512(wp[0] + k));
    A1 = _mm512_dpbusd_epi32(A1, va, ld512(wp[1] + k));
    A2 = _mm512_dpbusd_epi32(A2, va, ld512(wp[2] + k));
    A3 = _mm512_dpbusd_epi32(A3, va, ld512(wp[3] + k));
  }
  const int tail = K & 63;
  if (tail) {
    const __mmask16 mk = (__mmask16)((1u << ((tail + 3) >> 2)) - 1);
    __m512i va = _mm512_loadu_si512(a + k);
    A0 = _mm512_mask_dpbusd_epi32(A0, mk, va, _mm512_maskz_loadu_epi32(mk, (const void *)(wp[0] + k)));
    A1 = _mm512_mask_dpbusd_epi32(A1, mk, va, _mm512_maskz_loadu_epi32(mk, (const void *)(wp[1] + k)));
    A2 = _mm512_mask_dpbusd_epi32(A2, mk, va, _mm512_maskz_loadu_epi32(mk, (const void *)(wp[2] + k)));
    A3 = _mm512_mask_dpbusd_epi32(A3, mk, va, _mm512_maskz_loadu_epi32(mk, (const void *)(wp[3] + k)));
  }
  out[0] = hsum_epi32(A0); out[1] = hsum_epi32(A1);
  out[2] = hsum_epi32(A2); out[3] = hsum_epi32(A3);
}

void MT_ISAN(mt_gemm_q8)(float *y, const float *x, const int8_t *W, const float *wscale,
                const int32_t *rowsum, int M, int N, int K) {
  const int KP = (K + 63) & ~63;
  uint8_t *aq = aq_get((size_t)M * KP);
  float *as = as_get((size_t)M);
  if (!aq || !as) return;
  { MT_PROF_B0();
    for (int m = 0; m < M; m++) quant_row_pad(x + (size_t)m * K, aq + (size_t)m * KP, K, KP, &as[m]);
    MT_PROF_E0(PROF_QUANT); }

  const int ngrp = N & ~3;
  const int omin = mt_omp_min();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N >= omin) num_threads(mt_gemm_threads(N))
#endif
  for (int n0 = 0; n0 < ngrp; n0 += 4) {
    const int8_t *w0 = W + (size_t)n0 * K;
    const int32_t r[4] = {rowsum[n0], rowsum[n0 + 1], rowsum[n0 + 2], rowsum[n0 + 3]};
    const float s[4] = {wscale[n0], wscale[n0 + 1], wscale[n0 + 2], wscale[n0 + 3]};
    for (int m = 0; m < M; m++) {                   /* n 分组 + m 内层：4 行权重常驻 L1 */
                   /* m 的尾巴（M<4 或 M%4）走 1×4 */
      int32_t c[4];
      dot4_pad(aq + (size_t)m * KP, w0, w0 + K, w0 + 2 * K, w0 + 3 * K, K, c);
      float *yr = y + (size_t)m * N + n0;
      for (int j = 0; j < 4; j++) yr[j] = (float)(c[j] - 128 * r[j]) * as[m] * s[j];
    }
  }
  for (int n = ngrp; n < N; n++) {          /* N 不是 4 的倍数的尾部（如 D=608 也不是 4 的倍数时）*/
    const int8_t *wr = W + (size_t)n * K;
    const int32_t rs = rowsum[n];
    const float ws = wscale[n];
    for (int m = 0; m < M; m++)
      y[(size_t)m * N + n] = (float)(dot_pad(aq + (size_t)m * KP, wr, K) - 128 * rs) * as[m] * ws;
  }
}
#elif defined(MT_IMPL_AVX2)

/* ================= AVX2 变体（无 AVX-512 的机器；由 -DMT_IMPL_AVX2=1 选择）=================
 * 🔴 **不能用 `_mm256_maddubs_epi16(a_u8, w_s8)`**：本方案的激活是 u8（zero-point 128，
 *    a ∈ [1,255]）、权重是 s8（∈ [-127,127]）⇒ 单个积最大 32385，**成对和最大 ~64770 会溢出 i16**
 *    （maddubs 会饱和到 32767 ⇒ 结果错，且是静默错）。
 *    这里走 `cvtepu8_epi16` / `cvtepi8_epi16` → `madd_epi16`（i16×i16 → i32，**精确无饱和**），
 *    每轮 32 字节 = 32 MAC；尾巴走标量。
 * ✅ 逐位等价：int32 累加是精确整数、浮点只在最后一步 epilogue（与 AVX-512 分支**逐字相同**）
 *    ⇒ 分块/累加顺序无关 ⇒ 与 AVX-512 / 标量路径**逐位一致**（已用 1000 句 diff 复核）。
 */
static void quant_row_avx2(const float *x, uint8_t *q, int K, int KP, float *s_out) {
  /* ⚠ 与标量版逐字相同（量化的舍入/饱和口径不能动） */
  float amax = 0.f;
  for (int i = 0; i < K; i++) { float v = fabsf(x[i]); if (v > amax) amax = v; }
  if (amax <= 1e-12f) {
    memset(q, 128, (size_t)K); memset(q + K, 0, (size_t)(KP - K)); *s_out = 1e-8f; return;
  }
  const float scale = amax / 127.0f, inv = 1.0f / scale;
  for (int i = 0; i < K; i++) {
    int iv = (int)lrintf(x[i] * inv) + 128;
    if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
    q[i] = (uint8_t)iv;
  }
  memset(q + K, 0, (size_t)(KP - K));
  *s_out = scale;
}

static inline int32_t hsum256_epi32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

/* 32 字节 a × s8 w 的点积（只读 [0,K)，不依赖 KP 填充；尾巴标量 ⇒ 绝不越界） */
static inline int32_t dot_pad_avx2(const uint8_t *a, const int8_t *w, int K) {
  __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    __m256i a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + k)));
    __m256i a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + k + 16)));
    __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + k)));
    __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + k + 16)));
    c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, w0));
    c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1, w1));
  }
  int32_t s = hsum256_epi32(_mm256_add_epi32(c0, c1));
  for (; k < K; k++) s += (int32_t)a[k] * (int32_t)w[k];
  return s;
}

/* 4 个连续 n 共用同一份激活（与 AVX-512 的 dot4_pad 同思路：激活加载降到 1/4） */
static inline void dot4_pad_avx2(const uint8_t *a, const int8_t *w0, const int8_t *w1,
                                 const int8_t *w2, const int8_t *w3, int K, int32_t *out) {
  const int8_t *wp[4] = {w0, w1, w2, w3};
  __m256i C[4][2];
  for (int j = 0; j < 4; j++) { C[j][0] = _mm256_setzero_si256(); C[j][1] = _mm256_setzero_si256(); }
  int k = 0;
  for (; k + 32 <= K; k += 32) {
    __m256i a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + k)));
    __m256i a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + k + 16)));
    for (int j = 0; j < 4; j++) {
      __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(wp[j] + k)));
      __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(wp[j] + k + 16)));
      C[j][0] = _mm256_add_epi32(C[j][0], _mm256_madd_epi16(a0, b0));
      C[j][1] = _mm256_add_epi32(C[j][1], _mm256_madd_epi16(a1, b1));
    }
  }
  for (int j = 0; j < 4; j++) out[j] = hsum256_epi32(_mm256_add_epi32(C[j][0], C[j][1]));
  for (; k < K; k++) for (int j = 0; j < 4; j++) out[j] += (int32_t)a[k] * (int32_t)wp[j][k];
}

void MT_ISAN(mt_gemm_q8)(float *y, const float *x, const int8_t *W, const float *wscale,
                const int32_t *rowsum, int M, int N, int K) {
  const int KP = (K + 63) & ~63;
  uint8_t *aq = aq_get((size_t)M * KP);
  float *as = as_get((size_t)M);
  if (!aq || !as) return;
  { MT_PROF_B0();
    for (int m = 0; m < M; m++) quant_row_avx2(x + (size_t)m * K, aq + (size_t)m * KP, K, KP, &as[m]);
    MT_PROF_E0(PROF_QUANT); }

  const int ngrp = N & ~3;
  const int omin = mt_omp_min();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N >= omin) num_threads(mt_gemm_threads(N))
#endif
  for (int n0 = 0; n0 < ngrp; n0 += 4) {
    const int8_t *w0 = W + (size_t)n0 * K;
    const int32_t r[4] = {rowsum[n0], rowsum[n0 + 1], rowsum[n0 + 2], rowsum[n0 + 3]};
    const float s[4] = {wscale[n0], wscale[n0 + 1], wscale[n0 + 2], wscale[n0 + 3]};
    for (int m = 0; m < M; m++) {
      int32_t c[4];
      dot4_pad_avx2(aq + (size_t)m * KP, w0, w0 + K, w0 + 2 * K, w0 + 3 * K, K, c);
      float *yr = y + (size_t)m * N + n0;
      for (int j = 0; j < 4; j++) yr[j] = (float)(c[j] - 128 * r[j]) * as[m] * s[j];
    }
  }
  for (int n = ngrp; n < N; n++) {
    const int8_t *wr = W + (size_t)n * K;
    const int32_t rs = rowsum[n];
    const float ws = wscale[n];
    for (int m = 0; m < M; m++)
      y[(size_t)m * N + n] = (float)(dot_pad_avx2(aq + (size_t)m * KP, wr, K) - 128 * rs) * as[m] * ws;
  }
}

#else
/* 非 AVX512 环境的朴素回退（数值口径与上面完全一致） */
static inline int32_t dot_i8(const uint8_t *a, const int8_t *w, int K) {
  int32_t acc = 0;
  for (int k = 0; k < K; k++) acc += (int32_t)a[k] * (int32_t)w[k];
  return acc;
}
void MT_ISAN(mt_gemm_q8)(float *y, const float *x, const int8_t *W, const float *wscale,
                const int32_t *rowsum, int M, int N, int K) {
  uint8_t *aq = aq_get((size_t)M * K);
  float *as = as_get((size_t)M);
  if (!aq || !as) return;
  for (int m = 0; m < M; m++) as[m] = mt_quant_u8(x + (size_t)m * K, aq + (size_t)m * K, K);
  const int omin = mt_omp_min();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N >= omin) num_threads(mt_gemm_threads(N))
#endif
  for (int n = 0; n < N; n++) {
    const int8_t *wr = W + (size_t)n * K;
    const float ws = wscale[n];
    const int32_t rs = rowsum[n];
    for (int m = 0; m < M; m++)
      y[(size_t)m * N + n] = (float)(dot_i8(aq + (size_t)m * K, wr, K) - 128 * rs) * as[m] * ws;
  }
}
#endif
