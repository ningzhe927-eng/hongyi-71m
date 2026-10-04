#include "ops.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

/* ---- ISA 变体改名（MT_ISA=portable 时由 CMake 传 -DMT_ISA_SUFFIX）----
 * ⚠ 必须排在 `#include "ops.h"` **之后**：头里的声明保持**对外名**（由 isa_dispatch.c 提供）。
 * 本文件的定义改名成 mt_xxx_scalar / mt_xxx_avx512，供同一份源码编译两遍。 */
#include "isa_suffix.h"
#define mt_rmsnorm          MT_ISAN(mt_rmsnorm)
#define mt_silu_mul         MT_ISAN(mt_silu_mul)
#define mt_silu_mul2        MT_ISAN(mt_silu_mul2)
#define mt_sigmoid2         MT_ISAN(mt_sigmoid2)
#define mt_add_inplace      MT_ISAN(mt_add_inplace)
#define mt_softmax_rows     MT_ISAN(mt_softmax_rows)
#define mt_log_softmax      MT_ISAN(mt_log_softmax)
#define mt_log_softmax_rows MT_ISAN(mt_log_softmax_rows)
#define mt_argmax           MT_ISAN(mt_argmax)
#define mt_topk             MT_ISAN(mt_topk)

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#define MT_EXPF8 1
#define MT_EXPF8_512 1          /* 512 位 double 多项式 + vpermi2q 查表 */
#elif defined(MT_EXPF8_AVX2)
#include <immintrin.h>
#define MT_EXPF8 1
/* 256 位 double 多项式 + i32gather 查表（见下方 AVX2 分支） */
#endif

#ifdef MT_EXPF8
/* ================= 位精确的 SIMD expf（复刻 glibc 2.35 __expf） =================
 * 🔴 为什么能位精确：glibc 的 expf 不是"某种近似"，它是一条**确定的算法**
 *    （`sysdeps/ieee754/flt-32/e_expf.c`）：
 *      z = (32/ln2)·x ；kd = z + 1.5·2^52（⇒ 取整，ties-to-even）；r = z − kd
 *      s = asdouble(TAB[k & 31] + (k << 47))            // = 2^(k/32)
 *      y = ((C0·r + C1)·r² + (C2·r + 1))·s ；return (float)y
 *    全部是 **double** 运算（IEEE 精确可复现）⇒ 用 8 lane 的 `_mm512_*_pd`
 *    按同样顺序算，再 `_mm512_cvtpd_ps` 回来，就与标量版**逐位相同**。
 * ✅ 已用 1000 万随机值（[-88,88]/[-20,20]/[-1,1]）+ 特殊值
 *    （±0/±inf/nan/±1e30/次正规）与 `expf` 逐位比对 **全部相同**；实测 **5.7×** 快。
 * ⚠ 三个别踩的点（都实测过）：
 *    ① 多项式用 **mul + add**，不要用 `_mm512_fmadd_pd`（FMA 少一次舍入，会差 1 ULP）；
 *    ② 查表**不要用 gather** —— 512 位 `vpgatherqq` 在这里很慢（只有 1.77×，
 *       而两次 `vpermi2q` + blend 有 5.71×）；
 *    ③ `_mm256_mask_blend_ps(k,a,b)` 在 **gcc 上是 k?b:a**（与 Intel 文档相反！），
 *       写反了会让越界回退失效。
 * 🔴 |x| >= 88（以及 nan/inf）走标量 `expf` 回退：glibc 在那里有溢出/下溢/errno
 *    的特殊处理，逐个调最稳妥（这些 lane 极少出现，不影响速度）。 */
static const uint64_t MT_EXPF_TAB[32] = {
0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};
#define MT_EXPF_INVLN2N (0x1.71547652b82fep+0 * 32.0)
#define MT_EXPF_SHIFT   0x1.8p+52
#define MT_EXPF_C0 (0x1.c6af84b912394p-5 / 32768.0)   /* poly_scaled[0] = p0/N^3 */
#define MT_EXPF_C1 (0x1.ebfce50fac4f3p-3 / 1024.0)    /* poly_scaled[1] = p1/N^2 */
#define MT_EXPF_C2 (0x1.62e42ff0c52d6p-1 / 32.0)      /* poly_scaled[2] = p2/N   */

/* ---- logf 的查表 / 常量（**两个变体共享**，必须放在 ISA 分叉之前）---- */
static const double MT_LOGF_INVC[16] = {
0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0, 0x1.3c995b0b80385p+0,
0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0, 0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0,
0x1.0953f419900a7p+0, 0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1, 0x1.767dcf5534862p-1,
};
static const double MT_LOGF_LOGC[16] = {
-0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2, -0x1.b31d8a68224e9p-3,
-0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,  -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4,
-0x1.252f438e10c1ep-5, 0x0p+0,               0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,  0x1.1058bc8a07ee1p-2,  0x1.4043057b6ee09p-2,
};
#define MT_LOGF_LN2 0x1.62e42fefa39efp-1
#define MT_LOGF_A0 (-0x1.00ea348b88334p-2)
#define MT_LOGF_A1 ( 0x1.5575b0be00b6ap-2)
#define MT_LOGF_A2 (-0x1.ffffef20a4123p-2)

#ifdef MT_EXPF8_512
static inline __m512d mt_expf_core_pd(__m512d xd) {
  const __m512d vinv = _mm512_set1_pd(MT_EXPF_INVLN2N), vsh = _mm512_set1_pd(MT_EXPF_SHIFT);
  const __m512d vc0 = _mm512_set1_pd(MT_EXPF_C0), vc1 = _mm512_set1_pd(MT_EXPF_C1);
  const __m512d vc2 = _mm512_set1_pd(MT_EXPF_C2), vone = _mm512_set1_pd(1.0);
  __m512d z = _mm512_mul_pd(vinv, xd);
  __m512d kd0 = _mm512_add_pd(z, vsh);            /* ⇒ 取整（ties-to-even）*/
  __m512i ki = _mm512_castpd_si512(kd0);
  __m512d kd = _mm512_sub_pd(kd0, vsh);
  __m512d r = _mm512_sub_pd(z, kd);
  __m512i z0 = _mm512_loadu_si512((const void *)(MT_EXPF_TAB + 0));
  __m512i z1 = _mm512_loadu_si512((const void *)(MT_EXPF_TAB + 8));
  __m512i z2 = _mm512_loadu_si512((const void *)(MT_EXPF_TAB + 16));
  __m512i z3 = _mm512_loadu_si512((const void *)(MT_EXPF_TAB + 24));
  __m512i i15 = _mm512_and_si512(ki, _mm512_set1_epi64(15));
  __m512i p01 = _mm512_permutex2var_epi64(z0, i15, z1);
  __m512i p23 = _mm512_permutex2var_epi64(z2, i15, z3);
  __mmask8 hi = _mm512_test_epi64_mask(ki, _mm512_set1_epi64(16));
  __m512i t = _mm512_add_epi64(_mm512_mask_blend_epi64(hi, p01, p23),
                               _mm512_slli_epi64(ki, 52 - 5));
  __m512d s = _mm512_castsi512_pd(t);
  __m512d z2v = _mm512_add_pd(_mm512_mul_pd(vc0, r), vc1);
  __m512d r2 = _mm512_mul_pd(r, r);
  __m512d y = _mm512_add_pd(_mm512_mul_pd(vc2, r), vone);
  y = _mm512_add_pd(_mm512_mul_pd(z2v, r2), y);
  return _mm512_mul_pd(y, s);
}

/* 8 lane float → 8 lane float
 * 🔴 越界 lane **不能**用"逐个调标量 expf"来兜底：softmax 里被 PAD_BIAS 屏蔽的位置
 *    `p[i]-m ≈ -3.4e38`，每一行都有 ⇒ 回退会把 8 lane 全部拖成标量（实测整段零收益）。
 *    glibc 在这些区间的行为是**固定的常数**，直接在 SIMD 里按同样的阈值贴回去即可：
 *      x >  0x1.62e42ep6f (88.72) ⇒ +inf（__math_oflowf）
 *      x < -0x1.9d1d9ep6f (-103.28) ⇒ 0（__math_may_uflowf；-103.97 以下同样是 0）
 *      NaN ⇒ x + x
 *    于是主路径的适用范围正好是 glibc 的主路径区间 **[-103.28, 88.72]** ⇒ 逐位相同、
 *    且永不回退（errno 我们不管：本工程用 -fno-math-errno）。 */
static inline __m256 mt_expf8(__m256 x) {
  const __mmask8 nan = _mm256_cmp_ps_mask(x, x, _CMP_UNORD_Q);
  const __mmask8 big = _mm256_cmp_ps_mask(x, _mm256_set1_ps(0x1.62e42ep6f), _CMP_GT_OQ);
  const __mmask8 sml = _mm256_cmp_ps_mask(x, _mm256_set1_ps(-0x1.9d1d9ep6f), _CMP_LT_OQ);
  __m256 y = _mm512_cvtpd_ps(mt_expf_core_pd(_mm512_cvtps_pd(x)));
  /* ⚠ gcc 的 _mm256_mask_blend_ps(k,a,b) = k?b:a（与 Intel 文档相反）*/
  if (__builtin_expect((unsigned)(big | sml | nan), 0)) {
    if (big) y = _mm256_mask_blend_ps(big, y, _mm256_set1_ps(INFINITY));
    if (sml) y = _mm256_mask_blend_ps(sml, y, _mm256_setzero_ps());
    if (nan) y = _mm256_mask_blend_ps(nan, y, _mm256_add_ps(x, x));
  }
  return y;
}

/* ================= 位精确的 SIMD logf（复刻 glibc 2.35 __logf） =================
 * 与 expf 同一套路：glibc 的 logf 也是**确定的 double 算法**（`sysdeps/ieee754/flt-32/e_logf.c`）
 *    ix = bits(x)；若 x==1 直接返回 0；次正规/0/负数/inf/nan 单独处理
 *    tmp = ix - 0x3f330000；i = (tmp>>19)&15；k = (int32)tmp>>23；iz = ix & ~(0x1ff<<23)
 *    z = (double)asfloat(iz)；r = z*invc - 1；y0 = logc + (double)k*Ln2
 *    y = ((A0*r² + (A1*r + A2)))*r² + (y0 + r)
 *  ⇒ 全 double、顺序照抄，`_mm512_cvtpd_ps` 回来即逐位相同。表 T_invc/T_logc 各 16 项
 *    （正好两个 zmm）⇒ 一次 `_mm512_permutex2var_pd` 就够（不像 expf 要两次 + blend）。
 * ✅ 已用 1000 万随机值（(0,1]/[1,1e38]/[1e-38,1]/密集近 1/次正规）+ 特殊值逐位比对全同；实测 **35.8×**。
 * 🔴 三个坑：① 多项式仍用 mul+add（别用 fma）；② 越界判据必须是**无符号**比较
 *    （`ix - 0x00800000 >= 0x7f000000`，写成有符号会让次正规/0/负数漏掉回退）；
 *    ③ x==1 要单独贴 0（glibc 在 WANT_ROUNDING 下直接返回 0）。 */
static inline __m512d mt_logf_core_pd(__m256i ix) {
  const __m512d vln2 = _mm512_set1_pd(MT_LOGF_LN2), va0 = _mm512_set1_pd(MT_LOGF_A0);
  const __m512d va1 = _mm512_set1_pd(MT_LOGF_A1), va2 = _mm512_set1_pd(MT_LOGF_A2);
  const __m512d one = _mm512_set1_pd(1.0);
  const __m256i voff = _mm256_set1_epi32(0x3f330000);
  __m256i tmp = _mm256_sub_epi32(ix, voff);
  __m256i ii = _mm256_and_si256(_mm256_srli_epi32(tmp, 23 - 4), _mm256_set1_epi32(15));
  __m256i kk = _mm256_srai_epi32(tmp, 23);                     /* 算术右移（有符号）*/
  __m256i iz = _mm256_sub_epi32(ix, _mm256_and_si256(tmp, _mm256_set1_epi32((int)(0x1ffu << 23))));
  __m512i i64 = _mm512_cvtepi32_epi64(ii);
  __m512d invc = _mm512_permutex2var_pd(_mm512_loadu_pd(MT_LOGF_INVC), i64,
                                        _mm512_loadu_pd(MT_LOGF_INVC + 8));
  __m512d logc = _mm512_permutex2var_pd(_mm512_loadu_pd(MT_LOGF_LOGC), i64,
                                        _mm512_loadu_pd(MT_LOGF_LOGC + 8));
  __m512d z = _mm512_cvtps_pd(_mm256_castsi256_ps(iz));
  __m512d r = _mm512_sub_pd(_mm512_mul_pd(z, invc), one);
  __m512d y0 = _mm512_add_pd(logc, _mm512_mul_pd(_mm512_cvtepi32_pd(kk), vln2));
  __m512d r2 = _mm512_mul_pd(r, r);
  __m512d y = _mm512_add_pd(_mm512_mul_pd(va1, r), va2);
  y = _mm512_add_pd(_mm512_mul_pd(va0, r2), y);
  return _mm512_add_pd(_mm512_mul_pd(y, r2), _mm512_add_pd(y0, r));
}

static inline __m256 mt_logf8(__m256 x) {
  __m256i ix = _mm256_castps_si256(x);
  const __mmask8 m1 = _mm256_cmpeq_epi32_mask(ix, _mm256_set1_epi32(0x3f800000));   /* x == 1 */
  const __mmask8 bad = _mm256_cmp_epu32_mask(_mm256_sub_epi32(ix, _mm256_set1_epi32(0x00800000)),
                                             _mm256_set1_epi32((int)0x7f000000u), _MM_CMPINT_NLT);
  const __mmask8 fast = (__mmask8)~(m1 | bad);
  __m256 y = _mm512_cvtpd_ps(mt_logf_core_pd(ix));
  /* ⚠ gcc 的 _mm256_mask_blend_ps(k,a,b) = k?b:a */
  y = _mm256_mask_blend_ps(m1, y, _mm256_setzero_ps());
  if (__builtin_expect(bad != 0, 0)) {          /* 次正规/0/负数/inf/nan ⇒ 逐个调标量 logf */
    float xa[8], ya[8];
    _mm256_storeu_ps(xa, x);
    for (int i = 0; i < 8; i++) if (!((fast >> i) & 1)) ya[i] = logf(xa[i]);
    y = _mm256_mask_blend_ps(fast, _mm256_loadu_ps(ya), y);
  }
  return y;
}
#else /* !MT_EXPF8_512 ⇒ AVX2 变体 */

/* ================= AVX2：256 位 double 多项式 + i32gather 查表 =================
 * ⚠ AVX2 **没有 `vpermi2q`**（AVX-512 的跨 128 位通道、64 位变址置换）⇒ 查表改走
 *    `_mm256_i32gather_epi64` / `_mm256_i32gather_pd`（每次 4 车道；比 permute 慢但正确）。
 *    索引直接用 `k & 31`（AVX-512 版因为一个 zmm 只装 8 个 qword，要拆两次 permute + 按 bit4 blend；
 *    AVX2 一次 gather 就能横跨全表 32 项 ⇒ 更简单）。
 * ✅ 多项式与 512 版**逐步相同**（mul + add、**无 fma**）⇒ 与标量/AVX-512 **逐位一致**。
 * 🔴 别改成 fma / 别合并成 FMA 形式：会少一次舍入 ⇒ 差 1 ULP ⇒ 穿透 int8 量化边界。 */
static inline __m256d mt_expf_core_4d(__m256d xd) {
  const __m256d vinv = _mm256_set1_pd(MT_EXPF_INVLN2N), vsh = _mm256_set1_pd(MT_EXPF_SHIFT);
  const __m256d vc0 = _mm256_set1_pd(MT_EXPF_C0), vc1 = _mm256_set1_pd(MT_EXPF_C1);
  const __m256d vc2 = _mm256_set1_pd(MT_EXPF_C2), vone = _mm256_set1_pd(1.0);
  __m256d z = _mm256_mul_pd(vinv, xd);
  __m256d kd0 = _mm256_add_pd(z, vsh);                    /* ⇒ 取整（ties-to-even）*/
  __m256i ki = _mm256_castpd_si256(kd0);
  __m256d kd = _mm256_sub_pd(kd0, vsh);
  __m256d r = _mm256_sub_pd(z, kd);
  /* 4 个 qword 的低 dword = k（低 5 位即表索引） */
  __m128i k32 = _mm256_castsi256_si128(
      _mm256_permutevar8x32_epi32(ki, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
  __m128i i31 = _mm_and_si128(k32, _mm_set1_epi32(31));
  __m256i t = _mm256_i32gather_epi64((const long long *)MT_EXPF_TAB, i31, 8);
  t = _mm256_add_epi64(t, _mm256_slli_epi64(ki, 52 - 5));
  __m256d s = _mm256_castsi256_pd(t);
  __m256d z2v = _mm256_add_pd(_mm256_mul_pd(vc0, r), vc1);
  __m256d r2 = _mm256_mul_pd(r, r);
  __m256d y = _mm256_add_pd(_mm256_mul_pd(vc2, r), vone);
  y = _mm256_add_pd(_mm256_mul_pd(z2v, r2), y);
  return _mm256_mul_pd(y, s);
}

/* 8 lane float → 8 lane float（两条 4-double 通道；越界/NaN 用 AVX2 的 cmp+blendv 贴回，
 *  与 AVX-512 版的 mask_blend 语义一致：y ← big?+inf / sml?0 / nan?x+x） */
static inline __m256 mt_expf8(__m256 x) {
  __m256d cl = mt_expf_core_4d(_mm256_cvtps_pd(_mm256_castps256_ps128(x)));
  __m256d ch = mt_expf_core_4d(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)));
  __m256 y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(cl)),
                                  _mm256_cvtpd_ps(ch), 1);
  __m256 nanm = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  __m256 bigm = _mm256_cmp_ps(x, _mm256_set1_ps(0x1.62e42ep6f), _CMP_GT_OQ);
  __m256 smlm = _mm256_cmp_ps(x, _mm256_set1_ps(-0x1.9d1d9ep6f), _CMP_LT_OQ);
  if (__builtin_expect((unsigned)(_mm256_movemask_ps(nanm) | _mm256_movemask_ps(bigm)
                                  | _mm256_movemask_ps(smlm)) != 0, 0)) {
    if (_mm256_movemask_ps(bigm)) y = _mm256_blendv_ps(y, _mm256_set1_ps(INFINITY), bigm);
    if (_mm256_movemask_ps(smlm)) y = _mm256_blendv_ps(y, _mm256_setzero_ps(), smlm);
    if (_mm256_movemask_ps(nanm)) y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), nanm);
  }
  return y;
}

static inline __m256d mt_logf_core_4d(__m128i ix) {
  const __m256d vln2 = _mm256_set1_pd(MT_LOGF_LN2), va0 = _mm256_set1_pd(MT_LOGF_A0);
  const __m256d va1 = _mm256_set1_pd(MT_LOGF_A1), va2 = _mm256_set1_pd(MT_LOGF_A2);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m128i voff = _mm_set1_epi32(0x3f330000);
  __m128i tmp = _mm_sub_epi32(ix, voff);
  __m128i ii = _mm_and_si128(_mm_srli_epi32(tmp, 23 - 4), _mm_set1_epi32(15));
  __m128i kk = _mm_srai_epi32(tmp, 23);                   /* 算术右移（有符号）*/
  __m128i iz = _mm_sub_epi32(ix, _mm_and_si128(tmp, _mm_set1_epi32((int)(0x1ffu << 23))));
  __m256d invc = _mm256_i32gather_pd(MT_LOGF_INVC, ii, 8);
  __m256d logc = _mm256_i32gather_pd(MT_LOGF_LOGC, ii, 8);
  __m256d z = _mm256_cvtps_pd(_mm_castsi128_ps(iz));
  __m256d r = _mm256_sub_pd(_mm256_mul_pd(z, invc), one);
  __m256d y0 = _mm256_add_pd(logc, _mm256_mul_pd(_mm256_cvtepi32_pd(kk), vln2));
  __m256d r2 = _mm256_mul_pd(r, r);
  __m256d y = _mm256_add_pd(_mm256_mul_pd(va1, r), va2);
  y = _mm256_add_pd(_mm256_mul_pd(va0, r2), y);
  return _mm256_add_pd(_mm256_mul_pd(y, r2), _mm256_add_pd(y0, r));
}

static inline __m256 mt_logf8(__m256 x) {
  __m256i ix = _mm256_castps_si256(x);
  __m256i m1 = _mm256_cmpeq_epi32(ix, _mm256_set1_epi32(0x3f800000));      /* x == 1 */
  /* 无符号 `sub >= 0x7f000000`：AVX2 没有无符号比较 ⇒ 用 max_epu32 判等 */
  __m256i bound = _mm256_set1_epi32((int)0x7f000000u);
  __m256i sub = _mm256_sub_epi32(ix, _mm256_set1_epi32(0x00800000));
  __m256i bad = _mm256_cmpeq_epi32(_mm256_max_epu32(sub, bound), bound);
  __m256i fast = _mm256_xor_si256(_mm256_or_si256(m1, bad), _mm256_set1_epi32(-1));
  __m256d cl = mt_logf_core_4d(_mm_castps_si128(_mm256_castps256_ps128(x)));
  __m256d ch = mt_logf_core_4d(_mm_castps_si128(_mm256_extractf128_ps(x, 1)));
  __m256 y = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(cl)),
                                  _mm256_cvtpd_ps(ch), 1);
  y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_castsi256_ps(m1));
  if (__builtin_expect(_mm256_movemask_ps(_mm256_castsi256_ps(bad)) != 0, 0)) {
    float xa[8], ya[8];
    _mm256_storeu_ps(xa, x);
    for (int i = 0; i < 8; i++) if (!((_mm256_movemask_ps(_mm256_castsi256_ps(fast)) >> i) & 1)) ya[i] = logf(xa[i]);
    y = _mm256_blendv_ps(_mm256_loadu_ps(ya), y, _mm256_castsi256_ps(fast));
  }
  return y;
}
#endif /* MT_EXPF8_512 / AVX2 分支 */
#endif /* MT_EXPF8 */

void mt_rmsnorm(float *out, const float *x, const float *w, int rows, int dim, float eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * dim;
    float *or_ = out + (size_t)r * dim;
    /* 4 路部分和（fp32），与 PyTorch 的向量化归约同量级 */
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
      float a = xr[i], b = xr[i + 1], c = xr[i + 2], d = xr[i + 3];
      s0 += a * a; s1 += b * b; s2 += c * c; s3 += d * d;
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < dim; i++) s += xr[i] * xr[i];
    /* (mean + eps).rsqrt() */
    float rms = 1.0f / sqrtf(s / (float)dim + eps);
    for (int j = 0; j < dim; j++) or_[j] = (w[j] * xr[j]) * rms;
  }
}

void mt_silu_mul(float *gate, const float *up, int n) {
  mt_silu_mul2(gate, gate, up, n);
}

/* dst[i] = silu(gate[i]) * up[i]（可异址）；与「先 memcpy 再 mt_silu_mul」逐位等价。
 * ⚠ 必须保持 g*(1/(1+e))*u 的形式，不能写成 g/(1+e)*u（见 model.c 的 swiglu 注释）。 */
void mt_silu_mul2(float *dst, const float *gate, const float *up, int n) {
#ifdef MT_EXPF8
  /* 8 lane：`e = expf(-g)` → `s = 1/(1+e)` → `(g*s)*u`，**运算顺序与标量版逐字相同**
   * （加法/除法/乘法都是 IEEE 精确舍入 ⇒ SIMD 与标量逐位一致）。 */
  const __m256 sgn = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(gate + i);
    __m256 e = mt_expf8(_mm256_xor_ps(g, sgn));         /* 取负 = 翻符号位，精确 */
    __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_mul_ps(g, s), _mm256_loadu_ps(up + i)));
  }
  for (; i < n; i++) {
    float g = gate[i];
    dst[i] = (g * sigmoidf_(g)) * up[i];
  }
#else
  for (int i = 0; i < n; i++) {
    float g = gate[i];
    dst[i] = (g * sigmoidf_(g)) * up[i];
  }
#endif
}

void mt_sigmoid2(float *x, int n) {
#ifdef MT_EXPF8
  const __m256 sgn = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    __m256 e = mt_expf8(_mm256_xor_ps(v, sgn));
    __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
    _mm256_storeu_ps(x + i, _mm256_mul_ps(two, s));
  }
  for (; i < n; i++) x[i] = 2.0f * sigmoidf_(x[i]);
#else
  for (int i = 0; i < n; i++) x[i] = 2.0f * sigmoidf_(x[i]);
#endif
}

void mt_add_inplace(float *dst, const float *src, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] += src[i];
}

/* softmax（逐行）。三趟里第二趟最贵（expf），第三趟是逐元素乘。
 * 🔴 位精确的两条约束（别乱"优化"）：
 *   ① `s += e` 是**按 i 顺序的串行累加** ⇒ 不能改成 SIMD 部分和/两两求和
 *      （改了就破坏与标量的逐位一致，softmax 输出差 1 ULP 会穿透 int8 量化边界）；
 *      ⇒ 只把 `expf` 换成 `mt_expf8`（位精确），求和仍一个个加。
 *   ② 第一趟 max 保持**标量**：`p[i] > m` 的 tie 行为（±0）与 NaN 行为（NaN 不更新 m）
 *      和 `_mm256_max_ps` 不一样（`maxps` 遇 NaN 会传播）。 */
void mt_softmax_rows(float *x, int rows, int n) {
  for (int r = 0; r < rows; r++) {
    float *p = x + (size_t)r * n;
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) if (p[i] > m) m = p[i];
    float s = 0.f;
    int i = 0;
#ifdef MT_EXPF8
    const __m256 vm = _mm256_set1_ps(m);
    float eb[8];
    for (; i + 8 <= n; i += 8) {
      __m256 e = mt_expf8(_mm256_sub_ps(_mm256_loadu_ps(p + i), vm));
      _mm256_storeu_ps(p + i, e);
      _mm256_storeu_ps(eb, e);
      /* ⚠ 顺序累加，别合并：保持与标量 `s += e` 同样的舍入序列 */
      s += eb[0]; s += eb[1]; s += eb[2]; s += eb[3];
      s += eb[4]; s += eb[5]; s += eb[6]; s += eb[7];
    }
#endif
    for (; i < n; i++) { float e = expf(p[i] - m); p[i] = e; s += e; }
    const float inv = 1.0f / s;
    i = 0;
#ifdef MT_EXPF8
    const __m256 vinv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8)
      _mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vinv));
#endif
    for (; i < n; i++) p[i] *= inv;
  }
}

/* 线程数由**工作量**派生，不由机器核数派生。
 * 🔴 为什么不能用 omp_get_max_threads()：128 核机器上 V=30008 只够切 8 个 chunk
 *   （每线程 4096 元素），若起 128 线程 ⇒ 104 个线程各拿 1 个、剩余唤醒即空转 ⇒ 净亏。
 *   本机（8 线程）算出来正好是 8 ⇒ 与历史行为一致；4 核机器自动降到 4。
 *   env：MT_LS_PER_THREAD（每线程最小元素数，默认 4096）。 */
static int mt_chunk_threads(long long n) {
#ifdef _OPENMP
  const char *e = getenv("MT_LS_PER_THREAD");
  int per = (e && *e) ? atoi(e) : 4096;
  if (per < 256) per = 256;
  long long t = (n + per - 1) / per;
  int mx = omp_get_max_threads();
  long long r = t < mx ? t : mx;
  return (int)(r < 1 ? 1 : r);
#else
  (void)n; return 1;
#endif
}
static int mt_ls_min(void) {
  const char *e = getenv("MT_LS_MIN");
  int v = (e && *e) ? atoi(e) : 8192;
  return v;
}

#define MT_LS_ROWS_MAX 256

/* 一行「max 趟 + exp 趟 + log 趟」的完整串行实现，返回 ls=log(sum)。
 * 🔴 逐步位口径与原串行实现一字不差：max 用严格 >；s 按 i 顺序累加；logf 逐元素应用。
 *   nb>=0 ⇒ log 趟只处理尾巴 [nb*8, n)（对齐块交给调用方的并行段）；
 *   nb<0  ⇒ log 趟处理整行（非 SIMD 构建走这条）。 */
static float mt_ls_row(float *x, int n, int nb) {
  float m = -FLT_MAX;
  for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
  float s = 0.f;
  int i = 0;
#ifdef MT_EXPF8
  const __m256 vm = _mm256_set1_ps(m);
  float eb[8];
  for (; i + 8 <= n; i += 8) {
    __m256 e = mt_expf8(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm));
    _mm256_storeu_ps(x + i, e);
    _mm256_storeu_ps(eb, e);
    s += eb[0]; s += eb[1]; s += eb[2]; s += eb[3];
    s += eb[4]; s += eb[5]; s += eb[6]; s += eb[7];
  }
#endif
  for (; i < n; i++) { float e = expf(x[i] - m); x[i] = e; s += e; }
  const float ls = logf(s);
  int lo;
#ifdef MT_EXPF8
  lo = (nb >= 0) ? nb * 8 : 0;                  /* SIMD 块已处理过的前缀不用再做 */
#else
  lo = 0;                                        /* 无 SIMD ⇒ 整行都走标量 logf */
#endif
  for (int j = lo; j < n; j++) x[j] = logf(x[j]) - ls;   /* log(e_i / s) = x_i - m - log s */
  return ls;
}

/* 行级 log_softmax：一次调用做完所有行的三趟。
 * 🔴 与「逐行调用 mt_log_softmax」逐位等价：每行调用的是同一个 mt_ls_row，
 *   行间完全独立（缓冲不重叠）⇒ 跨行的调度顺序不影响任何一行的值。
 * ⚠ 必须从**串行上下文**调用：若外层已在 omp parallel 内，嵌套默认关闭 ⇒ 退化为串行。
 *   （decode.c 的批内 beam 原本是「外层按行并行 + 行内两趟」，行内两趟因嵌套关闭全部
 *    串行化了，改成本函数后由**一个** region 覆盖所有任务。） */
void mt_log_softmax_rows(float *x, int n, const int *ridx, int nrows) {
  if (nrows <= 0 || n <= 0) return;
  /* SIMD logf 块数（8 元素/块）。非 MT_EXPF8 构建下 nb=0 ⇒ 整行走标量，
   * mt_ls_row 收到的 nb<0 ⇒ 它自己把整行的 log 趟做完。 */
  int nb = 0, row_nb = -1;
#ifdef MT_EXPF8
  nb = n >> 3; row_nb = nb;
#endif
  const long long total = (long long)nrows * n;
  float ls[MT_LS_ROWS_MAX];
  float *pls = ls;
  if (nrows > MT_LS_ROWS_MAX) {               /* 兜底：分块处理，永不栈溢出 */
    int done = 0;
    while (done < nrows) {
      int chunk = nrows - done > MT_LS_ROWS_MAX ? MT_LS_ROWS_MAX : nrows - done;
      mt_log_softmax_rows(x, n, ridx ? ridx + done : NULL, chunk);
      done += chunk;
    }
    return;
  }
  int T = mt_chunk_threads(total);
  int dopar = 0;
#ifdef _OPENMP
  dopar = (T >= 2) && (total >= mt_ls_min());
#endif
  if (!dopar) {
    for (int r = 0; r < nrows; r++) {
      float *p = x + (size_t)(ridx ? ridx[r] : r) * n;
      const float lsr = mt_ls_row(p, n, row_nb);
#ifdef MT_EXPF8
      const __m256 vls = _mm256_set1_ps(lsr);
      for (int b = 0; b < nb; b++)
        _mm256_storeu_ps(p + (b << 3),
          _mm256_sub_ps(mt_logf8(_mm256_loadu_ps((const float *)(p + (b << 3)))), vls));
#endif
    }
    (void)pls;
    return;
  }
#ifdef _OPENMP
  #pragma omp parallel num_threads(T)
  {
    #pragma omp for schedule(static)
    for (int r = 0; r < nrows; r++)
      pls[r] = mt_ls_row(x + (size_t)(ridx ? ridx[r] : r) * n, n, row_nb);
#ifdef MT_EXPF8
    /* 第三趟：(行,块) 扁平并行 ⇒ 解决原来 schedule(static) 在 nrows 非 T 倍数时的尾倾斜
     *   （例：nrows=17、T=8 ⇒ 关键路径 3 行 = 理想的 1.5×）。逐元素无归约 ⇒ 不改数值。 */
    #pragma omp for collapse(2) schedule(static)
    for (int r = 0; r < nrows; r++)
      for (int b = 0; b < nb; b++) {
        float *p = x + (size_t)(ridx ? ridx[r] : r) * n;
        const __m256 vls = _mm256_set1_ps(pls[r]);
        _mm256_storeu_ps(p + (b << 3),
          _mm256_sub_ps(mt_logf8(_mm256_loadu_ps((const float *)(p + (b << 3)))), vls));
      }
#endif
  }
#endif
}

void mt_log_softmax(float *x, int n) {
  /* 🔴 2026-10-01 修复：原来写作 x[i] = log(e_i) - (log s + m)，把 m 减了两次
   *   （log(e_i)=x_i-m，正确结果是 x_i-m-log s）。greedy 走 argmax 对常数平移不敏感，
   *   所以只在 beam search 里暴露。 */
  mt_log_softmax_rows(x, n, NULL, 1);
  (void)0;
}

/* argmax：**严格 > ⇒ 并列取最小下标**（与标量版逐位等价，见下）
 * 🆕 2026-10-02：V=30008 时这一步原来是**单线程**扫全表（beam/greedy 每步都调）⇒ 并行化。
 *   ⚠ 归约必须**保持"严格大于"的合并规则**（不能用 omp 的 `max` reduction：
 *     它会传播 NaN，而标量版遇到 NaN 时比较为假、直接忽略）。这里每个线程用同一套
 *     "if (v > best)" 循环，再按**线程顺序**用同一条规则合并 ⇒ 并列仍取最小下标。 */
int mt_argmax(const float *x, int n) {
  int best = 0; float bv = x[0];
#ifdef _OPENMP
  /* 🆕 线程数由工作量派生（MT_LS_PER_THREAD），不再硬编码 8、也不再直接用
   *   omp_get_max_threads()：128 核机器上 V=30008 只够 8 个 chunk，多起的线程全是净亏。 */
  int T = mt_chunk_threads(n);
  if (n >= 4096 && T >= 2) {
    if (T > 64) T = 64;
    float pv[64]; int pi[64];
#pragma omp parallel for schedule(static) num_threads(T)
    for (int t = 0; t < T; t++) {
      const int lo = (int)((long)t * n / T), hi = (int)((long)(t + 1) * n / T);
      int b = lo; float v = x[lo];
      for (int i = lo + 1; i < hi; i++) if (x[i] > v) { v = x[i]; b = i; }
      pv[t] = v; pi[t] = b;
    }
    best = pi[0]; bv = pv[0];    /* 🔴 必须从 t=0 起：0 号线程覆盖的是前缀，它的最大值原来被漏掉了 */
    for (int t = 1; t < T; t++) if (pv[t] > bv) { bv = pv[t]; best = pi[t]; }
    return best;
  }
#endif
  for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }  /* 严格 > ⇒ 保留首个最大 */
  return best;
}

/* top-k（beam 用，k 很小）。
 * 🔴 2026-10-02 重写：原来每趟扫一遍全表（O(k·n)）+ 每趟 calloc(n) —— V=30008 时实测
 *   **0.57 ms/步**。现在**一趟扫描**维护一个长度 k 的有序候选表，只在**严格大于**表中最小者时插入
 *   ⇒ **与原来「k 趟、每趟取严格最大的最小下标」逐位等价**（并列仍取最小下标）；
 *   也没了 calloc。⚠ NaN / -FLT_MAX 的行为也保持一致（比较为假 ⇒ 不插入）。 */
void mt_topk(const float *x, int n, int k, float *val, int *idx) {
  for (int j = 0; j < k; j++) { val[j] = -FLT_MAX; idx[j] = -1; }
  for (int i = 0; i < n; i++) {
    const float v = x[i];
    if (!(v > val[k - 1])) continue;            /* 进不了前 k 就跳过 */
    int p = k - 1;
    while (p > 0 && v > val[p - 1]) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
    val[p] = v; idx[p] = i;
  }
}
