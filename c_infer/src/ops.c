#include "ops.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#define MT_EXPF8 1
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

void mt_log_softmax(float *x, int n) {
  /* 🔴 2026-10-01 修复：原来写作 x[i] = log(e_i) - (log s + m)，把 m 减了两次
   *   （log(e_i)=x_i-m，正确结果是 x_i-m-log s）。greedy 走 argmax 对常数平移不敏感，
   *   所以只在 beam search 里暴露。 */
  float m = -FLT_MAX;
#ifdef _OPENMP
  if (n >= 4096) {   /* 🆕 这一趟原来单线程（beam 每步 4 行 × 30K）；按"严格 >"规则分块归约，逐位等价 */
    const int T = 8;
    float pm[8];
#pragma omp parallel for schedule(static) num_threads(T)
    for (int t = 0; t < T; t++) {
      const int lo = (int)((long)t * n / T), hi = (int)((long)(t + 1) * n / T);
      float v = -FLT_MAX;
      for (int j = lo; j < hi; j++) if (x[j] > v) v = x[j];
      pm[t] = v;
    }
    for (int t = 0; t < T; t++) if (pm[t] > m) m = pm[t];
  } else
#endif
  for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
  float s = 0.f;
  int i = 0;
#ifdef MT_EXPF8
  /* expf 走位精确 SIMD（见 mt_silu_mul2 的说明）；求和仍**按 i 顺序串行**加 */
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
  i = 0;
#ifdef MT_EXPF8
  {   /* 这一趟是全词表 30K 次 logf（beam 的 log_softmax 里最贵的一段）⇒ 位精确 SIMD + 多线程
       （逐元素、无归约 ⇒ 并行不改数值；块边界按 8 对齐，尾巴留给串行段）*/
    const __m256 vls = _mm256_set1_ps(ls);
    const int nb = n >> 3;                 /* 8 元素块数 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb >= 4096)
#endif
    for (int b = 0; b < nb; b++) {
      const int o = b << 3;
      _mm256_storeu_ps(x + o, _mm256_sub_ps(mt_logf8(_mm256_loadu_ps(x + o)), vls));
    }
    i = nb << 3;
  }
#endif
  for (; i < n; i++) x[i] = logf(x[i]) - ls;         /* log(e_i / s) = x_i - m - log s */
}

/* argmax：**严格 > ⇒ 并列取最小下标**（与标量版逐位等价，见下）
 * 🆕 2026-10-02：V=30008 时这一步原来是**单线程**扫全表（beam/greedy 每步都调）⇒ 并行化。
 *   ⚠ 归约必须**保持"严格大于"的合并规则**（不能用 omp 的 `max` reduction：
 *     它会传播 NaN，而标量版遇到 NaN 时比较为假、直接忽略）。这里每个线程用同一套
 *     "if (v > best)" 循环，再按**线程顺序**用同一条规则合并 ⇒ 并列仍取最小下标。 */
int mt_argmax(const float *x, int n) {
  int best = 0; float bv = x[0];
#ifdef _OPENMP
  if (n >= 4096) {
    int nt = omp_get_max_threads();
    int T = nt > 8 ? 8 : nt;
    float pv[8]; int pi[8];
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
