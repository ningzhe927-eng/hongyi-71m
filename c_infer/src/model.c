#include "model.h"
#include "ops.h"
#include "prof.h"
#include "blas.h"
#include "weights.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* ---- QK 点积：一次算 8 个 key（不改数值） ----
 * 🔴 为什么不能像 `dot8` 那样用 8 路 SIMD 归约：那会**重排求和顺序**，ULP 级误差经 int8
 *    激活量化边界放大后**真的会改译文**（实测 300 句里 16 句不同）。
 * ✅ 这里的做法：8 条**独立的串行链**并行推进 —— 每个点积仍是 `a += q[d]*k[d]` 的
 *    逐次累加（顺序完全不变 ⇒ **与标量版逐位等价**，fp32/int8 两档都能用），
 *    但 8 条链互不依赖 ⇒ 把 FMA 延迟（~4 cycle）藏掉，ILP 填满。
 *    （动机：实测 QK 段比 MAC 数相同的 p·v 段慢 ~9×，就是因为 p·v 天然有 76 个独立累加器。） */
static void qk_block8(const float *qq, const float *kp, int dh, float *row, float scale) {
  float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int d = 0; d < dh; d++) {
    const float q = qq[d];
    for (int i = 0; i < 8; i++) a[i] += q * kp[(size_t)i * dh + d];
  }
  for (int i = 0; i < 8; i++) row[i] = a[i] * scale;
}

#define D_D D_MODEL
#define HS N_HEADS_SELF
#define DS D_H_SELF
#define HC N_HEADS_CROSS
#define DC D_H_CROSS
#define DQC D_QKV_CROSS
#define DF D_FF
#define V VOCAB_SIZE
#define EPSF RMS_EPS

/* ================= 权重指针绑定 ================= */
static void bind_weights(mt_model_t *m) {
  m->tok_embed = (mt_lin_t) LINIT_tok_embed;
  m->enc_final_norm = g_W_enc_final_norm;
  m->dec_final_norm = g_W_dec_final_norm;

#define SA(P) .sa = { .in_gate = LINIT_##P##_sa_in_gate, .in_b = g_W_##P##_mixer_in_proj_b, \
              .out = LINIT_##P##_mixer_out_proj, .out_b = g_W_##P##_mixer_out_proj_b, \
              .q_norm = g_W_##P##_mixer_q_norm, .k_norm = g_W_##P##_mixer_k_norm }

#define ENC_L(L) m->enc[L] = (mt_enc_layer_t){ \
    .norm1 = g_W_enc##L##_norm1, .norm2 = g_W_enc##L##_norm2, SA(enc##L), \
    .wgu = LINIT_enc##L##_mlp_wgu, .wd = LINIT_enc##L##_mlp_w_down }
  ENC_L(0); ENC_L(1); ENC_L(2); ENC_L(3); ENC_L(4); ENC_L(5); ENC_L(6);
#undef ENC_L

#define DEC_L(L) m->dec[L] = (mt_dec_layer_t){ \
    .norm1 = g_W_dec##L##_norm1, .norm_ca = g_W_dec##L##_norm_ca, .norm2 = g_W_dec##L##_norm2, \
    SA(dec##L), \
    .cqg = LINIT_dec##L##_ca_qg, .ckv = LINIT_dec##L##_ca_kv, .co = LINIT_dec##L##_cross_attn_o_proj, \
    .cq_norm = g_W_dec##L##_cross_attn_q_norm, .ck_norm = g_W_dec##L##_cross_attn_k_norm, \
    .wgu = LINIT_dec##L##_mlp_wgu, .wd = LINIT_dec##L##_mlp_w_down }
  DEC_L(0); DEC_L(1); DEC_L(2);
#undef DEC_L
#undef SA
}

/* in_proj 的 bias 只有 3·D，而拼接后的 in_gate 输出是 4·D ⇒ 补 D 个 0。
 * ⚠ 必须补零后用 `mt_lin_apply_bias`（bias 在 sgemm 内部以 beta=1 累加），
 *   不能「先 GEMM 再手动加 bias」：两者**浮点不等价**，M>1 时差 ~1e-7，
 *   经 int8 激活量化边界放大后可达 1e-2。补的 0.0f 是精确加法，不改数值。 */
static int pad_in_bias(mt_model_t *m) {
  float *p = (float *)calloc((size_t)(N_ENC + N_DEC) * 4 * D_D, sizeof(float));
  if (!p) return -1;
  m->in_b_pad = p;
  for (int i = 0; i < N_ENC; i++) {
    memcpy(p, m->enc[i].sa.in_b, 3 * D_D * sizeof(float));
    m->enc[i].sa.in_b = p; p += 4 * D_D;
  }
  for (int i = 0; i < N_DEC; i++) {
    memcpy(p, m->dec[i].sa.in_b, 3 * D_D * sizeof(float));
    m->dec[i].sa.in_b = p; p += 4 * D_D;
  }
  return 0;
}

mt_model_t *mt_model_create(void) {
  mt_model_t *m = (mt_model_t *)calloc(1, sizeof(mt_model_t));
  if (!m) return NULL;
  bind_weights(m);
  if (pad_in_bias(m) != 0) { free(m); return NULL; }
  if (mt_rope_init(&m->rope, ROPE_MAX_POS, DS, ROPE_BASE) != 0) { mt_model_destroy(m); return NULL; }
  return m;
}
void mt_model_destroy(mt_model_t *m) {
  if (!m) return;
  mt_rope_free(&m->rope);
  free(m->in_b_pad);
  free(m);
}

/* ================= ctx / scratch ================= */
static float *amalloc(size_t n) {
  void *p = NULL;
  if (n == 0) return NULL;
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) return NULL;
  return (float *)p;
}

mt_ctx_t *mt_ctx_create(int B, int cap_src, int cap_tgt) {
  mt_ctx_t *c = (mt_ctx_t *)calloc(1, sizeof(mt_ctx_t));
  if (!c) return NULL;
  c->B = B; c->cap_src = cap_src; c->cap_tgt = cap_tgt;
  int maxL = cap_src > cap_tgt ? cap_src : cap_tgt;
  int maxS = maxL;
  c->enc_out = amalloc((size_t)B * cap_src * D_D);
  c->b1 = amalloc((size_t)B * maxL * D_D);
  c->b2 = amalloc((size_t)B * maxL * D_D);
  c->b3 = amalloc((size_t)B * maxL * D_D);
  c->mg = amalloc((size_t)B * maxL * D_D);
  c->qkvg = amalloc((size_t)B * maxL * 4 * D_D);
  c->qh = amalloc((size_t)B * HS * maxL * DS);
  c->kh = amalloc((size_t)B * HS * maxL * DS);
  c->vh = amalloc((size_t)B * HS * maxL * DS);
  c->ah = amalloc((size_t)B * HS * maxL * DS);
  c->cq = amalloc((size_t)B * HC * maxL * DC);
  c->scores = amalloc((size_t)B * HS * maxL * maxS);
  c->g = amalloc((size_t)B * maxL * D_D);
  c->ff_g = amalloc((size_t)B * maxL * DF);
  c->ff_gu = amalloc((size_t)B * maxL * 2 * DF);
  c->gf = amalloc((size_t)B * maxL * 2 * DQC);
  for (int i = 0; i < N_DEC; i++) {
    c->L[i].k = amalloc((size_t)B * HS * cap_tgt * DS);
    c->L[i].v = amalloc((size_t)B * HS * cap_tgt * DS);
    c->L[i].ck = amalloc((size_t)B * HC * cap_src * DC);
    c->L[i].cv = amalloc((size_t)B * HC * cap_src * DC);
    c->L[i].len = 0;
  }
  if (!c->enc_out || !c->b1 || !c->b2 || !c->b3 || !c->mg || !c->qkvg || !c->qh ||
      !c->kh || !c->vh || !c->ah || !c->cq || !c->scores || !c->g || !c->ff_g ||
      !c->gf || !c->ff_gu || !c->L[0].k || !c->L[0].ck) {
    mt_ctx_destroy(c); return NULL;
  }
  return c;
}

void mt_ctx_destroy(mt_ctx_t *c) {
  if (!c) return;
  free(c->enc_out); free(c->b1); free(c->b2); free(c->b3); free(c->mg); free(c->qkvg);
  free(c->qh); free(c->kh); free(c->vh); free(c->ah); free(c->cq);
  free(c->scores); free(c->g); free(c->ff_g); free(c->gf); free(c->ff_gu);
  for (int i = 0; i < N_DEC; i++) { free(c->L[i].k); free(c->L[i].v); free(c->L[i].ck); free(c->L[i].cv); }
  free(c);
}
void mt_ctx_reset_kv(mt_ctx_t *c) { for (int i = 0; i < N_DEC; i++) c->L[i].len = 0; }

/* ================= 转置 ================= */
/* s: [b][l][h][dh]，每行(l)的**起始间隔**由 srow 给出（拼接后 q/k/v 不再紧邻 ⇒ srow=2·DQC） */
static void t_lhd2hld(const float *s, float *d, int B, int L, int H, int Dh, int dstride, int srow) {
  for (int b = 0; b < B; b++)
    for (int l = 0; l < L; l++)
      for (int h = 0; h < H; h++) {
        const float *sp = s + (size_t)(b * L + l) * srow + (size_t)h * Dh;
        float *dp = d + (((size_t)b * H + h) * dstride + l) * Dh;
        memcpy(dp, sp, (size_t)Dh * sizeof(float));
      }
}
/* ⚠ `t_hld2lhd` 已删：2026-10-02 起「回排 + 乘 gate」融成一趟（见 self_attn/cross_attn 里的融合循环）*/

/* SwiGLU：wgu = w_gate ⊕ w_up 一次算完（导出期已拼接），再按行拆开做 silu(gate)*up
 * ⚠ 必须走 `mt_silu_mul`（即 `g*(1/(1+e))*u`），不能写成 `g/(1+e)*u`：
 *   两者代数相等但**浮点不等价**，ULP 级差异会被 int8 的激活量化在舍入边界放大成 ~1e-2。 */
static void swiglu(mt_ctx_t *c, const float *x, float *y, int M,
                   const mt_lin_t *wgu, const mt_lin_t *wd) {
  mt_lin_apply(c->ff_gu, x, wgu, M);          /* (M, 2·D_FF)：[0,D_FF)=gate，[D_FF,2·D_FF)=up */
  /* 逐元素部分（2026-10-01 优化）：
   *   ① 走 `mt_silu_mul2` 异址写，**省掉每行那次 memcpy**（实测 memcpy 只占本段 ~1%，聊胜于无）；
   *   ② **按行线程化**（`if(M > 1)`）：编码侧 M=L_src≈48，本段实测 0.42 ms/层·句，
   *      解码侧 M=1 只有 0.014 ms ⇒ 收益全在编码侧；M=1 时开 omp 反而更慢（与 self_attn 同型）。
   *      🔴 **但只能在「没有 BLAS 线程池」的构建里开**（`mt_blas_get_threads()==0` ⇒ int8 档）：
   *      fp32 档的 OpenBLAS 自带 8 线程（且忙等自旋），再叠一层 OMP ⇒ 过订阅，
   *      实测 fp32 batch=16 −19%、beam4 −29%（而 int8 档 +17%）。
   *   ✅ 逐元素 ⇒ 行间无依赖、无归约 ⇒ **不改数值**（已用 300 句 diff=0 验证）。 */
  { MT_PROF_B0();
    const int par = (mt_blas_get_threads() == 0) && (M > 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(par)
#endif
    for (int m = 0; m < M; m++) {
      const float *g = c->ff_gu + (size_t)m * 2 * DF;
      mt_silu_mul2(c->ff_g + (size_t)m * DF, g, g + DF, DF);
    }
  MT_PROF_E0(PROF_SILU); }
  mt_lin_apply(y, c->ff_g, wd, M);
}

/* embedding 行取值（fp32 直拷 / int8 反量化） */
static inline void embed_row(float *dst, const mt_lin_t *E, int id) {
  if (E->w32) {
    memcpy(dst, E->w32 + (size_t)id * E->K, (size_t)E->K * sizeof(float));
  } else {
    const int8_t *row = E->w8 + (size_t)id * E->K;
    const float s = E->wscale[id];
    for (int k = 0; k < E->K; k++) dst[k] = (float)row[k] * s;
  }
}

/* ================= Self-Attention ================= */
static void self_attn(mt_model_t *m, mt_ctx_t *c, const float *x, float *y,
                      int B, int L, int causal, const uint8_t *key_pad,
                      float *kc, float *vc, int cap, int past, const mt_sa_w_t *W) {
  const int M = B * L;
  const float scale = (float)(1.0 / sqrt((double)DS));
  /* 注意力两段的线程化判据（2026-10-01 改）：
   * 旧判据只看 `L > 1` ⇒ 解码侧（L=1）**无论 batch 多大都串行**，而注意力工作量 ∝ B
   * ⇒ batch=16 / beam 的解码注意力全压在单核上（int8 b16 反而慢于 b1 的原因之一）。
   * 新判据：任务数 B·H 够多就开（(b,h) 之间独立 ⇒ **不改数值**）。
   * ⚠ 仍不在 fp32 档开新档（有 BLAS 线程池时再叠 OMP ⇒ 过订阅，同 swiglu 的教训）。 */
  /* 解码（L=1, B=1）时自注意力**故意不并行**：实测「scores 省 4.5 µs + p·v 多花 3.1 µs（多一次 fork）」
   ⇒ 基本白干；而 cross_attn 那边的 ca_core 有 1.85× 收益（见那里的阈值）。*/
  const int par_attn = (L > 1) || ((B * HS > 8) && (mt_blas_get_threads() == 0));
  /* in_gate 一次算完 qkv+gate（行 [0,3D)=qkv，[3D,4D)=gate）；in_b 已补 0 到 4D */
  { MT_PROF_B0();
    mt_lin_apply_bias(c->qkvg, x, &W->in_gate, W->in_b, M);
    MT_PROF_E0(PROF_QKVPROJ); }
  { MT_PROF_B0();
    for (int b = 0; b < B; b++)
      for (int l = 0; l < L; l++) {
        const float *base = c->qkvg + ((size_t)b * L + l) * 4 * D_D;
        for (int h = 0; h < HS; h++) {
          size_t dst = (((size_t)b * HS + h) * L + l) * DS;
          memcpy(c->qh + dst, base + h * DS, DS * sizeof(float));
          memcpy(c->kh + dst, base + D_D + h * DS, DS * sizeof(float));
          memcpy(c->vh + dst, base + 2 * D_D + h * DS, DS * sizeof(float));
        }
      }
    MT_PROF_E0(PROF_QKVSPLIT); }
  mt_rmsnorm(c->qh, c->qh, W->q_norm, B * HS * L, DS, EPSF);
  mt_rmsnorm(c->kh, c->kh, W->k_norm, B * HS * L, DS, EPSF);
  { MT_PROF_B0();
  mt_rope_apply(&m->rope, c->qh, B, HS, L, past);
  mt_rope_apply(&m->rope, c->kh, B, HS, L, past);
    MT_PROF_E0(PROF_ROPE); }

  int S_keys, kstride;
  const float *kbase, *vbase;
  { MT_PROF_B0();
  if (kc) {
    for (int b = 0; b < B; b++)
      for (int h = 0; h < HS; h++) {
        size_t dst = (((size_t)b * HS + h) * cap + past) * DS;
        size_t src = (((size_t)b * HS + h) * L) * DS;
        memcpy(kc + dst, c->kh + src, (size_t)L * DS * sizeof(float));
        memcpy(vc + dst, c->vh + src, (size_t)L * DS * sizeof(float));
      }
    S_keys = past + L; kbase = kc; vbase = vc; kstride = cap;
  } else {
    S_keys = L; kbase = c->kh; vbase = c->vh; kstride = L;
  }
    MT_PROF_E0(PROF_QKVSPLIT); }

  { MT_PROF_B0();
    /* (b,h) 之间完全独立 ⇒ 线程化**不改数值**。
     * ⚠ 只在 L>1 时开：解码逐步 L=1，8 个头的活太小，omp 调度开销反超收益（实测）。 */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(par_attn)
#endif
    for (int b = 0; b < B; b++)
      for (int h = 0; h < HS; h++) {
        const float *qp = c->qh + (((size_t)b * HS + h) * L) * DS;
        const float *kp = kbase + (((size_t)b * HS + h) * kstride) * DS;
        float *sp = c->scores + (((size_t)b * HS + h) * L) * S_keys;
        for (int l = 0; l < L; l++) {
          float *row = sp + (size_t)l * S_keys;
          const float *qq = qp + (size_t)l * DS;
          int s = 0;
          for (; s + 8 <= S_keys; s += 8) qk_block8(qq, kp + (size_t)s * DS, DS, row + s, scale);
          for (; s < S_keys; s++) {
            const float *kk = kp + (size_t)s * DS;
            float acc = 0.f;
            for (int d = 0; d < DS; d++) acc += qq[d] * kk[d];
            row[s] = acc * scale;
          }
          if (key_pad) {
            const uint8_t *pm = key_pad + (size_t)b * S_keys;
            for (int s = 0; s < S_keys; s++) if (pm[s]) row[s] += PAD_BIAS;
          }
          if (causal && kc == NULL && L > 1)
            for (int s = l + 1; s < S_keys; s++) row[s] += PAD_BIAS;
        }
        mt_softmax_rows(sp, L, S_keys);
      }
    MT_PROF_E0(PROF_SCORES); }
  { MT_PROF_B0();
    /* p·v 加权和（与 scores 分成两个并行区：各自负载均衡；代价是两次 omp fork） */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(par_attn)
#endif
    for (int b = 0; b < B; b++)
      for (int h = 0; h < HS; h++) {
        const float *vp = vbase + (((size_t)b * HS + h) * kstride) * DS;
        float *sp = c->scores + (((size_t)b * HS + h) * L) * S_keys;
        float *ap = c->ah + (((size_t)b * HS + h) * L) * DS;
        for (int l = 0; l < L; l++) {
          const float *row = sp + (size_t)l * S_keys;
          float *o = ap + (size_t)l * DS;
          for (int d = 0; d < DS; d++) o[d] = 0.f;
          for (int s = 0; s < S_keys; s++) {
            float p = row[s];
            if (p == 0.f) continue;
            const float *vv = vp + (size_t)s * DS;
            for (int d = 0; d < DS; d++) o[d] += p * vv[d];
          }
        }
      }
    MT_PROF_E0(PROF_ATTNOUT); }
  /* 🆕 融合（2026-10-02）：原来是「memcpy 抠 gate → sigmoid → 逐元素乘 → 转置」四趟。
   *   ① gate 本来就在 `qkvg` 的 [3D,4D) 里，而 qkv_split 之后 **没人再读 qkvg** ⇒ 直接**原地** sigmoid，
   *      省掉那次 memcpy（也省掉 c->g 这块中间缓冲的读写）；
   *   ② 「乘 gate」折进「转置」：`mg[lhd] = ah[hld] * gate[lhd]` 一趟出结果，省一趟 ah 的读+写。
   *   ⚠ 逐位等价：乘法仍是 `ap[d] * gp[d]`（同一对操作数、同一顺序），只是目的地从 ah 换成了 mg。 */
  { MT_PROF_B0();
    for (int b = 0; b < B; b++)
      for (int l = 0; l < L; l++)
        mt_sigmoid2(c->qkvg + ((size_t)b * L + l) * 4 * D_D + 3 * D_D, D_D);
    for (int b = 0; b < B; b++)
      for (int l = 0; l < L; l++) {
        const float *gp = c->qkvg + ((size_t)b * L + l) * 4 * D_D + 3 * D_D;
        for (int h = 0; h < HS; h++) {
          const float *ap = c->ah + (((size_t)b * HS + h) * L + l) * DS;
          float *mp = c->mg + (((size_t)b * L + l) * HS + h) * DS;
          for (int d = 0; d < DS; d++) mp[d] = ap[d] * gp[h * DS + d];
        }
      }
    MT_PROF_E0(PROF_GATEMUL); }
  { MT_PROF_B0();
  mt_lin_apply_bias(y, c->mg, &W->out, W->out_b, M);
    MT_PROF_E0(PROF_OUTPROJ); }
}

/* ================= Cross-Attention ================= */
static void build_cross_kv(mt_ctx_t *c, float *ck, float *cv, int B, int L_src,
                           const mt_dec_layer_t *W) {
  /* k_proj ⊕ v_proj 一次算完（同吃 encoder 输出）；k 在行 [0,DQC)，v 在 [DQC,2·DQC) */
  mt_lin_apply(c->gf, c->enc_out, &W->ckv, B * L_src);
  for (int m = 0; m < B * L_src; m++)          /* 只对 k 部分做 RMSNorm（HC×DC 连续） */
    mt_rmsnorm(c->gf + (size_t)m * 2 * DQC, c->gf + (size_t)m * 2 * DQC,
               W->ck_norm, HC, DC, EPSF);
  t_lhd2hld(c->gf, ck, B, L_src, HC, DC, L_src, 2 * DQC);
  t_lhd2hld(c->gf + DQC, cv, B, L_src, HC, DC, L_src, 2 * DQC);
}

static void cross_attn(mt_ctx_t *c, const float *x, float *y, int B, int L,
                       const float *ck, const float *cv, int L_src,
                       const mt_dec_layer_t *W) {
  const int M = B * L;
  const float scale = (float)(1.0 / sqrt((double)DC));
  /* 同 self_attn：解码侧 L=1 也要按 B·H 决定线程化（判据与数值安全性见 self_attn 注释） */
  const int par_attn = (L > 1) || ((B * HC >= 8) && (mt_blas_get_threads() == 0));
  /* q_proj ⊕ gate_proj 一次算完（同吃 norm_ca 输出）；q 在行 [0,DQC)，gate 在 [DQC,2·DQC) */
  mt_lin_apply(c->gf, x, &W->cqg, M);
  for (int m = 0; m < M; m++)
    mt_rmsnorm(c->gf + (size_t)m * 2 * DQC, c->gf + (size_t)m * 2 * DQC,
               W->cq_norm, HC, DC, EPSF);
  t_lhd2hld(c->gf, c->cq, B, L, HC, DC, L, 2 * DQC);

  { MT_PROF_B0();
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(par_attn)
#endif
  for (int b = 0; b < B; b++)
    for (int h = 0; h < HC; h++) {
      const float *qp = c->cq + (((size_t)b * HC + h) * L) * DC;
      const float *kp = ck + (((size_t)b * HC + h) * L_src) * DC;
      const float *vp = cv + (((size_t)b * HC + h) * L_src) * DC;
      float *sp = c->scores + (((size_t)b * HC + h) * L) * L_src;
      for (int l = 0; l < L; l++) {
        float *row = sp + (size_t)l * L_src;
        const float *qq = qp + (size_t)l * DC;
        int s = 0;
        for (; s + 8 <= L_src; s += 8) qk_block8(qq, kp + (size_t)s * DC, DC, row + s, scale);
        for (; s < L_src; s++) {
          const float *kk = kp + (size_t)s * DC;
          float acc = 0.f;
          for (int d = 0; d < DC; d++) acc += qq[d] * kk[d];
          row[s] = acc * scale;
        }
        if (c->enc_pad) {
          const uint8_t *pm = c->enc_pad + (size_t)b * L_src;
          for (int s = 0; s < L_src; s++) if (pm[s]) row[s] += PAD_BIAS;
        }
      }
      mt_softmax_rows(sp, L, L_src);
      float *ap = c->ah + (((size_t)b * HC + h) * L) * DC;
      for (int l = 0; l < L; l++) {
        const float *row = sp + (size_t)l * L_src;
        float *o = ap + (size_t)l * DC;
        for (int d = 0; d < DC; d++) o[d] = 0.f;
        for (int s = 0; s < L_src; s++) {
          float p = row[s];
          if (p == 0.f) continue;
          const float *vv = vp + (size_t)s * DC;
          for (int d = 0; d < DC; d++) o[d] += p * vv[d];
        }
      }
    }
    MT_PROF_E0(PROF_CACORE); }

  /* 同 self_attn 的融合（见那里的说明）：gate 在 `c->gf` 的上半，原地 sigmoid + 乘与转置合一 */
  for (int m = 0; m < M; m++)
    mt_sigmoid2(c->gf + (size_t)m * 2 * DQC + DQC, DQC);
  for (int b = 0; b < B; b++)
    for (int l = 0; l < L; l++) {
      const float *gp = c->gf + ((size_t)b * L + l) * 2 * DQC + DQC;
      for (int h = 0; h < HC; h++) {
        const float *ap = c->ah + (((size_t)b * HC + h) * L + l) * DC;
        float *mp = c->mg + (((size_t)b * L + l) * HC + h) * DC;
        for (int d = 0; d < DC; d++) mp[d] = ap[d] * gp[h * DC + d];
      }
    }
  mt_lin_apply(y, c->mg, &W->co, M);
}

/* ================= Blocks ================= */
static void enc_block(mt_model_t *m, mt_ctx_t *c, float *h, int B, int L,
                      const uint8_t *key_pad, const mt_enc_layer_t *W) {
  size_t n = (size_t)B * L * D_D;
  { MT_PROF_B0(); mt_rmsnorm(c->b2, h, W->norm1, B * L, D_D, EPSF); MT_PROF_E0(PROF_RMSNORM); }
  { MT_PROF_B0(); self_attn(m, c, c->b2, c->b3, B, L, 0, key_pad, NULL, NULL, 0, 0, &W->sa); MT_PROF_E0(PROF_SELF); }
  mt_add_inplace(h, c->b3, n);
  { MT_PROF_B0(); mt_rmsnorm(c->b2, h, W->norm2, B * L, D_D, EPSF); MT_PROF_E0(PROF_RMSNORM); }
  { MT_PROF_B0(); swiglu(c, c->b2, c->b3, B * L, &W->wgu, &W->wd); MT_PROF_E0(PROF_SWIGLU); }
  mt_add_inplace(h, c->b3, n);
}

static void dec_block(mt_model_t *m, mt_ctx_t *c, float *h, int B, int L, int li,
                      const mt_dec_layer_t *W) {
  size_t n = (size_t)B * L * D_D;
  int past = c->L[li].len;
  { MT_PROF_B0(); mt_rmsnorm(c->b2, h, W->norm1, B * L, D_D, EPSF); MT_PROF_E0(PROF_RMSNORM); }
  { MT_PROF_B0(); self_attn(m, c, c->b2, c->b3, B, L, 1, NULL, c->L[li].k, c->L[li].v, c->cap_tgt, past, &W->sa); MT_PROF_E0(PROF_SELF); }
  mt_add_inplace(h, c->b3, n);
  c->L[li].len = past + L;
  { MT_PROF_B0(); mt_rmsnorm(c->b2, h, W->norm_ca, B * L, D_D, EPSF); MT_PROF_E0(PROF_RMSNORM); }
  { MT_PROF_B0(); cross_attn(c, c->b2, c->b3, B, L, c->L[li].ck, c->L[li].cv, c->L_src, W); MT_PROF_E0(PROF_CROSS); }
  mt_add_inplace(h, c->b3, n);
  { MT_PROF_B0(); mt_rmsnorm(c->b2, h, W->norm2, B * L, D_D, EPSF); MT_PROF_E0(PROF_RMSNORM); }
  { MT_PROF_B0(); swiglu(c, c->b2, c->b3, B * L, &W->wgu, &W->wd); MT_PROF_E0(PROF_SWIGLU); }
  mt_add_inplace(h, c->b3, n);
}

/* ================= 对外接口 ================= */
void mt_encode(mt_model_t *m, mt_ctx_t *c, const int *src_ids, const uint8_t *src_pad,
               int B, int L) {
  c->L_src = L; c->enc_pad = src_pad;
  mt_prof_phase = 0;                 /* 阶段分离：以下都算「编码」 */
  { MT_PROF_B0();
    for (int b = 0; b < B; b++)
      for (int l = 0; l < L; l++)
        embed_row(c->b1 + ((size_t)b * L + l) * D_D, &m->tok_embed, src_ids[(size_t)b * L + l]);
    MT_PROF_E0(PROF_EMBED); }
  for (int i = 0; i < N_ENC; i++) enc_block(m, c, c->b1, B, L, src_pad, &m->enc[i]);
  mt_rmsnorm(c->b1, c->b1, m->enc_final_norm, B * L, D_D, EPSF);
  memcpy(c->enc_out, c->b1, (size_t)B * L * D_D * sizeof(float));
  for (int i = 0; i < N_DEC; i++) {
    build_cross_kv(c, c->L[i].ck, c->L[i].cv, B, L, &m->dec[i]);
    c->L[i].len = 0;
  }
}

void mt_decode_step(mt_model_t *m, mt_ctx_t *c, const int *tgt_ids, int B, int L,
                    float *logits) {
  mt_prof_phase = 1;                 /* 阶段分离：以下都算「解码」 */
  if (mt_prof_on) mt_prof_steps++;   /* 一次调用 = 一步（batch/beam 内并行算作 1 步）*/
  { MT_PROF_B0();
    for (int b = 0; b < B; b++)
      for (int l = 0; l < L; l++)
        embed_row(c->b1 + ((size_t)b * L + l) * D_D, &m->tok_embed, tgt_ids[(size_t)b * L + l]);
    MT_PROF_E0(PROF_EMBED); }
  for (int i = 0; i < N_DEC; i++) dec_block(m, c, c->b1, B, L, i, &m->dec[i]);
  mt_rmsnorm(c->b1, c->b1, m->dec_final_norm, B * L, D_D, EPSF);
  { MT_PROF_B0();
    for (int b = 0; b < B; b++)
      mt_lin_apply(logits + (size_t)b * V, c->b1 + ((size_t)b * L + L - 1) * D_D, &m->tok_embed, 1);
    MT_PROF_E0(PROF_LMHEAD); }
}
