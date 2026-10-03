#ifndef MT_MODEL_H
#define MT_MODEL_H

#include <stdint.h>
#include "config.h"
#include "rope.h"
#include "blas.h"

/* self-attention 权重（enc/dec 共有）
 * 🔴 `in_gate` 是**导出期拼接**的 (4·D, D)：前 3·D 行 = in_proj(qkv)，后 D 行 = gate_proj。
 *    两者都吃同一个 norm1 输出 ⇒ 一次 GEMM 出 qkv+gate（M7 报告 §7：让内存流变长的唯一被数据支持的杠杆）。 */
typedef struct {
  mt_lin_t in_gate;              /* (3D+D, D) = in_proj ⊕ gate_proj */
  mt_lin_t out;                  /* (D, D) */
  const float *in_b, *out_b;     /* in_b 只作用于前 3·D 行（gate 无 bias） */
  const float *q_norm, *k_norm;  /* E6 (Dh,) */
} mt_sa_w_t;

typedef struct {
  const float *norm1, *norm2;
  mt_sa_w_t sa;
  mt_lin_t wgu;                  /* (2·D_FF, D) = w_gate ⊕ w_up（导出期拼接） */
  mt_lin_t wd;                   /* (D, D_FF) */
} mt_enc_layer_t;

typedef struct {
  const float *norm1, *norm_ca, *norm2;
  mt_sa_w_t sa;
  mt_lin_t cqg;                  /* (2·D_QKV, D) = q_proj ⊕ gate_proj（同吃 norm_ca 输出） */
  mt_lin_t ckv;                  /* (2·D_QKV, D) = k_proj ⊕ v_proj（同吃 encoder 输出） */
  mt_lin_t co;                   /* (D, D_QKV) */
  const float *cq_norm, *ck_norm;
  mt_lin_t wgu;                  /* (2·D_FF, D) */
  mt_lin_t wd;
} mt_dec_layer_t;

typedef struct {
  mt_lin_t tok_embed;            /* (V, D) tied：embedding gather + lm_head */
  const float *enc_final_norm;
  const float *dec_final_norm;
  float *in_b_pad;               /* (N_ENC+N_DEC, 4D)：in_proj bias 补 0 到 4D（见 model.c） */
  mt_enc_layer_t enc[N_ENC];
  mt_dec_layer_t dec[N_DEC];
  mt_rope_t rope;
} mt_model_t;

typedef struct {
  float *k, *v;        /* self: (B, H_SELF, cap_tgt, D_H_SELF) */
  float *ck, *cv;      /* cross: (B, H_CROSS, cap_src, D_H_CROSS) */
  int len;
} mt_kv_layer_t;

typedef struct {
  int B, cap_src, cap_tgt, L_src;
  const uint8_t *enc_pad;
  mt_kv_layer_t L[N_DEC];
  float *enc_out;                    /* (B, cap_src, D) */
  float *b1, *b2, *b3;               /* (B, maxL, D) */
  float *mg;                         /* (B, maxL, D) */
  float *qkvg;                       /* (B, maxL, 4D) = qkv ⊕ gate（导出期拼接的 in_gate 输出） */
  float *qh, *kh, *vh, *ah;          /* self: (B, H_SELF, maxL, D_H_SELF) */
  float *cq;                         /* cross q: (B, H_CROSS, maxL, D_H_CROSS) */
  float *scores;                     /* (B, H, maxL, maxS) */
  float *g;                          /* (B, maxL, D)：sigmoid 后的 gate（从 qkvg/gf 拷入） */
  float *ff_g;                       /* (B, maxL, D_FF) */
  float *ff_gu;                      /* (B, maxL, 2·D_FF)：拼接版 MLP 的输出 */
  float *gf;                         /* (B, maxL, 2·D_QKV_CROSS)：cross 拼接投影的输出 */
} mt_ctx_t;

mt_model_t *mt_model_create(void);
void        mt_model_destroy(mt_model_t *m);

mt_ctx_t   *mt_ctx_create(int B, int cap_src, int cap_tgt);
void        mt_ctx_destroy(mt_ctx_t *c);
void        mt_ctx_reset_kv(mt_ctx_t *c);

void mt_encode(mt_model_t *m, mt_ctx_t *c, const int *src_ids, const uint8_t *src_pad,
               int B, int L);
void mt_decode_step(mt_model_t *m, mt_ctx_t *c, const int *tgt_ids, int B, int L,
                    float *logits);

#endif /* MT_MODEL_H */
