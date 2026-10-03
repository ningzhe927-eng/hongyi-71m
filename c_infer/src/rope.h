#ifndef MT_ROPE_H
#define MT_ROPE_H

/* 相邻维成对旋转（interleaved），与 seq2seq.py MHASelfAttention._rope 一致。
 * cos/sin 表按绝对位置索引，fp32。 */
typedef struct {
  int max_pos;
  int dh;            /* head dim */
  float *cos_t;      /* (max_pos, dh/2) */
  float *sin_t;
} mt_rope_t;

/* base：10000.0f */
int  mt_rope_init(mt_rope_t *r, int max_pos, int dh, float base);
void mt_rope_free(mt_rope_t *r);

/* x 形状 (B, H, L, dh)，就地旋转；offset = 该块首 token 的绝对位置 */
void mt_rope_apply(const mt_rope_t *r, float *x, int B, int H, int L, int offset);

#endif /* MT_ROPE_H */
