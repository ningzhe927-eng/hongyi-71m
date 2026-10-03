#ifndef MT_DECODE_H
#define MT_DECODE_H

#include "model.h"

/* Beam search（语义逐行对齐 litmamba3/scripts/beam_decode_kv.py）：
 *   - 累加 log_softmax 对数概率（不是 raw logits）
 *   - 序列级去重；EOS 进 finished，分数按 ((5+L)/6)^lp_alpha 归一（L = 不含 bos 的长度）
 *   - 结束：finished 数 >= K，或没有新 beam
 *   - 每步按父下标重排 self-KV，batch 补齐到 K 保持形状
 * 返回生成的 token 数（不含 tgt_bos）；>=0 成功，<0 失败。
 *
 * 要求：ctx 以 B == beam 创建，cap_tgt >= max_new+1；src 只有 1 行。 */
int mt_beam_search(mt_model_t *m, mt_ctx_t *c,
                   const int *src_ids, const uint8_t *src_pad, int L,
                   int tgt_bos, int beam, int max_new, float lp_alpha,
                   int tgt_eos, int tgt_pad,
                   int *out, int out_cap,
                   /* 🆕 n-best（opt-in, 2026-10-03）：nbest>0 且 nb_* 非空时，
                    *   额外把 beam 收集到的 finished 序列按归一化分数降序写入 nb_*（最多 nbest 条）。
                    *   nbest=0 ⇒ 与旧版**逐位一致**（不分配、不写入）。返回写入条数。 */
                   int nbest, int *nb_out, int *nb_lens, float *nb_scores, int nb_stride);

/* 批内 B 行同时贪心解码（每行走自己的 KV 行）。out 为 B*max_new，outlen 为 B 个长度。
 * 结束的行继续喂 EOS 以保持各行 past 对齐（不影响其已生成结果）。 */
void mt_greedy_batch(mt_model_t *m, mt_ctx_t *c, int B, int max_new,
                     int tgt_bos, int tgt_eos, int tgt_pad,
                     int *out, int *outlen);

/* 🆕 批内 beam（2026-10-03）：N 句 × K beam 行一次前向；输出应与逐句 beam 逐位一致。
 *   N=1 时等价于 mt_beam_search（但走批量布局）；要求 ctx 的 B == N*K。*/
int mt_beam_search_batch(mt_model_t *m, mt_ctx_t *c,
                         const int *src_ids, const uint8_t *src_pad, const int *Ls, int Lmax,
                         int N, const int *tgt_bos_per, int beam, int max_new, float lp_alpha,
                         int tgt_eos, int tgt_pad,
                         int *out, int out_cap, int *outlens);

#endif /* MT_DECODE_H */
