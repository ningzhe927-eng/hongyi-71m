#ifndef MT_PROF_H
#define MT_PROF_H

#include <time.h>

/* ⚠ 父子关系（改这里就要改 实验/prof_summary.py 的 CHILD）：
 *   self_attn 含 qkv_proj / rope / qkvsplit / scores+softmax / attn_out / gate / gatemul / transpose / outproj
 *   cross_attn 含 ca_core（beam 管理的三段不属于任何父段）
 *   swiglu 含 silu(逐元素) */
enum { PROF_EMBED, PROF_RMSNORM, PROF_SELF, PROF_CROSS, PROF_SWIGLU,
       PROF_LMHEAD, PROF_QKVPROJ, PROF_SCORES, PROF_ATTNOUT, PROF_GATE,
       PROF_SILU,          /* swiglu 里的逐元素部分（silu*up），是 swiglu 的子段 */
       PROF_ROPE, PROF_QKVSPLIT, PROF_GATEMUL, PROF_TRANSPOSE, PROF_OUTPROJ,
       PROF_CACORE,        /* cross_attn 里的 (b,h) 主循环：scores+softmax+p·v，是 cross_attn 的子段 */
       PROF_LOGSOFT, PROF_TOPK, PROF_KVREORDER,   /* beam 管理：全都在 mt_decode_step 之外 */
       PROF_QUANT,         /* GEMM 内的激活量化（quant_row_pad）：所有 GEMM 的子段 */
       PROF_N };

extern int mt_prof_on;
/* 🔴 阶段分离（2026-10-02 加）：`mt_prof_phase` 由 model.c 在 encode/decode 入口设置，
 *   `mt_prof_add` 同时往「全部」和「解码阶段」两份累加 ⇒ 可以直接读出
 *   "解码单步里 GEMM 占多少、单线程的逐元素/注意力占多少"（用来判断批量/投机解码有没有戏）。 */
extern int mt_prof_phase;          /* 0 = 编码阶段，1 = 解码阶段 */
extern double mt_prof_t[PROF_N];   /* 全部（含父子重复计数） */
extern double mt_prof_td[PROF_N];  /* 仅解码阶段 */
extern long mt_prof_steps;         /* mt_decode_step 的调用次数 = 解码步数（batch/beam 算 1 步）*/
void mt_prof_add(int i, double sec);
void mt_prof_report(void);

#define MT_PROF_B0() struct timespec _pt0; do { if (mt_prof_on) clock_gettime(CLOCK_MONOTONIC, &_pt0); } while (0)
#define MT_PROF_E0(i) do { if (mt_prof_on) { struct timespec _pt1; clock_gettime(CLOCK_MONOTONIC, &_pt1); \
    mt_prof_add((i), (_pt1.tv_sec - _pt0.tv_sec) + (_pt1.tv_nsec - _pt0.tv_nsec) * 1e-9); } } while (0)
#endif
