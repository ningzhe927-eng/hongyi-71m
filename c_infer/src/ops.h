#ifndef MT_OPS_H
#define MT_OPS_H

#include <stddef.h>

/* y = w * (x / sqrt(mean(x^2)+eps))  —— 全程 fp32，复刻 mhc.py RMSNorm */
void mt_rmsnorm(float *out, const float *x, const float *w, int rows, int dim, float eps);

/* gate = silu(gate) * up  （原地，n 个元素） */
void mt_silu_mul(float *gate, const float *up, int n);

/* dst = silu(gate) * up  （异址版；省掉 swiglu 里那次 memcpy） */
void mt_silu_mul2(float *dst, const float *gate, const float *up, int n);

/* x = 2 * sigmoid(x)  （原地） */
void mt_sigmoid2(float *x, int n);

/* 逐行 softmax（rows 行，每行 n 个连续元素） */
void mt_softmax_rows(float *x, int rows, int n);

/* 原地 log_softmax（单行 n 个元素） */
void mt_log_softmax(float *x, int n);
/* 行级 log_softmax：一次做完 nrows 行的三趟（与「逐行调 mt_log_softmax」逐位等价）。
 * ridx==NULL ⇒ 行 0..nrows-1；否则是稀疏行表。⚠ 必须从**串行上下文**调用（外层若在
 * omp parallel 内，嵌套默认关闭 ⇒ 退化为串行 —— 见 decode.c 批内 beam 的改法）。 */
void mt_log_softmax_rows(float *x, int n, const int *ridx, int nrows);

/* argmax —— 并列取最小下标（复刻 torch.argmax） */
int mt_argmax(const float *x, int n);

/* 取前 k 大：val 降序；并列时 idx 升序（稳定，复刻 torch.topk） */
void mt_topk(const float *x, int n, int k, float *val, int *idx);

void mt_add_inplace(float *dst, const float *src, size_t n);

#endif /* MT_OPS_H */
