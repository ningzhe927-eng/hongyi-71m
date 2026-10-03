#include "prof.h"
#include <stdio.h>
#include <stdlib.h>

int mt_prof_on = 0;
int mt_prof_phase = 0;
double mt_prof_t[PROF_N];
double mt_prof_td[PROF_N];
long mt_prof_steps = 0;
static const char *NMS[PROF_N] = {"embed", "rmsnorm", "self_attn", "cross_attn", "swiglu",
                                  "lm_head", "qkv_proj", "scores+softmax", "attn_out", "gate",
                                  "silu(逐元素)", "rope", "qkv_split", "gate_mul", "transpose",
                                  "out_proj", "ca_core", "log_softmax", "topk", "kv_reorder", "quant(激活)"};
void mt_prof_add(int i, double sec) {
  if (i < 0 || i >= PROF_N) return;
  mt_prof_t[i] += sec;
  if (mt_prof_phase) mt_prof_td[i] += sec;
}
void mt_prof_report(void) {
  double tot = 0; for (int i = 0; i < PROF_N; i++) tot += mt_prof_t[i];
  fprintf(stderr, "[prof] 合计 %.1f ms\n", tot * 1e3);
  for (int i = 0; i < PROF_N; i++)
    if (mt_prof_t[i] > 0)
      fprintf(stderr, "[prof]   %-14s %8.1f ms  %5.1f%%  (解码 %8.1f ms)\n", NMS[i],
              mt_prof_t[i] * 1e3, tot > 0 ? 100.0 * mt_prof_t[i] / tot : 0.0, mt_prof_td[i] * 1e3);
  fprintf(stderr, "[prof] 解码步数 %ld\n", mt_prof_steps);
}
