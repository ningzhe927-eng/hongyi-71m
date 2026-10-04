#include "decode.h"
#include "ops.h"
#include "prof.h"
#include "config.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define D_D D_MODEL
#define HS  N_HEADS_SELF
#define DS  D_H_SELF
#define HC  N_HEADS_CROSS
#define DC  D_H_CROSS
#define V   VOCAB_SIZE

/* self-KV 行重排（k/v 形状 (B, HS, cap, DS)）
 * 🔴 2026-10-02：原来每步把**整段 cap_tgt（65 个位置）**的 K/V 全搬一遍（K·HS·cap·DS ×2 ×3 层
 *   ≈ 1.9 MB/步，实测 434 µs/步 = 4.6 GB/s 的低效带宽）。但解码只用得到 `c->L[i].len` 这个前缀
 *   （self_attn 只读 [0, past+L)）⇒ 只搬前缀，流量降 ~9×。**不读的东西不搬，数值不受影响。**
 *   ⚠ 布局是 (b, h, l, d)：前缀不是连续的，所以按 h 切段搬（每段 len·DS 个 float）。 */
static float *g_kvr_buf = NULL;   static size_t g_kvr_cap = 0;
static void self_kv_reorder(mt_ctx_t *c, const int *idx, int K) {
  const int cap = c->cap_tgt;
  for (int i = 0; i < N_DEC; i++) {
    const int len = c->L[i].len;
    if (len <= 0 || len > cap) continue;
    const size_t seg = (size_t)len * DS;                 /* 一个 (b,h) 段 */
    const size_t used = (size_t)HS * seg;                /* 一个 beam 的前缀 */
    const size_t need = 2 * (size_t)K * used;
    if (need > g_kvr_cap) {                              /* 复用缓冲区，别每步 malloc */
      free(g_kvr_buf);
      g_kvr_buf = (float *)malloc(need * sizeof(float));
      g_kvr_cap = g_kvr_buf ? need : 0;
    }
    if (!g_kvr_buf) return;
    float *tmp = g_kvr_buf, *tk = tmp, *tv = tmp + (size_t)K * used;
    float *k = c->L[i].k, *v = c->L[i].v;
    for (int b = 0; b < K; b++)
      for (int h = 0; h < HS; h++) {
        const size_t off = ((size_t)b * HS + h) * cap * DS;
        const size_t po = ((size_t)b * HS + h) * seg;
        memcpy(tk + po, k + off, seg * sizeof(float));
        memcpy(tv + po, v + off, seg * sizeof(float));
      }
    for (int b = 0; b < K; b++) {
      const int p = idx[b];
      for (int h = 0; h < HS; h++) {
        const size_t off = ((size_t)b * HS + h) * cap * DS;
        const size_t po = ((size_t)p * HS + h) * seg;
        memcpy(k + off, tk + po, seg * sizeof(float));
        memcpy(v + off, tv + po, seg * sizeof(float));
      }
    }
  }
}

/* 把 B=1 的 enc_out / cross-KV 复制到 K 行 */
static void expand_cross(mt_ctx_t *c, int K) {
  const int L = c->L_src;
  for (int b = 1; b < K; b++)
    memcpy(c->enc_out + (size_t)b * L * D_D, c->enc_out, (size_t)L * D_D * sizeof(float));
  /* 🔴 cross-KV 的行步长是 **L_src（本句实际长度）**，不是 cap_src：
   *   `build_cross_kv` 是按 L_src 打包的（见 model.c 的 t_lhd2hld 调用）。
   *   改"按实际长度编码"之前两者都等于 cap ⇒ 这个错位一直没暴露；
   *   L_src != cap_src 后，beam 1..K-1 会读到错位的 K/V（实测 50 句里 21 句不同）。 */
  const size_t ckn = (size_t)HC * c->L_src * DC;
  for (int i = 0; i < N_DEC; i++)
    for (int b = 1; b < K; b++) {
      memcpy(c->L[i].ck + (size_t)b * ckn, c->L[i].ck, ckn * sizeof(float));
      memcpy(c->L[i].cv + (size_t)b * ckn, c->L[i].cv, ckn * sizeof(float));
    }
}

static inline float len_pen(int len, float a) {
  return (float)pow((5.0 + (double)len) / 6.0, (double)a);
}

int mt_beam_dbg = 0;

int mt_beam_search(mt_model_t *m, mt_ctx_t *c,
                   const int *src_ids, const uint8_t *src_pad, int L,
                   int tgt_bos, int beam, int max_new, float lp_alpha,
                   int tgt_eos, int tgt_pad,
                   int *out, int out_cap,
                   int nbest, int *nb_out, int *nb_lens, float *nb_scores, int nb_stride) {
  const int K = beam;
  if (K < 1 || c->B != K || c->cap_tgt < max_new + 1) return -1;
  const int stride = max_new + 2;
  mt_beam_dbg = getenv("MT_BEAM_TRACE") != NULL;

  mt_ctx_reset_kv(c);
  mt_encode(m, c, src_ids, src_pad, 1, L);
  expand_cross(c, K);
  uint8_t *padk = (uint8_t *)malloc((size_t)K * L);
  if (!padk) return -1;
  for (int b = 0; b < K; b++) memcpy(padk + (size_t)b * L, src_pad, (size_t)L);
  c->enc_pad = padk;

  float *logits   = (float *)malloc(sizeof(float) * (size_t)K * V);
  int   *cur      = (int *)malloc(sizeof(int) * K);
  int   *seqs     = (int *)malloc(sizeof(int) * (size_t)K * stride);
  int   *lens     = (int *)calloc(K, sizeof(int));
  float *scores   = (float *)calloc(K, sizeof(float));
  int   *idxb     = (int *)malloc(sizeof(int) * K);
  int   *nseq     = (int *)malloc(sizeof(int) * (size_t)K * stride);
  int   *nlens    = (int *)malloc(sizeof(int) * K);
  float *nscores  = (float *)malloc(sizeof(float) * K);
  int   *parents  = (int *)malloc(sizeof(int) * K);
  float *tv       = (float *)malloc(sizeof(float) * K);
  int   *ti       = (int *)malloc(sizeof(int) * K);
  int   *fin_seq  = (int *)malloc(sizeof(int) * stride);
  /* 🆕 n-best 收集（opt-in；nbest<=0 时为 NULL ⇒ 默认路径不分配、不写） */
  const int fin_cap = (nbest > 0) ? (K + 2) : 0;
  int   *fin_all  = fin_cap ? (int *)malloc(sizeof(int) * (size_t)fin_cap * stride) : NULL;
  float *fin_pen  = fin_cap ? (float *)malloc(sizeof(float) * fin_cap) : NULL;
  int   *fin_len  = fin_cap ? (int *)malloc(sizeof(int) * fin_cap) : NULL;
  int    fin_ct   = 0;
  char  *used     = (char *)malloc((size_t)K * K + K + 1);   /* 候选去重标记 */
  if (!logits || !cur || !seqs || !lens || !scores || !idxb || !nseq || !nlens ||
      !nscores || !parents || !tv || !ti || !fin_seq || !used) return -1;

  /* 候选池（n_live*K 个） */
  const int cap_cand = K * K + 1;
  float *cs = (float *)malloc(sizeof(float) * cap_cand);
  int   *cp = (int *)malloc(sizeof(int) * cap_cand);
  int   *ct = (int *)malloc(sizeof(int) * cap_cand);
  int   *accp = (int *)malloc(sizeof(int) * (K + 2));
  int   *acct = (int *)malloc(sizeof(int) * (K + 2));
  if (!cs || !cp || !ct || !accp || !acct) return -1;

  /* prefill：K 行都喂 tgt_bos */
  for (int b = 0; b < K; b++) cur[b] = tgt_bos;
  mt_decode_step(m, c, cur, K, 1, logits);
  for (int b = 0; b < K; b++) { seqs[(size_t)b * stride] = tgt_bos; lens[b] = 1; scores[b] = 0.f; }

  int n_live = 1, n_finished = 0, best_fin_len = -1;
  float best_fin = -1e30f;

  for (int step = 0; step < max_new; step++) {
    { MT_PROF_B0();
      /* b1 路径是**纯串行 for**（不在 omp parallel 内）⇒ 行内的并行区是真并行。
       * 改用行级版本：一个 region 覆盖 K 行的全部三趟（原来是每行各自 fork，且第
       * 三趟的 logf 因 ops.c 阈值单位写错从未并行过，见 ops.c 注释）。 */
      mt_log_softmax_rows(logits, V, NULL, K);
      MT_PROF_E0(PROF_LOGSOFT); }

    int nc = 0;
    { MT_PROF_B0();
      for (int b = 0; b < n_live; b++) {
        mt_topk(logits + (size_t)b * V, V, K, tv, ti);
        for (int k = 0; k < K; k++) { cs[nc] = scores[b] + tv[k]; cp[nc] = b; ct[nc] = ti[k]; nc++; }
      }
      MT_PROF_E0(PROF_TOPK); }
    /* 按分数降序（并列保持原序 ⇒ 稳定） */
    for (int i = 0; i < nc; i++) {
      int bi = i;
      for (int j = i + 1; j < nc; j++) if (cs[j] > cs[bi]) bi = j;
      if (bi != i) {
        float ts = cs[i]; cs[i] = cs[bi]; cs[bi] = ts;
        int tp = cp[i]; cp[i] = cp[bi]; cp[bi] = tp;
        int tt = ct[i]; ct[i] = ct[bi]; ct[bi] = tt;
      }
    }

    if (mt_beam_dbg) {
      fprintf(stderr, "[step %d] n_live=%d n_finished=%d\n", step, n_live, n_finished);
      for (int ci = 0; ci < nc && ci < 4; ci++)
        fprintf(stderr, "   cand %d: sc=%.4f parent=%d tok=%d\n", ci, cs[ci], cp[ci], ct[ci]);
    }
    int n_new = 0, n_acc = 0;
    for (int ci = 0; ci < nc; ci++) {
      int p = cp[ci], tok = ct[ci];
      float sc = cs[ci];
      /* 与「本步已接受」的候选（含 finished）比完整序列去重 */
      int dup = 0;
      for (int q = 0; q < n_acc; q++) {
        if (acct[q] != tok) continue;
        int p2 = accp[q];
        if (lens[p2] != lens[p]) continue;
        const int *A = seqs + (size_t)p * stride;
        const int *B2 = seqs + (size_t)p2 * stride;
        int same = 1;
        for (int z = 0; z < lens[p]; z++) if (A[z] != B2[z]) { same = 0; break; }
        if (same) { dup = 1; break; }
      }
      if (dup) continue;
      accp[n_acc] = p; acct[n_acc] = tok; n_acc++;
      (void)used;

      if (mt_beam_dbg) fprintf(stderr, "   take sc=%.4f parent=%d tok=%d -> %s\n", sc, p, tok, (tok==tgt_eos?"EOS":"new"));
      if (tok == tgt_eos || tok == tgt_pad) {
        float pen = sc / len_pen(lens[p], lp_alpha);
        n_finished++;
        if (fin_cap && fin_ct < fin_cap) {
          memcpy(fin_all + (size_t)fin_ct * stride, seqs + (size_t)p * stride,
                 (size_t)lens[p] * sizeof(int));
          fin_pen[fin_ct] = pen;
          fin_len[fin_ct] = lens[p];
          fin_ct++;
        }
        if (pen > best_fin) {
          best_fin = pen;
          memcpy(fin_seq, seqs + (size_t)p * stride, (size_t)lens[p] * sizeof(int));
          best_fin_len = lens[p];
        }
      } else {
        memcpy(nseq + (size_t)n_new * stride, seqs + (size_t)p * stride, (size_t)lens[p] * sizeof(int));
        nseq[(size_t)n_new * stride + lens[p]] = tok;
        nlens[n_new] = lens[p] + 1;
        nscores[n_new] = sc;
        parents[n_new] = p;
        n_new++;
      }
      if (n_new + n_finished >= K) break;
    }

    if (n_finished >= K || n_new == 0) break;

    { MT_PROF_B0();
      for (int b = 0; b < K; b++) idxb[b] = (b < n_new) ? parents[b] : 0;
      self_kv_reorder(c, idxb, K);
      MT_PROF_E0(PROF_KVREORDER); }

    for (int b = 0; b < n_new; b++) {
      memcpy(seqs + (size_t)b * stride, nseq + (size_t)b * stride, (size_t)nlens[b] * sizeof(int));
      lens[b] = nlens[b];
      scores[b] = nscores[b];
    }
    for (int b = n_new; b < K; b++) {              /* 填充槽（输出被丢弃） */
      memcpy(seqs + (size_t)b * stride, seqs, (size_t)lens[0] * sizeof(int));
      lens[b] = lens[0];
      scores[b] = 0.f;
    }
    for (int b = 0; b < K; b++) cur[b] = seqs[(size_t)b * stride + lens[b] - 1];
    mt_decode_step(m, c, cur, K, 1, logits);
    n_live = n_new;
  }

  int nout = 0;
  if (best_fin_len > 0) {
    for (int i = 1; i < best_fin_len && nout < out_cap; i++) out[nout++] = fin_seq[i];
  } else {
    for (int i = 1; i < lens[0] && nout < out_cap; i++) out[nout++] = seqs[(size_t)i];
  }

  /* 🆕 写 n-best（默认 nbest<=0 ⇒ 直接跳过，行为与旧版一致） */
  int nb_written = 0;
  if (nbest > 0 && nb_out && nb_lens && nb_scores && fin_ct > 0) {
    int *ord = (int *)malloc(sizeof(int) * fin_ct);
    if (ord) {
      for (int i = 0; i < fin_ct; i++) ord[i] = i;
      for (int i = 0; i < fin_ct; i++)         /* 简单插入排序（fin_ct ≤ K+2） */
        for (int j = i + 1; j < fin_ct; j++)
          if (fin_pen[ord[j]] > fin_pen[ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
      int m = nbest < fin_ct ? nbest : fin_ct;
      for (int k = 0; k < m; k++) {
        int id = ord[k], n = 0;
        for (int i = 1; i < fin_len[id] && n < nb_stride; i++)
          nb_out[(size_t)k * nb_stride + n++] = fin_all[(size_t)id * stride + i];
        nb_lens[k] = n;
        nb_scores[k] = fin_pen[id];
      }
      nb_written = m;
      free(ord);
    }
  }
  free(fin_all); free(fin_pen); free(fin_len);
  free(logits); free(cur); free(seqs); free(lens); free(scores); free(idxb);
  free(nseq); free(nlens); free(nscores); free(parents); free(tv); free(ti);
  free(fin_seq); free(used); free(cs); free(cp); free(ct); free(accp); free(acct);
  free(padk);
  return nout;
}

void mt_greedy_batch(mt_model_t *m, mt_ctx_t *c, int B, int max_new,
                     int tgt_bos, int tgt_eos, int tgt_pad,
                     int *out, int *outlen) {
  if (c->B != B || c->cap_tgt < max_new + 1) return;
  int *nt = (int *)malloc(sizeof(int) * B);
  int *done = (int *)calloc(B, sizeof(int));
  float *logits = (float *)malloc(sizeof(float) * (size_t)B * V);
  if (!nt || !done || !logits) { free(nt); free(done); free(logits); return; }
  for (int b = 0; b < B; b++) { nt[b] = tgt_bos; outlen[b] = 0; done[b] = 0; }
  mt_decode_step(m, c, nt, B, 1, logits);
  for (int step = 0; step < max_new; step++) {
    int alive = 0;
    for (int b = 0; b < B; b++) {
      if (done[b]) { nt[b] = tgt_eos; continue; }
      int nx = mt_argmax(logits + (size_t)b * V, V);
      if (nx == tgt_eos || nx == tgt_pad) { done[b] = 1; nt[b] = tgt_eos; continue; }
      out[(size_t)b * max_new + outlen[b]++] = nx;
      nt[b] = nx; alive++;
    }
    if (alive == 0) break;
    mt_decode_step(m, c, nt, B, 1, logits);
  }
  free(nt); free(done); free(logits);
}

/* ============================================================================
 * 🆕 批内 beam（2026-10-03）：N 个句子各占 K=beam 行，**一次前向跑完 N·K 行**。
 *   目标：① 输出与「逐句跑 N 次 mt_beam_search」**逐位一致**；② 吞吐 ≥1.3×。
 *   做法：每步对 N·K 行做一次 mt_decode_step，宿主侧**逐句**做候选扩展/去重/结束判定（与原函数同序同判据），
 *         再做**组内** self-KV 重排。已结束的句子保持喂 EOS（与批内贪心一致，不影响其结果）。
 *   要求：ctx 以 B == N*K 创建；cap_tgt >= max_new+1。
 *   返回 0 成功，<0 失败。out 为 N*out_cap（每句一段），outlens 为 N 个长度。
 * ==========================================================================*/
static void expand_cross_groups(mt_ctx_t *c, int N, int K) {
  const int L = c->L_src;
  /* enc_out / cross-KV：第 s 句（编码时写在**行 s**）复制到它的 K 行 s*K..s*K+K-1。
   *   🔴 必须**包含 k=0**：只有 s=0 时 s*K 才恰好等于 s；s≥1 时行 s*K 是未初始化的
   *   （编码只写了行 0..N-1）——漏掉 k=0 会让第 2 句起读到垃圾（输出乱码）。
   *   倒序 s 保证源行在被覆盖前读取。*/
  for (int s = N - 1; s >= 0; s--)
    for (int k = 0; k < K; k++)
      memcpy(c->enc_out + ((size_t)s * K + k) * L * D_D,
             c->enc_out + (size_t)s * L * D_D, (size_t)L * D_D * sizeof(float));
  const size_t ckn = (size_t)HC * c->L_src * DC;
  for (int i = 0; i < N_DEC; i++) {
    float *ck = c->L[i].ck, *cv = c->L[i].cv;
    for (int s = N - 1; s >= 0; s--)
      for (int k = 0; k < K; k++) {
        memcpy(ck + ((size_t)s * K + k) * ckn, ck + (size_t)s * ckn, ckn * sizeof(float));
        memcpy(cv + ((size_t)s * K + k) * ckn, cv + (size_t)s * ckn, ckn * sizeof(float));
      }
  }
  if (getenv("MT_BBEAM_CK")) {                 /* 调试：各层 ck 每行前 32 个 float 的校验和 */
    fprintf(stderr, "[ck] N=%d K=%d R=%d HC=%d DC=%d L_src=%d ckn=%zu\n", N, K, N * K, HC, DC, c->L_src, ckn);
    for (int i = 0; i < N_DEC && i < 1; i++) {
      for (int r = 0; r < N * K; r++) {
        double s2 = 0; const float *p2 = c->L[i].ck + (size_t)r * ckn;
        for (int q = 0; q < 32 && q < (int)ckn; q++) s2 += (double)p2[q];
        fprintf(stderr, "  layer%d row%d sum=%.6f\n", i, r, s2);
      }
    }
  }

}

/* 组内重排：行 (s*K+b) ← 行 (s*K+idx[s*K+b])，逐句独立。
 * 🆕 P3 降本（数值不变）：
 * ① done 组整组跳过——行 base 每步都映射到自己（等价于自拷，内容冻结）；
 *   行 base+1..K-1 变成 stale，但其输出只喂被丢弃的 logits（done 句不进候选、
 *   cur=tgt_eos），且 self/cross attn 跨组无引用（idxg 恒在组内）⇒ 不影响任何输出。
 * ② copy-out 自映射跳过（目的=源，memcpy 无操作）。
 * ③ stage 消去：tmp[p] 只被「映射到 p 的行」读；自映射那次读已随 ② 消去，
 *   故 cnt[p]−(p 自映射?1:0)==0 的行无需 stage。 */
static void self_kv_reorder_groups(mt_ctx_t *c, const int *idxg, int N, int K, const int *done) {
  const int cap = c->cap_tgt;
  const int R = N * K;
  int *cnt = (int *)malloc(sizeof(int) * K);
  if (!cnt) return;
  for (int i = 0; i < N_DEC; i++) {
    const int len = c->L[i].len;
    if (len <= 0 || len > cap) continue;
    const size_t seg = (size_t)len * DS;
    const size_t used = (size_t)HS * seg;
    const size_t need = 2 * (size_t)R * used;
    if (need > g_kvr_cap) {
      free(g_kvr_buf);
      g_kvr_buf = (float *)malloc(need * sizeof(float));
      g_kvr_cap = g_kvr_buf ? need : 0;
    }
    if (!g_kvr_buf) { free(cnt); return; }
    float *tmp = g_kvr_buf, *tk = tmp, *tv = tmp + (size_t)R * used;
    float *k = c->L[i].k, *v = c->L[i].v;
    for (int s = 0; s < N; s++) {
      if (done[s]) continue;                       /* ① done 组整组跳过 */
      const int base = s * K;
      for (int b = 0; b < K; b++) cnt[b] = 0;
      for (int b = 0; b < K; b++) cnt[idxg[base + b] - base]++;
      for (int b = 0; b < K; b++) {                /* copy-in（仅会被读的行） */
        if (cnt[b] - (idxg[base + b] == base + b ? 1 : 0) <= 0) continue;
        for (int h = 0; h < HS; h++) {
          const size_t off = ((size_t)(base + b) * HS + h) * cap * DS;
          const size_t po  = ((size_t)(base + b) * HS + h) * seg;
          memcpy(tk + po, k + off, seg * sizeof(float));
          memcpy(tv + po, v + off, seg * sizeof(float));
        }
      }
      for (int b = 0; b < K; b++) {                /* copy-out（跳过自映射） */
        const int p = idxg[base + b];
        if (p == base + b) continue;
        for (int h = 0; h < HS; h++) {
          const size_t off = ((size_t)(base + b) * HS + h) * cap * DS;
          const size_t po  = ((size_t)p * HS + h) * seg;
          memcpy(k + off, tk + po, seg * sizeof(float));
          memcpy(v + off, tv + po, seg * sizeof(float));
        }
      }
    }
  }
  free(cnt);
}

int mt_beam_search_batch(mt_model_t *m, mt_ctx_t *c,
                         const int *src_ids, const uint8_t *src_pad, const int *Ls, int Lmax,
                         int N, const int *tgt_bos_per, int beam, int max_new, float lp_alpha,
                         int tgt_eos, int tgt_pad,
                         int *out, int out_cap, int *outlens) {
  const int K = beam, R = N * K;
  (void)Ls;                                  /* 逐句实际长度：批内已用 Lmax + per-row pad 表达 */
  if (K < 2 || N < 1 || c->B != R || c->cap_tgt < max_new + 1) return -1;
  const int stride = max_new + 2;
  mt_beam_dbg = getenv("MT_BEAM_TRACE") != NULL;
  /* 🆕 尾浪费仪表（MT_TAIL_STAT=1 才打印）：整批要跑到 all_done 才结束，已结束的句子
   *   的 K 行仍每步参与前向 ⇒ 统计"付了的行数" vs "真正有用的行数"，用来判断值不值得
   *   做动态批次（refill）。**只测量，不改调度、不改数值**。 */
  const int tail_stat = getenv("MT_TAIL_STAT") != NULL;
  long long tail_steps = 0, tail_live = 0;

  mt_ctx_reset_kv(c);
  mt_encode(m, c, src_ids, src_pad, N, Lmax);
  expand_cross_groups(c, N, K);
  /* 🔴 关键修复：cross_attn 用 `c->enc_pad + b*L_src` **按行**取 padding 掩码，
   *   而编码只提供 N 行 ⇒ 必须把每句的 pad **复制到它的 K 行**（否则第 1..K-1 行越界读到垃圾）。*/
  uint8_t *padr = (uint8_t *)malloc((size_t)R * Lmax);
  if (!padr) return -1;
  for (int s2 = 0; s2 < N; s2++)
    for (int k = 0; k < K; k++)
      memcpy(padr + ((size_t)s2 * K + k) * Lmax, src_pad + (size_t)s2 * Lmax, (size_t)Lmax);
  c->enc_pad = padr;

  float *logits = (float *)malloc(sizeof(float) * (size_t)R * V);
  int   *cur    = (int *)malloc(sizeof(int) * R);
  int   *seqs   = (int *)malloc(sizeof(int) * (size_t)R * stride);
  int   *lens   = (int *)calloc(R, sizeof(int));
  float *scores = (float *)calloc(R, sizeof(float));
  int   *idxg   = (int *)malloc(sizeof(int) * R);
  int   *nseq   = (int *)malloc(sizeof(int) * (size_t)R * stride);
  int   *nlens  = (int *)calloc(R, sizeof(int));
  float *nscores= (float *)calloc(R, sizeof(float));
  int   *parents= (int *)malloc(sizeof(int) * (size_t)N * K);
  float *tvb    = (float *)malloc(sizeof(float) * (size_t)R * K);
  int   *tib    = (int *)malloc(sizeof(int) * (size_t)R * K);
  int   *rlist  = (int *)malloc(sizeof(int) * R);
  int   *slot   = (int *)malloc(sizeof(int) * N);
  int   *best_seq = (int *)malloc(sizeof(int) * (size_t)N * stride);
  int   *best_len = (int *)calloc(N, sizeof(int));
  float *best_sc  = (float *)malloc(sizeof(float) * N);
  int   *n_live   = (int *)calloc(N, sizeof(int));
  int   *n_fin    = (int *)calloc(N, sizeof(int));
  int   *done     = (int *)calloc(N, sizeof(int));
  const int cap_cand = K * K + 1;
  float *cs = (float *)malloc(sizeof(float) * cap_cand);
  int   *cp = (int *)malloc(sizeof(int) * cap_cand);
  int   *ct = (int *)malloc(sizeof(int) * cap_cand);
  int   *accp = (int *)malloc(sizeof(int) * (K + 2));
  int   *acct = (int *)malloc(sizeof(int) * (K + 2));
  if (!logits || !cur || !seqs || !lens || !scores || !idxg || !nseq || !nlens || !nscores ||
      !parents || !tvb || !tib || !rlist || !slot || !best_seq || !best_len || !best_sc || !n_live || !n_fin ||
      !done || !cs || !cp || !ct || !accp || !acct) return -1;
  for (int s = 0; s < N; s++) { best_sc[s] = -1e30f; best_len[s] = -1; n_live[s] = 1; n_fin[s] = 0; }

  /* prefill：每句的 K 行都喂该句的 tgt_bos（一次前向覆盖 R 行） */
  for (int s = 0; s < N; s++)
    for (int b = 0; b < K; b++) cur[s * K + b] = tgt_bos_per[s];
  mt_decode_step(m, c, cur, R, 1, logits);
  for (int s = 0; s < N; s++)
    for (int b = 0; b < K; b++) {
      seqs[(size_t)(s * K + b) * stride] = tgt_bos_per[s];
      lens[s * K + b] = 1; scores[s * K + b] = 0.f;
    }

  const int btr = getenv("MT_BBEAM_TRACE") != NULL;
  for (int step = 0; step < max_new; step++) {
    if (btr && step < 3) {
      fprintf(stderr, "[bstep %d] 每行 argmax:", step);
      for (int r = 0; r < R; r++) fprintf(stderr, " %d", mt_argmax(logits + (size_t)r * V, V));
      fprintf(stderr, "\n");
    }
    /* 行表：与旧逐行版处理集合完全一致（!done[s] && b<n_live[s]，s 升序 b 升序）。
     * 🔴 逐位一致：每行结果只依赖本行输入、行缓冲互不重叠；mt_log_softmax/mt_topk
     *   本体未动（其内部 OMP 区嵌套下退化为串行，但行内运算顺序不变）⇒ 跨行并行不改值。
     *   ⚠ 若将来开启 OMP 嵌套，行内并行会真嵌套——数值仍对，性能塌陷。 */
    int nrl = 0;
    for (int s = 0; s < N; s++) {
      slot[s] = nrl;
      if (!done[s]) for (int b = 0; b < n_live[s]; b++) rlist[nrl++] = s * K + b;
    }
    if (tail_stat) { tail_steps++; tail_live += nrl; }   /* 尾浪费仪表（只测量，不改调度） */
    { MT_PROF_B0();
      /* 🔴 原来是「外层按行并行 + 行内 mt_log_softmax」：行内的三趟因嵌套默认关闭
       *   **全部退化成串行**，等于行级并行只用上了 spaced 一小部分；且 schedule(static)
       *   在 nrl 非线程数倍数时有尾倾斜（nrl=17、T=8 ⇒ 关键路径 3 行 = 理想 1.5×）。
       *   改成一次调用覆盖全部行的三趟（第三趟用 (行,块) 扁平任务）⇒ 完美均衡。
       *   数值：每行走的还是同一套 mt_ls_row（行间独立）⇒ 逐位不变。 */
      mt_log_softmax_rows(logits, V, rlist, nrl);
      MT_PROF_E0(PROF_LOGSOFT); }
    { MT_PROF_B0();
      #pragma omp parallel for schedule(static) if(nrl >= 2)
      for (int i = 0; i < nrl; i++)
        mt_topk(logits + (size_t)rlist[i] * V, V, K, tvb + (size_t)i * K, tib + (size_t)i * K);
      MT_PROF_E0(PROF_TOPK); }
    if (mt_beam_dbg) { int nd = 0; for (int s = 0; s < N; s++) nd += done[s]; fprintf(stderr, "[bstep %d] done=%d/%d\n", step, nd, N); }

    int all_done = 1;
    for (int s = 0; s < N; s++) {
      const int base = s * K;
      if (done[s]) { for (int b = 0; b < K; b++) idxg[base + b] = base; continue; }
      all_done = 0;
      /* --- 候选池（与原函数同序：按 b 顺序、每行 top-K） --- */
      int nc = 0;
      /* topk 已在上面的合批段算好（tvb/tib）；这里只拼候选池，读取顺序（b 升序 k 升序）
       * 与旧版 mt_topk 逐行调用完全一致 ⇒ cs 的浮点加法序列不变 ⇒ 候选/排序/去重逐位一致。 */
      for (int b = 0; b < n_live[s]; b++)
        for (int k = 0; k < K; k++) {
          const size_t sl = (size_t)(slot[s] + b) * K + k;
          cs[nc] = scores[base + b] + tvb[sl]; cp[nc] = b; ct[nc] = tib[sl]; nc++;
        }
      for (int i = 0; i < nc; i++) {          /* 稳定降序（与单句版同写法） */
        int bi = i;
        for (int j = i + 1; j < nc; j++) if (cs[j] > cs[bi]) bi = j;
        if (bi != i) {
          float ts = cs[i]; cs[i] = cs[bi]; cs[bi] = ts;
          int tp = cp[i]; cp[i] = cp[bi]; cp[bi] = tp;
          int tt = ct[i]; ct[i] = ct[bi]; ct[bi] = tt;
        }
      }
      int n_new = 0, n_acc = 0;
      for (int ci = 0; ci < nc; ci++) {
        int p = cp[ci], tok = ct[ci]; float sc = cs[ci];
        int dup = 0;
        for (int q = 0; q < n_acc; q++) {
          if (acct[q] != tok) continue;
          int p2 = accp[q];
          if (lens[base + p2] != lens[base + p]) continue;
          const int *A = seqs + (size_t)(base + p) * stride;
          const int *B2 = seqs + (size_t)(base + p2) * stride;
          int same = 1;
          for (int z = 0; z < lens[base + p]; z++) if (A[z] != B2[z]) { same = 0; break; }
          if (same) { dup = 1; break; }
        }
        if (dup) continue;
        accp[n_acc] = p; acct[n_acc] = tok; n_acc++;
        if (tok == tgt_eos || tok == tgt_pad) {
          float pen = sc / len_pen(lens[base + p], lp_alpha);
          n_fin[s]++;
          if (pen > best_sc[s]) {
            best_sc[s] = pen; best_len[s] = lens[base + p];
            memcpy(best_seq + (size_t)s * stride, seqs + (size_t)(base + p) * stride,
                   (size_t)lens[base + p] * sizeof(int));
          }
        } else {
          memcpy(nseq + (size_t)(base + n_new) * stride, seqs + (size_t)(base + p) * stride,
                 (size_t)lens[base + p] * sizeof(int));
          nseq[(size_t)(base + n_new) * stride + lens[base + p]] = tok;
          nlens[base + n_new] = lens[base + p] + 1;
          nscores[base + n_new] = sc;
          parents[base + n_new] = p;
          n_new++;
        }
        if (n_new + n_fin[s] >= K) break;
      }
      if (n_fin[s] >= K || n_new == 0) {       /* 该句结束（与单句版同判据） */
        done[s] = 1;
        for (int b = 0; b < K; b++) idxg[base + b] = base;
        continue;
      }
      /* 🔴 parents[] 存的是**组内相对下标**（0..K-1）⇒ 重排映射要加 base 变成绝对行号，
       *   否则第 2 句起会去读别的组的 KV（用"句子完全相同"的测试看不出来）。*/
      for (int b = 0; b < K; b++)
        idxg[base + b] = base + ((b < n_new) ? parents[base + b] : 0);
      for (int b = 0; b < n_new; b++) {
        memcpy(seqs + (size_t)(base + b) * stride, nseq + (size_t)(base + b) * stride,
               (size_t)nlens[base + b] * sizeof(int));
        lens[base + b] = nlens[base + b];
        scores[base + b] = nscores[base + b];
      }
      for (int b = n_new; b < K; b++) {         /* 填充槽（输出被丢弃） */
        memcpy(seqs + (size_t)(base + b) * stride, seqs + (size_t)base * stride,
               (size_t)lens[base] * sizeof(int));
        lens[base + b] = lens[base]; scores[base + b] = 0.f;
      }
      n_live[s] = n_new;
    }
    if (all_done) break;

    self_kv_reorder_groups(c, idxg, N, K, done);
    for (int s = 0; s < N; s++)
      for (int b = 0; b < K; b++) {
        const int base = s * K + b;
        cur[base] = done[s] ? tgt_eos : seqs[(size_t)base * stride + lens[base] - 1];
      }
    mt_decode_step(m, c, cur, R, 1, logits);
  }

  if (tail_stat && tail_steps > 0) {
    long long paid = tail_steps * (long long)R;
    fprintf(stderr, "[tail] steps=%lld live_sum=%lld paid=%lld waste=%.1f%%\n",
            tail_steps, tail_live, paid, 100.0 * (1.0 - (double)tail_live / (double)paid));
  }

  for (int s = 0; s < N; s++) {
    int n = 0;
    if (best_len[s] > 0) {
      for (int i = 1; i < best_len[s] && n < out_cap; i++)
        out[(size_t)s * out_cap + n++] = best_seq[(size_t)s * stride + i];
    }
    outlens[s] = n;
  }
  int rc = 0;
  c->enc_pad = (uint8_t *)src_pad;           /* 复原，避免悬垂指针（下次 mt_encode 也会重设）*/
  free(padr);
  free(logits); free(cur); free(seqs); free(lens); free(scores); free(idxg); free(nseq);
  free(nlens); free(nscores); free(parents); free(tvb); free(tib); free(rlist); free(slot); free(best_seq);
  free(best_len); free(best_sc); free(n_live); free(n_fin); free(done);
  free(cs); free(cp); free(ct); free(accp); free(acct);
  return rc;
}
