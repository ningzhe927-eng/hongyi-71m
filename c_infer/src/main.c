#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>

#include "config.h"
#include "model.h"
#include "ops.h"
#include "cpu_topology.h"

/* 批内 beam 的总行数上限（= batch*beam）。原来散在两处硬编码 128 ⇒ 抽成常量，
 * 免得 --batch auto 的选档与建 ctx 前的裁剪打架。 */
#define MT_ROWS_MAX 128
#include "tokenizer_bpe.h"
#include "decode.h"
#include "prof.h"
#include "postproc.h"
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void report_rss(void) {
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0)
    fprintf(stderr, "RSS_KB=%ld\n", (long)ru.ru_maxrss);
}

static void print_info(void) {
  printf("mt50m C inference\n");
  printf("  d_model=%d vocab=%d enc/dec=%d/%d d_ff=%d\n", D_MODEL, VOCAB_SIZE, N_ENC, N_DEC, D_FF);
  printf("  self %dx%d  cross %dx%d\n", N_HEADS_SELF, D_H_SELF, N_HEADS_CROSS, D_H_CROSS);
  printf("  ids pad/from_zh/from_en/to_zh/to_en/src_eos/tgt_eos = %d/%d/%d/%d/%d/%d/%d\n",
         ID_PAD, ID_FROM_ZH, ID_FROM_EN, ID_TO_ZH, ID_TO_EN, ID_SRC_EOS, ID_TGT_EOS);
}

static int read_ints(const char *path, int *out, int cap) {
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
  int n = 0, v;
  while (n < cap && fscanf(f, "%d", &v) == 1) out[n++] = v;
  fclose(f);
  return n;
}

static int read_bytes(const char *path, uint8_t *out, int cap) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  int n = 0, v;
  while (n < cap && fscanf(f, "%d", &v) == 1) out[n++] = (uint8_t)v;
  fclose(f);
  return n;
}

static void dump_floats(const char *path, const float *p, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
  fwrite(p, sizeof(float), n, f);
  fclose(f);
  fprintf(stderr, "  wrote %s (%zu floats)\n", path, n);
}


/* 🆕 分句再译（2026-10-03，opt-in）：把源句在**句末标点**处切开，返回每段的 (起始偏移, 长度)。
 *   中文：。！？；  英文：. ! ? 后跟空白且下一非空字符为大写字母（避免 Mr. 之类误切）
 *   太短的段（<4 字节）并入上一段；最多 cap 段。**单句输入返回 1 段** ⇒ 调用方走原路径，逐位不变。
 *   实测（flores zh→en）：多句源 BLEU 18.7→24.2、全集 25.9→26.5；单句集（iwslt/tatoeba）无变化。*/
static int mt_split_sent(const char *s, int n, int *off, int *len, int cap) {
  int cnt = 0, start = 0;
  for (int i = 0; i < n; ) {
    unsigned char c = (unsigned char)s[i];
    int adv = 1, cut = 0;
    if (c == 0xE3 && i + 2 < n && (unsigned char)s[i + 1] == 0x80 && (unsigned char)s[i + 2] == 0x82) { adv = 3; cut = 1; }  /* 。 */
    else if (c == 0xEF && i + 2 < n && (unsigned char)s[i + 1] == 0xBC &&
             ((unsigned char)s[i + 2] == 0x81 || (unsigned char)s[i + 2] == 0x9F || (unsigned char)s[i + 2] == 0x9B)) { adv = 3; cut = 1; }  /* ！？； */
    else if ((c == '.' || c == '!' || c == '?') && i + 1 < n && (s[i + 1] == ' ' || s[i + 1] == '\t')) {
      int j = i + 1;
      while (j < n && (s[j] == ' ' || s[j] == '\t')) j++;
      if (j < n && s[j] >= 'A' && s[j] <= 'Z') cut = 1;      /* 英文句子边界 */
    }
    i += adv;
    if (cut) {
      int seg = i - start;
      if (seg >= 4) {
        if (cnt == 0 || seg >= 4) { off[cnt] = start; len[cnt] = seg; cnt++; }
        start = i;
        if (cnt >= cap - 1) break;
      }
    }
  }
  if (start < n) { off[cnt] = start; len[cnt] = n - start; cnt++; }
  return cnt;
}

int main(int argc, char **argv) {
#if !defined(MT_QUANT8)
  /* 🔴 OpenBLAS 在**库构造期**就按逻辑 CPU 数建线程池（本机 64，无视 8 核 cgroup 配额），
   *    而可执行文件的 constructor 晚于共享库 ⇒ 只能靠环境变量，且必须在进程 exec 前设好。
   *    这里做一次性 re-exec 注入（未显式设置时）。 */
  if (getenv("OPENBLAS_NUM_THREADS") == NULL) {
    /* 🆕 与下面 mt_threads_resolve 用的是**同一套**decision（原来这里自己扫 argv 且默认 8，
     *   一旦 main 里的策略改了而这里没改，fp32 档会用到错误的线程数）。 */
    int want = 0, smt = 1, explicit_threads = 0;
    for (int i = 1; i + 1 < argc; i++) {
      if (!strcmp(argv[i], "--threads")) { want = atoi(argv[i + 1]); explicit_threads = 1; }
      if (!strcmp(argv[i], "--smt"))     { smt   = atoi(argv[i + 1]); }
    }
    if (!explicit_threads) want = 0;              /* 未显式指定 ⇒ 交给 mt_threads_resolve（含 --smt 语义）*/
    want = mt_threads_resolve(want, smt);
    char buf[16]; snprintf(buf, sizeof buf, "%d", want);
    setenv("OPENBLAS_NUM_THREADS", buf, 1);
    execv("/proc/self/exe", argv);
  }
#endif
  const char *text_arg = NULL, *src_file = NULL;
  const char *ids_file = NULL, *pad_file = NULL, *dump_enc = NULL, *dump_logits = NULL, *tgt_file = NULL, *bpe_file = NULL;
  mt_prof_on = getenv("MT_PROF") != NULL;
  if (mt_prof_on) atexit(mt_prof_report);
  int detok = 0, bpe_nul = 0, dir = 0, dir_forced = 0, beam = 1, batch = 1;
  float lp_alpha = LP_ALPHA_DEFAULT;
  /* ⚠ 2026-10-02 修：max_new 原来硬编码 32，而 config 里的 MAX_NEW_DEFAULT 是 96。
   *   en→zh 的中文输出 token 更多，32 会在句中被砍断（看起来像"译文乱/不完整"）。*/
  /* 🆕 threads=0 ⇒ 自动（按真实可用 CPU，非 nproc；再按 MT_THREAD_CAP 封顶）。
   *   历史所有基准都显式传了 --threads 8 ⇒ 历史数字不受此改动影响。 */
  int cap = SRC_CAP_DEFAULT, max_new = MAX_NEW_DEFAULT, greedy = 1, tgt_bos = ID_TO_EN, threads = 0, repeat = 1;
  int smt = 1;                                    /* 1=用全部逻辑 CPU（默认）；0=只用物理核 */
  /* 输出侧后处理：**默认开启 recase**（2026-10-03 用户定为部署默认；实测 345k 均值 +0.184 BLEU、
   * tatoeba zh→en +1.26，en→zh 无变化 —— 规则 B 对中文输出天然不改）。
   * 🔴 正确性判据随之变更：与旧 golden 的逐位比对必须加 `--no_recase`（见 src/postproc.h）。 */
  int pp_opts = MT_PP_RECASE;
  /* 🆕 n-best 导出（**默认 0 = 关闭**；>0 时在每条 OUT 后额外打印 `ALT <k> <score> <text>`，供外部重排） */
  int nbest = 0;
  int split_sent = 0;                      /* 🆕 分句再译（默认关；只作用于逐句路径）*/
  int   *nb_out = NULL, *nb_lens = NULL;   /* n-best 缓冲（nbest>0 时惰性分配） */
  float *nb_scores = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--info")) { print_info(); return 0; }
    else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_file = argv[++i];
    else if (!strcmp(argv[i], "--pad") && i + 1 < argc) pad_file = argv[++i];
    else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--max_new") && i + 1 < argc) max_new = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tgt_bos") && i + 1 < argc) tgt_bos = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump_enc") && i + 1 < argc) dump_enc = argv[++i];
    else if (!strcmp(argv[i], "--dump_logits") && i + 1 < argc) dump_logits = argv[++i];
    else if (!strcmp(argv[i], "--tgt_file") && i + 1 < argc) tgt_file = argv[++i];
    else if (!strcmp(argv[i], "--nogen")) greedy = 0;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--smt") && i + 1 < argc) smt = atoi(argv[++i]);   /* 0=只用物理核 */
    else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bpe") && i + 1 < argc) bpe_file = argv[++i];
    else if (!strcmp(argv[i], "--detok")) detok = 1;
    else if (!strcmp(argv[i], "--bpe_nul")) bpe_nul = 1;
    else if (!strcmp(argv[i], "--text") && i + 1 < argc) text_arg = argv[++i];
    else if (!strcmp(argv[i], "--src_file") && i + 1 < argc) src_file = argv[++i];
    else if (!strcmp(argv[i], "--en2zh")) { dir = 1; dir_forced = 1; }
    else if (!strcmp(argv[i], "--beam") && i + 1 < argc) beam = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
      if (!strcmp(argv[i + 1], "auto")) batch = 0;          /* 0 = 自适应（按 token 预算） */
      else batch = atoi(argv[i + 1]);
      i++;
    }
    else if (!strcmp(argv[i], "--lp_alpha") && i + 1 < argc) lp_alpha = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--recase")) pp_opts |= MT_PP_RECASE;      /* 默认已开（幂等） */
    else if (!strcmp(argv[i], "--no_recase")) pp_opts &= ~MT_PP_RECASE;  /* 关掉 recase ⇒ 与旧 golden 逐字节一致 */
    else if (!strcmp(argv[i], "--num_norm")) pp_opts |= MT_PP_NUMNORM;   /* 全角数字→半角（实测 ≈0） */
    else if (!strcmp(argv[i], "--nbest") && i + 1 < argc) nbest = atoi(argv[++i]);  /* 🆕 默认 0=关 */
    else if (!strcmp(argv[i], "--split_sent")) split_sent = 1;   /* 🆕 分句再译（默认关）*/
    else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
  }

  if (bpe_file) {            /* 分词对拍模式：逐行 → IDS / DETOK */
    mt_tok_t tk;
    if (mt_tok_init(&tk) != 0) { fprintf(stderr, "tok init failed\n"); return 1; }
    FILE *f = fopen(bpe_file, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", bpe_file); return 2; }
    if (bpe_nul) {                      /* 整文件按 '\0' 分隔记录（可含换行） */
      fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      char *buf = (char *)malloc((size_t)sz + 1);
      size_t rd = fread(buf, 1, (size_t)sz, f); buf[rd] = 0; fclose(f);
      int *ids2 = (int *)malloc(sizeof(int) * 16384);
      char *tx2 = (char *)malloc(262144);
      size_t pos = 0;
      while (pos <= rd) {
        size_t e = pos; while (e < rd && buf[e] != 0) e++;
        int m = mt_tok_encode(&tk, buf + pos, (int)(e - pos), ids2, 16384);
        if (m < 0) printf("IDS ERR\n");
        else {
          printf("IDS");
          for (int i = 0; i < m; i++) printf(" %d", ids2[i]);
          printf("\n");
          if (detok) { int nb = mt_tok_decode(&tk, ids2, m, tx2, 262144); fwrite("DETOK", 1, 5, stdout); fputc(0, stdout); fwrite(tx2, 1, (size_t)nb, stdout); fputc(0, stdout); fputc('\n', stdout); }
        }
        fflush(stdout);
        if (e >= rd) break;
        pos = e + 1;
      }
      free(buf); free(ids2); free(tx2);
      mt_tok_free(&tk);
      return 0;
    }

    char *line = NULL; size_t cap2 = 0; ssize_t rl;
    int *ids = (int *)malloc(sizeof(int) * 8192);
    char *txt = (char *)malloc(65536);
    while ((rl = getline(&line, &cap2, f)) != -1) {
      while (rl > 0 && (line[rl-1] == '\n' || line[rl-1] == '\r')) line[--rl] = 0;
      int m = mt_tok_encode(&tk, line, (int)rl, ids, 8192);
      if (m < 0) { printf("IDS ERR\n"); fflush(stdout); continue; }
      printf("IDS");
      for (int i = 0; i < m; i++) printf(" %d", ids[i]);
      printf("\n");
      if (detok) {
        int nb = mt_tok_decode(&tk, ids, m, txt, 65536);
        printf("DETOK %.*s\n", nb, txt);
      }
      fflush(stdout);
    }
    free(ids); free(txt); free(line); fclose(f);
    mt_tok_free(&tk);
    return 0;
  }

  if (!ids_file && !text_arg && !src_file) { print_info(); fprintf(stderr, "\nusage: %s --ids FILE [--pad FILE] [--cap N] [--max_new N] [--tgt_bos ID] [--dump_enc F] [--dump_logits F]\n", argv[0]); return 2; }

#ifdef _OPENMP
  threads = mt_threads_resolve(threads, smt);
  omp_set_num_threads(threads);
  /* OMP_DYNAMIC 的默认值是 implementation-defined：若部署环境（如 AMD pod）全局设了
   * OMP_DYNAMIC=true，libgomp 会给少于请求的线程 ⇒ 基准不可复现、跨机数字不可比。
   * 显式设 0 只为**确定性**，不为加速。*/
  omp_set_dynamic(0);
#ifndef MT_QUANT8
  mt_blas_set_threads(threads);   /* 兜底；真正生效靠上面的一次性 re-exec */
#endif
  {   /* 可观测：每次运行都把"机器给了多少、我们用了多少"打出来。
       * ⚠ 2026-10-04 实测（本机）：绑核**不要做** —— `taskset -c 0-7` 比不绑慢 ~55%
       *   （16.0s vs 10.1s），说明 sysfs 的 online=0-7 并不等于容器实际可用 CPU
       *   （亲和掩码是 0-63，cgroup 配额 800%）。线程数取"online ∩ 亲和"= 8 是对的，
       *   但**别把线程钉到具体核上**。⇒ 这里只报数，不给绑核建议。 */
    fprintf(stderr, "[cpu] avail=%d phys=%d smt=%d threads=%d\n",
            mt_cpu_available(), mt_cpu_physical(), smt, threads);
  }
#endif

  int *src = (int *)malloc(sizeof(int) * cap);
  uint8_t *pad = (uint8_t *)calloc(cap, 1);
  int L = 0;
  if (ids_file) {
    L = read_ints(ids_file, src, cap);
    if (L <= 0) { fprintf(stderr, "empty ids\n"); return 2; }
  }
  if (pad_file) { int m = read_bytes(pad_file, pad, cap); if (m != L) fprintf(stderr, "⚠ pad 长度 %d != ids 长度 %d（按 0 补齐）\n", m, L); }

  mt_model_t *m = mt_model_create();
  if (!m) { fprintf(stderr, "model create failed\n"); return 1; }
  if (text_arg || src_file) {          /* 文本 → 文本：C 内 BPE + 翻译 */
    mt_tok_t tk;
    if (mt_tok_init(&tk) != 0) { fprintf(stderr, "tok init failed\n"); return 1; }
    if (split_sent && batch > 1) { batch = 1; fprintf(stderr, "[split_sent] 与批处理不兼容 ⇒ 已改为 batch=1\n"); }
    /* 🔴 2026-10-03 修静默坑：beam>1 时 c_infer 要求 ctx 的 B == beam（见 decode.h），
     *   而 B 由 max(batch,beam) 决定 ⇒ `--batch 16 --beam 4` 会让 mt_beam_search 返回 -1、**输出空行且不报错**。
     *   这里在**建 ctx 之前**强制 batch=1 并提示（官方 bench 一直用 batch=1/beam=4，历史数字未受影响）。*/
    /* 🆕 2026-10-03：`--batch>1 --beam>1` 现在走**批内 beam**（N 句 × K beam 行）。
     *   只有 --nbest 仍不支持批内（n-best 只在逐句路径实现）⇒ 那种组合才回退 batch=1。*/
    if (beam > 1 && batch > 1 && nbest > 0) {
      fprintf(stderr, "⚠ --nbest 暂不支持批内 beam ⇒ 忽略 --batch，按 batch=1 跑\n");
      batch = 1;
    }
    /* 🆕 自适应 batch（batch==0 ⇒ --batch auto）：按 **token 预算**决定行数。
     * 依据：解码步的 KV 读流量 = R × len × 3层×2(k,v)×HS(8)×DS(76)×4B = R×len×14592 B，
     *   与权重流量（~35.5MB/步，与 R 无关）一起决定能不能被 L3 兜住。
     *   实测（tatoeba zh，Lmax≈15）R=32(b8) 115.4 / R=64(b16) 99.6 / R=128(b32) 82.4 句/s
     *   ⇒ 膝点在 R≈32 ⇒ 用 R×(Lmax+EST_OUT) ≤ BUDGET 外推到别的句长。
     * 🔴 不改数值：R 只影响**分组**，int8 GEMM 对 M 逐位等价（blas.c memcmp 已验 M=1/16/48），
     *   行间独立 + PAD_BIAS 屏蔽 pad ⇒ 每句输出与 batch 无关（0bx 节已双向验证）。
     * ⚠ 必须放在 `batch > 1` 守卫**之前**：auto 时 batch==0。 */
    const int Rmax = MT_ROWS_MAX;
    if (batch == 0) {
        /* 采样估句长中位数：此刻还没读全文（RL/ord 都在后面才生成），只读前 64 行分词。
         * 分桶后真正的长度分布更集中，采样中位数足够定档。 */
        int lens[64], nsamp = 0;
        int *tmp = (int *)malloc(sizeof(int) * 16384);
        if (src_file && tmp) {
          FILE *f0 = fopen(src_file, "r");
          if (f0) {
            char *ln = NULL; size_t lc0 = 0; ssize_t r0;
            while ((r0 = getline(&ln, &lc0, f0)) > 0 && nsamp < 64) {
              while (r0 > 0 && (ln[r0 - 1] == '\n' || ln[r0 - 1] == '\r')) ln[--r0] = 0;
              int nid = mt_tok_encode(&tk, ln, (int)r0, tmp, 16384);
              lens[nsamp++] = (nid >= 0 ? nid + 2 : 2);      /* +2 = 方向标记 + src_eos */
            }
            free(ln); fclose(f0);
          }
        } else if (text_arg && tmp) {
          int nid = mt_tok_encode(&tk, text_arg, (int)strlen(text_arg), tmp, 16384);
          lens[nsamp++] = (nid >= 0 ? nid + 2 : 2);
        }
        free(tmp);
        for (int i = 1; i < nsamp; i++) {                    /* 小样本插入排序取中位数 */
          int k = lens[i], j = i - 1;
          while (j >= 0 && lens[j] > k) { lens[j + 1] = lens[j]; j--; }
          lens[j + 1] = k;
        }
        int Lref = nsamp > 0 ? lens[nsamp / 2] : 16;
        const char *be = getenv("MT_BATCH_BUDGET"), *ee = getenv("MT_BATCH_ESTOUT");
        /* 🔴 预算**按 beam 分档**（2026-10-04 实测，tatoeba zh 1000 句交替取中位）：
         *   beam=4 ⇒ 最佳 **32 行**（b8 8.4s < b16 9.8s < b32 12.1s）
         *   greedy ⇒ 最佳 **8~16 行**（b8 2.83s ≈ b16 2.90s < b32 3.33s；b1 4.9s）
         *   ⇒ 两者相差 ~2~4×，不能共用一个 budget，否则 greedy 会被选到 34 行（3.29s）。
         *   env：MT_BATCH_BUDGET（beam>1）/ MT_BATCH_BUDGET_G（greedy）。 */
        const char *bg = getenv("MT_BATCH_BUDGET_G");
        int budget = (be && *be) ? atoi(be) : 1248;
        if (beam <= 1) budget = (bg && *bg) ? atoi(bg) : 512;
        int estout = (ee && *ee) ? atoi(ee) : 24;
        int denom  = (Lref > 0 ? Lref : 1) + estout;
        int R = budget / denom;
        if (R < 1) R = 1;
        if (R > Rmax) R = Rmax;
        batch = R / beam;
        if (batch < 1) batch = 1;
        if (batch > 64) batch = 64;
        fprintf(stderr, "[autobatch] 采样%d句 Lref=%d beam=%d ⇒ rows=%d batch=%d\n",
                nsamp, Lref, beam, R, batch);
      }
    if (beam > 1 && batch > 1 && batch * beam > Rmax) {
      batch = Rmax / beam; if (batch < 1) batch = 1;
      fprintf(stderr, "[批内beam] batch*beam 超过 %d ⇒ 批大小自动降到 %d\n", Rmax, batch);
    }
    int Brows = (beam > 1 && batch > 1) ? batch * beam
                                       : ((batch > beam) ? batch : (beam < 1 ? 1 : beam));
    mt_ctx_t *cx = mt_ctx_create(Brows, cap, max_new + 1);
    if (!cx) { fprintf(stderr, "ctx create failed\n"); return 1; }
    FILE *sf = NULL;
    if (src_file) { sf = fopen(src_file, "r"); if (!sf) { fprintf(stderr, "cannot open %s\n", src_file); return 2; } }
    char *line = NULL; size_t lc = 0; ssize_t rl;
    int *bid = (int *)malloc(sizeof(int) * 16384);
    int *srcb = (int *)malloc(sizeof(int) * cap);
    uint8_t *padb = (uint8_t *)calloc(cap, 1);
    float *lgb = (float *)malloc(sizeof(float) * VOCAB_SIZE);
    char *ibuf = (char *)malloc(1 << 20);
    char *obuf = (char *)malloc(1 << 20);
    const int tgt_bos = dir ? ID_TO_ZH : ID_TO_EN;

    if (batch > 1) {                       /* ---- 批内并行（贪心 / 批内 beam）---- */
      char **L = NULL; int nL = 0, capL = 0;
      if (sf) {
        while ((rl = getline(&line, &lc, sf)) >= 0) {
          while (rl > 0 && (line[rl - 1] == '\n' || line[rl - 1] == '\r')) line[--rl] = 0;
          if (nL == capL) { capL = capL ? capL * 2 : 256; L = (char **)realloc(L, sizeof(char *) * capL); }
          L[nL++] = strdup(line);
        }
      } else { L = (char **)malloc(sizeof(char *)); L[0] = strdup(text_arg); nL = 1; }
      int *S = (int *)malloc(sizeof(int) * (size_t)batch * cap);
      uint8_t *P = (uint8_t *)malloc((size_t)batch * cap);
      int *OB = (int *)malloc(sizeof(int) * (size_t)batch * max_new);
      int *OL = (int *)malloc(sizeof(int) * batch);
      /* 🆕 长度分桶（2026-10-02）：**先把所有句分词，按长度排序后再分批**。
       *   起因：批处理把长短不齐的句子硬塞一起 ⇒ ① 编码按批内 `Lmax` padding 白算
       *   ② 解码步数被批内**最长**句拖住（实测 100 句 b16 用了 1278 个 token 却跑了 3040 个 slot）。
       *   实测（100 句）：乱序 0.972 s → 按长度排序 0.739 s（**1.32×**）。
       *   数值上无影响：批内各句互相独立（pad 行被 key_pad 屏蔽），只是**换了个分组**。 */
      int *T = (int *)malloc(sizeof(int) * (size_t)nL * cap);   /* 每句的 ids（cap 步长）*/
      int *RL = (int *)malloc(sizeof(int) * nL);                /* 每句的实际长度 */
      int *ord = (int *)malloc(sizeof(int) * nL);
      if (!T || !RL || !ord) { fprintf(stderr, "oom\n"); return 1; }
      for (int i = 0; i < nL; i++) {
        const char *sp = L[i];
        int len = (int)strlen(sp);
        int *trow = T + (size_t)i * cap;
        memset(trow, 0, sizeof(int) * (size_t)cap);
        int nid = mt_tok_encode(&tk, sp, len, bid, 16384);
        int real = 0;
        if (nid >= 0) {
          int has_cjk = 0;
          for (int j = 0; j + 2 < len; j++) { unsigned char c = (unsigned char)sp[j]; if (c >= 0xE4 && c <= 0xE9) { has_cjk = 1; break; } }
          trow[real++] = has_cjk ? ID_FROM_ZH : ID_FROM_EN;
          for (int j = 0; j < nid && real < cap - 1; j++) trow[real++] = bid[j];
          trow[real++] = ID_SRC_EOS;
          if (nid > cap - 2) fprintf(stderr, "⚠ 第 %d 行源句被截断到 %d 个 token（共 %d）⇒ 译文可能不完整，可用 --cap 调大\n",
                                     i + 1, cap - 2, nid);
        } else { trow[0] = ID_FROM_ZH; trow[1] = ID_SRC_EOS; real = 2; }
        RL[i] = real; ord[i] = i;
      }
      {   /* 按长度升序（简单插入排序：nL 是文件行数，量级不大；同长度保持原序）*/
        for (int i = 1; i < nL; i++) {
          int k = ord[i], j = i - 1;
          while (j >= 0 && RL[ord[j]] > RL[k]) { ord[j + 1] = ord[j]; j--; }
          ord[j + 1] = k;
        }
      }
      char **outs = (char **)calloc((size_t)nL, sizeof(char *));
      if (!outs) { fprintf(stderr, "oom\n"); return 1; }
      for (int start = 0; start < nL; start += batch) {
        int B = nL - start; if (B > batch) B = batch;
        memset(P, 0, (size_t)batch * cap);
        int Lmax = 2;
        for (int b = 0; b < B; b++) if (RL[ord[start + b]] > Lmax) Lmax = RL[ord[start + b]];
        for (int b = 0; b < batch; b++) {
          const int src = ord[start + (b < B ? b : 0)];        /* 尾部不足则复制本批第 0 句 */
          int *row = S + (size_t)b * Lmax; uint8_t *pr = P + (size_t)b * Lmax;
          for (int l = 0; l < Lmax; l++) {
            if (l < RL[src]) { row[l] = T[(size_t)src * cap + l]; pr[l] = 0; }
            else { row[l] = ID_PAD; pr[l] = 1; }
          }
        }
        mt_ctx_reset_kv(cx);
        /* 🔴 按**实际长度**编码（2026-10-02 改）：原来固定传 `cap`(48)，
         *    而实测每句只生成 ~12.8 个 token ⇒ 源句实际也就 ~15 个，白算 3 倍。
         *    数值上**逐位等价**：被裁掉的 padding key 在 softmax 里的贡献是 exp(-3.4e38)=0，
         *    加 0 不改变串行求和；p·v 里 p==0 本来就被 `continue` 跳掉。
         *    ⚠ 批内仍要取 **max**（一个 batch 共用一个 ctx），短句的 pad 行由 key_pad 屏蔽。 */
        /* 🆕 方向自动判定（2026-10-02）：源语言标记一直是**自动**判的（含 CJK ⇒ FROM_ZH），
         *   但目标语言原来必须手动传 `--en2zh` —— 不传就是「英文进、英文出」，静默产出垃圾。
         *   现在未显式指定时按本批首句的语言定：源中文 ⇒ 译成英文；源英文 ⇒ 译成中文。
         *   （首句的 marker 就存在 ids 的第 0 位，见下面的分词循环）*/
        const int tb = dir_forced ? tgt_bos
                                  : (T[(size_t)ord[start] * cap] == ID_FROM_ZH ? ID_TO_EN : ID_TO_ZH);
        if (!dir_forced && tb == ID_TO_ZH && start == 0)
          fprintf(stderr, "[方向] 源为英文 ⇒ 生成中文（zh→en 请传中文输入；可用 --en2zh 显式指定）\n");
        int *OB_use = OB, *OL_use = OL;
        int *OBb = NULL, *OLb = NULL;
        if (beam > 1) {                       /* 🆕 批内 beam：一次前向跑 batch*beam 行 */
          int tbs[256];
          for (int b = 0; b < batch; b++) {
            const int src = ord[start + (b < B ? b : 0)];
            tbs[b] = dir_forced ? tgt_bos
                                : (T[(size_t)src * cap] == ID_FROM_ZH ? ID_TO_EN : ID_TO_ZH);
          }
          OBb = (int *)malloc(sizeof(int) * (size_t)batch * max_new);
          OLb = (int *)malloc(sizeof(int) * batch);
          if (!OBb || !OLb) { fprintf(stderr, "oom\n"); return 1; }
          if (mt_beam_search_batch(m, cx, S, P, RL, Lmax, batch, tbs, beam, max_new,
                                   lp_alpha, ID_TGT_EOS, ID_PAD,
                                   OBb, max_new, OLb) != 0) {
            fprintf(stderr, "⚠ 批内 beam 失败\n"); return 1;
          }
          OB_use = OBb; OL_use = OLb;
        } else {
          mt_encode(m, cx, S, P, batch, Lmax);
          mt_greedy_batch(m, cx, batch, max_new, tb, ID_TGT_EOS, ID_PAD, OB, OL);
        }
        for (int b = 0; b < B; b++) {
          int nb = mt_tok_decode(&tk, OB_use + (size_t)b * max_new, OL_use[b], obuf, 1 << 20);
          for (int i = 0; i < nb; i++) if (obuf[i] == '\n' || obuf[i] == '\r') obuf[i] = ' ';
          nb = mt_postproc(obuf, nb, pp_opts);      /* opt-in：默认 0 ⇒ 逐字节不变 */
          char *cp2 = (char *)malloc((size_t)nb + 1);
          if (!cp2) { fprintf(stderr, "oom\n"); return 1; }
          memcpy(cp2, obuf, (size_t)nb); cp2[nb] = 0;
          outs[ord[start + b]] = cp2;                          /* 按**原始行号**存放 */
        }
        free(OBb); free(OLb);
      }
      for (int i = 0; i < nL; i++) { printf("OUT %s\n", outs[i] ? outs[i] : ""); free(outs[i]); }
      fflush(stdout);
      free(outs); free(T); free(RL); free(ord);
      for (int i = 0; i < nL; i++) free(L[i]);
      free(L); free(S); free(P); free(OB); free(OL);
      free(bid); free(srcb); free(padb); free(lgb); free(ibuf); free(obuf); free(line);
      mt_ctx_destroy(cx); mt_tok_free(&tk); mt_model_destroy(m);
      report_rss();
      return 0;
    }
    for (;;) {
      int len;
      const char *srcp;                /* 🔴 必须指向本句文本本身（之前误用了未初始化的 ibuf） */
      if (sf) {
        rl = getline(&line, &lc, sf);
        if (rl < 0) break;
        while (rl > 0 && (line[rl - 1] == '\n' || line[rl - 1] == '\r')) line[--rl] = 0;
        len = (int)rl; srcp = line;
      } else {
        len = (int)strlen(text_arg);
        if (len > (1 << 20) - 1) len = (1 << 20) - 1;
        memcpy(ibuf, text_arg, (size_t)len); ibuf[len] = 0;
        srcp = ibuf;
      }
      memset(padb, 0, (size_t)cap);      /* 🔴 每句必须清零：上一句的 padding 位残留会污染掩码 */
      /* 🆕 分句再译（opt-in）：单句输入 ⇒ nch==1 ⇒ 与旧版逐字节一致 */
      int chs[64], chl[64];
      int nch = split_sent ? mt_split_sent(srcp, len, chs, chl, 64) : 0;
      if (nch <= 1) { nch = 1; chs[0] = 0; chl[0] = len; }
      char acc[1 << 20]; int accn = 0; acc[0] = 0;
      char cbuf[1 << 20];
      for (int ci = 0; ci < nch; ci++) {
      /* 🔴 每段拷进**独立缓冲**并 NUL 结尾，且**每段重置 padb**：
       *   直接在原行内偏移上分词会退化（实测第 2 段会循环复读）——疑与分词/编码的缓冲假设有关。*/
      const int clen = chl[ci] < (int)sizeof(cbuf) - 1 ? chl[ci] : (int)sizeof(cbuf) - 1;
      memcpy(cbuf, srcp + chs[ci], (size_t)clen); cbuf[clen] = 0;
      const char *cp = cbuf;
      memset(padb, 0, (size_t)cap);
      int nid = mt_tok_encode(&tk, cp, clen, bid, 16384);
      int real = 0, shown_dir = 0;
      if (nid >= 0) {
        /* 方向标记：含 CJK ⇒ from_zh，否则 from_en（与评测脚本口径一致） */
        int has_cjk = 0;
        for (int i = 0; i + 2 < clen; i++) {
          unsigned char c = (unsigned char)cp[i];
          if (c >= 0xE4 && c <= 0xE9) { has_cjk = 1; break; }
        }
        srcb[real++] = has_cjk ? ID_FROM_ZH : ID_FROM_EN;
        for (int i = 0; i < nid && real < cap - 1; i++) srcb[real++] = bid[i];
        srcb[real++] = ID_SRC_EOS;
        if (nid > cap - 2) fprintf(stderr, "⚠ 源句被截断到 %d 个 token（共 %d）⇒ 译文可能不完整，可用 --cap 调大\n",
                                   cap - 2, nid);
        for (int i = real; i < cap; i++) { srcb[i] = ID_PAD; padb[i] = 1; }
        /* 同上：未显式 `--en2zh` 时按本句语言自动定目标语言 */
        const int tb = dir_forced ? tgt_bos : (has_cjk ? ID_TO_EN : ID_TO_ZH);
        if (nbest > 0 && !nb_out) {         /* n-best 缓冲（惰性分配一次） */
          nb_out = (int *)malloc(sizeof(int) * (size_t)nbest * max_new);
          nb_lens = (int *)malloc(sizeof(int) * nbest);
          nb_scores = (float *)malloc(sizeof(float) * nbest);
          if (!nb_out || !nb_lens || !nb_scores) { fprintf(stderr, "oom nbest\n"); return 1; }
        }
        int nn = 0, got[8192];
        if (beam > 1) {                     /* beam search（语义对齐 beam_decode_kv.py） */
          int n = mt_beam_search(m, cx, srcb, padb, real, tb, beam, max_new,
                                 lp_alpha, ID_TGT_EOS, ID_PAD, got, 8192,
                                 nbest, nb_out, nb_lens, nb_scores, nbest > 0 ? max_new : 0);
          nn = (n > 0) ? n : 0;
          if (n < 0) fprintf(stderr, "⚠ beam search 返回 %d（检查 --beam 与 --batch/ctx 是否匹配）⇒ 本句输出为空\n", n);
          if (nbest > 0 && nn > 0) {        /* 打印 n-best（k=0 即最优，与上一条 OUT 相同） */
            for (int k = nbest - 1; k >= 0; k--) {     /* 行内按分数降序 ⇒ 倒着写保证 k=0 最后 */
              int nb = mt_tok_decode(&tk, nb_out + (size_t)k * max_new, nb_lens[k], obuf, 1 << 20);
              for (int q = 0; q < nb; q++) if (obuf[q] == '\n' || obuf[q] == '\r') obuf[q] = ' ';
              nb = mt_postproc(obuf, nb, pp_opts);
              printf("ALT %d %.4f %.*s\n", k, nb_scores[k], nb, obuf);
            }
          }
        } else {                            /* greedy + KV cache */
          mt_ctx_reset_kv(cx);
          mt_encode(m, cx, srcb, padb, 1, real);   /* 🔴 按实际长度（原来是 cap=48）*/
          if (!dir_forced && tb == ID_TO_ZH && nn == 0 && !shown_dir) {
            fprintf(stderr, "[方向] 源为英文 ⇒ 生成中文（zh→en 请传中文输入；可用 --en2zh 显式指定）\n");
            shown_dir = 1;
          }
          int cur = tb;
          mt_decode_step(m, cx, &cur, 1, 1, lgb);
          while (nn < max_new) {
            int nx = mt_argmax(lgb, VOCAB_SIZE);
            if (nx == ID_TGT_EOS || nx == ID_PAD) break;
            got[nn++] = nx;
            cur = nx;
            mt_decode_step(m, cx, &cur, 1, 1, lgb);
          }
        }
        int nb = mt_tok_decode(&tk, got, nn, obuf, 1 << 20);
        for (int i = 0; i < nb; i++) if (obuf[i] == '\n' || obuf[i] == '\r') obuf[i] = ' ';
        nb = mt_postproc(obuf, nb, pp_opts);        /* opt-in：默认 0 ⇒ 逐字节不变 */
        if (nb > 0) {                               /* 累加本段；英文输出补空格，中文不补 */
          if (ci > 0 && accn > 0 && accn < (int)sizeof(acc) - 2) acc[accn++] = ' ';
          int m = nb; if (m > (int)sizeof(acc) - 2 - accn) m = (int)sizeof(acc) - 2 - accn;
          memcpy(acc + accn, obuf, (size_t)m); accn += m; acc[accn] = 0;
        }
      }
        if (getenv("MT_SPLIT_TRACE"))
          fprintf(stderr, "[split] ci=%d/%d off=%d len=%d accn=%d acc='%.80s'\n",
                  ci, nch, chs[ci], chl[ci], accn, acc);
      }   /* ← 分句循环结束 */
      printf("OUT %s\n", acc);
      if (!sf) break;
    }
    free(bid); free(srcb); free(padb); free(lgb); free(ibuf); free(obuf); free(line);
    mt_ctx_destroy(cx); mt_tok_free(&tk);
    mt_model_destroy(m);
    report_rss();
    return 0;
  }

  /* 上下文按 L 与 max_new 分配；增量步 L=1 ⇒ cap_tgt = max_new+1 */
  mt_ctx_t *c = mt_ctx_create(1, L, max_new + 1);
  if (!c) { fprintf(stderr, "ctx create failed\n"); return 1; }

  if (repeat > 1) {                     /* 进程内重复，摊掉启动开销 */
    float *lg = (float *)malloc(sizeof(float) * VOCAB_SIZE);
    for (int r = 0; r < repeat; r++) {
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      mt_ctx_reset_kv(c);
      mt_encode(m, c, src, pad_file ? pad : NULL, 1, L);
      int cur = tgt_bos, nn = 0;
      mt_decode_step(m, c, &cur, 1, 1, lg);
      while (nn < max_new) {
        int nx = mt_argmax(lg, VOCAB_SIZE);
        if (nx == ID_TGT_EOS || nx == ID_PAD) break;
        cur = nx; nn++;
        mt_decode_step(m, c, &cur, 1, 1, lg);
      }
      clock_gettime(CLOCK_MONOTONIC, &t1);
      double mst = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
      struct rusage ru;
      getrusage(RUSAGE_SELF, &ru);
      fprintf(stderr, "  iter %2d: %7.2f ms  utime=%7.1f stime=%6.1f minflt=%7ld majflt=%5ld\n",
              r, mst,
              ru.ru_utime.tv_sec * 1e3 + ru.ru_utime.tv_usec / 1e3,
              ru.ru_stime.tv_sec * 1e3 + ru.ru_stime.tv_usec / 1e3,
              ru.ru_minflt, ru.ru_majflt);
    }
    free(lg);
    return 0;
  }

  mt_encode(m, c, src, pad_file ? pad : NULL, 1, L);
  if (dump_enc) dump_floats(dump_enc, c->enc_out, (size_t)L * D_MODEL);

  int *tgt = (int *)malloc(sizeof(int) * (max_new + 2));
  tgt[0] = tgt_bos;
  float *logits = (float *)malloc(sizeof(float) * VOCAB_SIZE);
  int n = 0;

  if (tgt_file) {   /* 强制喂入给定 tgt 前缀（teacher-forcing 对拍用） */
    int *pre = (int *)malloc(sizeof(int) * (max_new + 2));
    int pn = read_ints(tgt_file, pre, max_new + 2);
    for (int i = 0; i < pn; i++) {
      tgt[n] = pre[i];
      mt_decode_step(m, c, &tgt[n], 1, 1, logits);
      n++;
      if (n >= max_new + 1) break;
    }
    free(pre);
    if (dump_logits) dump_floats(dump_logits, logits, VOCAB_SIZE);
    fprintf(stderr, "teacher-forced %d steps, last argmax=%d\n", n, mt_argmax(logits, VOCAB_SIZE));
    return 0;
  }

  mt_decode_step(m, c, tgt, 1, 1, logits);
  n = 1;
  if (dump_logits) dump_floats(dump_logits, logits, VOCAB_SIZE);
  if (greedy) {
    for (int step = 0; step < max_new; step++) {
      int nxt = mt_argmax(logits, VOCAB_SIZE);
      if (nxt == ID_TGT_EOS || nxt == ID_PAD) break;   /* 与 PyTorch 一致：eos 不 append */
      tgt[n++] = nxt;
      mt_decode_step(m, c, &tgt[n - 1], 1, 1, logits);
    }
  }
  printf("GEN");
  for (int i = 1; i < n; i++) printf(" %d", tgt[i]);
  printf("\n");

  mt_ctx_destroy(c);
  mt_model_destroy(m);
  free(src); free(pad); free(tgt); free(logits);
  return 0;
}
