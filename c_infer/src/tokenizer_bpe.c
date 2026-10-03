#include "tokenizer_bpe.h"
#include "tok_tables.h"

#include <stdlib.h>
#include <string.h>

/* ================= Unicode 分类（区间二分） ================= */
static int in_ranges(const tok_range_t *r, int n, uint32_t cp) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (cp < r[mid].lo) hi = mid - 1;
    else if (cp > r[mid].hi) lo = mid + 1;
    else return 1;
  }
  return 0;
}
static int is_letter(uint32_t cp) { return in_ranges(TOK_LETTER, TOK_N_LETTER, cp); }
static int is_number(uint32_t cp) { return in_ranges(TOK_NUMBER, TOK_N_NUMBER, cp); }
static int is_space (uint32_t cp) { return in_ranges(TOK_SPACE,  TOK_N_SPACE,  cp); }

/* ================= UTF-8 解码 ================= */
/* 解码 s[off..len) 的下一个码点；返回码点，*adv = 消耗字节数（非法字节当作单字节 0xFFFD 处理） */
static uint32_t utf8_next(const unsigned char *s, int len, int off, int *adv) {
  if (off >= len) { *adv = 0; return 0; }
  unsigned char c = s[off];
  if (c < 0x80) { *adv = 1; return c; }
  int need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
  if (need == 1 || off + need > len) { *adv = 1; return 0xFFFD; }
  uint32_t cp = c & ((1u << (7 - need)) - 1);
  for (int i = 1; i < need; i++) {
    unsigned char cc = s[off + i];
    if ((cc & 0xC0) != 0x80) { *adv = 1; return 0xFFFD; }
    cp = (cp << 6) | (cc & 0x3F);
  }
  *adv = need;
  return cp;
}

/* ================= GPT-2 正则切分 =================
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
 *  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * 输入：码点数组 cp[0..n)、字节偏移 boff[0..n]；返回本次匹配的“码点结束下标”，-1 = 失败。
 */
static int lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? (int)(c + 32) : (int)c; }

static int match_here(const uint32_t *cp, int n, int i) {
  uint32_t c = cp[i];

  /* 分支 1：'s 't 're 've 'm 'll 'd（大小写不敏感） */
  if (c == '\'') {
    int k = lower_ascii(i + 1 < n ? cp[i + 1] : 0);
    if (k == 's' || k == 't' || k == 'm' || k == 'd') return i + 2;
    if (k == 'r' && i + 2 < n && lower_ascii(cp[i + 2]) == 'e') return i + 3;
    if (k == 'v' && i + 2 < n && lower_ascii(cp[i + 2]) == 'e') return i + 3;
    if (k == 'l' && i + 2 < n && lower_ascii(cp[i + 2]) == 'l') return i + 3;
  }

  /* 分支 2：[^\r\n\p{L}\p{N}]?\p{L}+ */
  {
    int j = i;
    if (!(is_letter(c))) {
      if (c != '\r' && c != '\n' && !is_number(c)) j = i + 1;   /* 可选的一个前导非字母数字 */
      else j = -1;
    }
    if (j >= 0 && j < n && is_letter(cp[j])) {
      int e = j;
      while (e < n && is_letter(cp[e])) e++;
      if (e > j) return e;
    }
  }

  /* 分支 3：\p{N} */
  if (is_number(c)) return i + 1;

  /* 分支 4： ?[^\s\p{L}\p{N}]+[\r\n]* */
  {
    int j = i;
    if (cp[j] == ' ') j++;
    int s = j;
    while (j < n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j])) j++;
    if (j > s) {
      while (j < n && (cp[j] == '\r' || cp[j] == '\n')) j++;
      return j;
    }
    if (i + 1 == s && s < n) { /* 只有可选空格，后面判失败 → 落到后面的分支 */ }
  }

  /* 分支 5：\s*[\r\n]+  —— 取“最后一个 \r\n 游程”的结束位置 */
  if (is_space(c)) {
    int j = i;
    while (j < n && is_space(cp[j])) j++;
    int last_run_start = -1, last_run_end = -1;
    for (int p = i; p < j; p++) {
      if ((cp[p] == '\r' || cp[p] == '\n') && (p == i || (cp[p - 1] != '\r' && cp[p - 1] != '\n'))) {
        int q = p;
        while (q < j && (cp[q] == '\r' || cp[q] == '\n')) q++;
        last_run_start = p; last_run_end = q;
      }
    }
    if (last_run_start >= 0) return last_run_end;      /* \s* = [i,last_run_start) */

    /* 分支 6：\s+(?!\S) —— 游程后是非空白 ⇒ 退一个字符；到串尾则全取 */
    if (j == n) return j;
    if (j - i >= 2) return j - 1;

    /* 分支 7：\s+ */
    return j;
  }
  return -1;
}

/* 返回 i 处匹配的码点结束下标；失败返回 i+1（防御性，不应发生） */
static int split_next(const uint32_t *cp, int n, int i) {
  int e = match_here(cp, n, i);
  return (e > i) ? e : i + 1;
}

/* ================= 运行时 merge 哈希 =================
 * 🔴🔴 2026-10-01 修：原来两处都写 `h = (uint32_t)(key ^ (key>>32))`，`key = (L<<20)|R`。
 *    `p = h & mask`（mask=2^17-1）只取**低 17 位**，而 `key` 的低 17 位只由 `R` 和 `L` 的高 8 位
 *    决定（`(L&0xFFF)<<20` 那部分落在 20 位以上，完全没参与）⇒ 51203 条全挤在 ~2000 个槽里，
 *    建表退化成线性扫描：**实测 736 ms、探测 10.9 亿次**（每次 merge_rank 也要扫 ~25 个条目）。
 * ✅ 换成 `hash_pair`（本文件里本来就有、却一直没人调用的那个）：只改槽位分布，
 *    **不改任何查表结果**（同 key ⇒ 同 rank）⇒ 输出完全不变，已用 17,874 句对拍验证。
 *    init: 736 ms → ~2 ms。 */
static uint32_t hash_pair(uint32_t l, uint32_t r) {
  uint64_t x = ((uint64_t)l << 20) ^ ((uint64_t)r * 0x9E3779B97F4A7C15ull);
  x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
  return (uint32_t)x;
}

int mt_tok_init(mt_tok_t *t) {
  uint32_t cap = 1;
  while (cap < (uint32_t)TOK_N_MERGES * 2u) cap <<= 1;
  t->mask = cap - 1;
  t->mrank = (int32_t *)calloc(cap, sizeof(int32_t));
  if (!t->mrank) return -1;
  for (uint32_t r = 0; r < (uint32_t)TOK_N_MERGES; r++) {
    uint32_t p = hash_pair(TOK_MERGE_LEFT[r], TOK_MERGE_RIGHT[r]) & t->mask;
    while (t->mrank[p] != 0) p = (p + 1) & t->mask;
    t->mrank[p] = (int32_t)(r + 1);
  }
  return 0;
}
void mt_tok_free(mt_tok_t *t) { free(t->mrank); t->mrank = NULL; }

static int merge_rank(const mt_tok_t *t, uint32_t l, uint32_t r) {
  uint32_t p = hash_pair(l, r) & t->mask;
  while (t->mrank[p] != 0) {
    uint32_t rank = (uint32_t)t->mrank[p] - 1;
    if (TOK_MERGE_LEFT[rank] == l && TOK_MERGE_RIGHT[rank] == r) return (int)rank;
    p = (p + 1) & t->mask;
  }
  return -1;
}

/* ================= BPE（对一段字节做合并；结果写入 out） ================= */

/* 优先级队列条目：相邻对 (i, j) 的合并 rank + 入队时的两个符号值。
 * 🔴 `si/sj` 是为了**过期判定**：只查 `live[i] && live[j] && nx[i]==j` 不够 ——
 *    j 作为**左元素**被合并时 `sym[j]` 变了，但 `nx[i]` 仍指向 j（死的是 j 的右邻居），
 *    该条目会被误判为有效，从而用旧 rank 去合并一个**新对**（实测就是这里出的错）。 */
typedef struct { int rank, i, j, si, sj; } bpe_ent_t;

/* 排序：先按 rank，**rank 相同取最左的 i**。
 * 🔴 这个 tie-break 不是可选的：旧实现是「从左到右扫，取 rank 最小的**第一处**」，
 *    对 (X,X) 这类**自重叠**对（如 "aaa" 里的 (a,a)）必须先合左边，否则结果不同。 */
static int ent_less(bpe_ent_t a, bpe_ent_t b) {
  return (a.rank != b.rank) ? (a.rank < b.rank) : (a.i < b.i);
}
static void heap_push(bpe_ent_t *h, int *hn, bpe_ent_t e) {
  int p = (*hn)++;
  h[p] = e;
  while (p > 0) {
    int q = (p - 1) >> 1;
    if (!ent_less(h[p], h[q])) break;
    bpe_ent_t s = h[q]; h[q] = h[p]; h[p] = s;
    p = q;
  }
}
static bpe_ent_t heap_pop(bpe_ent_t *h, int *hn) {
  bpe_ent_t top = h[0];
  h[0] = h[--(*hn)];
  int p = 0, n = *hn;
  for (;;) {
    int l = 2 * p + 1, r = l + 1, m = p;
    if (l < n && ent_less(h[l], h[m])) m = l;
    if (r < n && ent_less(h[r], h[m])) m = r;
    if (m == p) break;
    bpe_ent_t s = h[m]; h[m] = h[p]; h[p] = s;
    p = m;
  }
  return top;
}

/* ⚠ 旧实现（已换掉）：每轮 **全扫** n-1 个相邻对求最小 rank，再**整段重建**一次 ⇒ O(n²)。
 *   🔴 实测这是端到端最大的一块：100 句 encode 要 **2.56 s**，比模型推理还贵。
 * ✅ 现在：双向链表 + 最小堆（按 (rank, i) 排序）。每个相邻对只在**创建/被影响**时算一次 rank
 *   ⇒ O(n log n)。已用 17,874 句（zh+en 全量）+ 边界用例对拍，token id **全等**。
 * 🔴🔴 三个必须踩准的点（都是实测踩出来的，改这段代码前务必读）：
 *   ① **「一次合并所有出现」不能退化成「一次合并一处」**：旧实现每轮把 (L,R) 的**所有**出现
 *      从左到右合并完，才重新取全局最小。若合并一处就重新取最小，一旦新对 (RES,x) 的 rank
 *      比当前 r 更小就会先走岔 ⇒ 实测 en 语料 8937 行里 **7993 行** token 不同。
 *      （⇨ 下面每次合并后必须先把堆里所有 rank==r 的条目**排干**。）
 *   ② **过期判定必须带上符号值 `si/sj`**：只查 `live && nx[i]==j` 不够 —— j 作为**左元素**
 *      被合并时 `sym[j]` 变了，而 `nx[i]` 仍指向 j（死的是 j 的右邻居）⇒ 会被误判为有效，
 *      拿旧 rank 去合并一个**新对**。
 *   ③ **tie-break 取最左的 i**：对 (X,X) 这类**自重叠**对（如 "aaa"）必须先合左边，
 *      否则与旧实现的「从左到右扫」不一致。 */
/* 过期判定：两个节点都活着、仍相邻、且符号值没变过 ⇒ 入队时的 rank 仍然有效。 */
static int ent_live(const char *live, const int *nx, const int *sym, const bpe_ent_t *e) {
  return live[e->i] && live[e->j] && nx[e->i] == e->j &&
         sym[e->i] == e->si && sym[e->j] == e->sj;
}

/* 合并相邻对 (i,j)：i 变成 RES，j 摘掉，只重算左右两个新相邻对并入堆。 */
static void do_merge(int *sym, int *pr, int *nx, char *live, bpe_ent_t *heap, int *hn,
                     const mt_tok_t *t, const bpe_ent_t *e) {
  int i = e->i, j = e->j, k = nx[j];
  sym[i] = (int)TOK_MERGE_RESULT[e->rank];
  live[j] = 0;
  nx[i] = k;
  if (k >= 0) pr[k] = i;
  int p = pr[i];
  if (p >= 0) {
    int r = merge_rank(t, (uint32_t)sym[p], (uint32_t)sym[i]);
    if (r >= 0) heap_push(heap, hn, (bpe_ent_t){r, p, i, sym[p], sym[i]});
  }
  if (k >= 0) {
    int r = merge_rank(t, (uint32_t)sym[i], (uint32_t)sym[k]);
    if (r >= 0) heap_push(heap, hn, (bpe_ent_t){r, i, k, sym[i], sym[k]});
  }
}

static int bpe(const mt_tok_t *t, const unsigned char *seg, int len, int *out, int cap) {
  if (len <= 0) return 0;
  const size_t N = (size_t)len;
  int *sym = (int *)malloc(N * sizeof(int));
  int *pr = (int *)malloc(N * sizeof(int));
  int *nx = (int *)malloc(N * sizeof(int));
  char *live = (char *)malloc(N);
  /* 堆容量上界：初始 len-1 条，每次合并净增 1 条，合并次数 ≤ len-1 ⇒ 最多 2·len-2 */
  bpe_ent_t *heap = (bpe_ent_t *)malloc((3 * N + 8) * sizeof(bpe_ent_t));
  if (!sym || !pr || !nx || !live || !heap) {
    free(sym); free(pr); free(nx); free(live); free(heap); return -1;
  }

  int hn = 0;
  for (int i = 0; i < len; i++) {
    sym[i] = seg[i];                       /* 初始符号 = 单字节 */
    pr[i] = i - 1;
    nx[i] = (i + 1 < len) ? i + 1 : -1;
    live[i] = 1;
  }
  for (int i = 0; i + 1 < len; i++) {
    int r = merge_rank(t, (uint32_t)sym[i], (uint32_t)sym[i + 1]);
    if (r >= 0) heap_push(heap, &hn, (bpe_ent_t){r, i, i + 1, sym[i], sym[i + 1]});
  }

  while (hn > 0) {
    bpe_ent_t e = heap_pop(heap, &hn);
    if (!ent_live(live, nx, sym, &e)) continue;      /* 过期，丢弃 */
    /* 🔴 合并 rank=r 的对时，必须把**所有** rank==r 的出现一次合并完，再看更小的 rank。
     *   （rank 相同 ⇒ 必然是同一个 (L,R) 对，因为表里 rank ↔ pair 一一对应）
     *   ⇦ 旧实现是「一轮里把 (L,R) 的所有出现从左到右合并完再重新全局扫描」；
     *     若这里改成「一次只合并一处」，一旦 (RES,x) 的 rank 比 r 小就会先走岔，
     *     实测 en 语料 8937 行里有 **7993 行**对不上。 */
    int r = e.rank;
    do_merge(sym, pr, nx, live, heap, &hn, t, &e);
    /* 同 rank 的其余出现（可能夹着过期条目）也要合并完；新 push 的条目若 rank 更小，
     * `heap[0].rank == r` 会自然不成立 ⇒ 回到外层重新取全局最小（与旧实现一致）。 */
    while (hn > 0 && heap[0].rank == r) {
      bpe_ent_t e2 = heap_pop(heap, &hn);
      if (ent_live(live, nx, sym, &e2)) do_merge(sym, pr, nx, live, heap, &hn, t, &e2);
    }
  }

  int outn = 0, rc = 0;
  for (int i = 0; i >= 0; i = nx[i]) {      /* 表头 0 永远存活（它不可能成为右元素） */
    int vid = (sym[i] >= 0 && sym[i] < TOK_N_SYM) ? TOK_SYM2VOCAB[sym[i]] : -1;
    if (vid < 0) { rc = -1; break; }
    if (outn >= cap) { rc = -1; break; }
    out[outn++] = vid;
  }
  free(sym); free(pr); free(nx); free(live); free(heap);
  return rc < 0 ? -1 : outn;
}

int mt_tok_encode(const mt_tok_t *t, const char *text, int len, int *out, int cap) {
  const unsigned char *s = (const unsigned char *)text;
  /* 1) 解码码点 + 字节偏移 */
  int maxcp = len + 1;
  uint32_t *cp = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)maxcp);
  int *bo = (int *)malloc(sizeof(int) * (size_t)(maxcp + 1));
  if (!cp || !bo) { free(cp); free(bo); return -1; }
  int n = 0, off = 0;
  while (off < len) {
    int adv = 0;
    cp[n] = utf8_next(s, len, off, &adv);
    bo[n] = off;
    off += adv; n++;
  }
  bo[n] = len;

  int total = 0;
  int i = 0;
  while (i < n) {
    int e = split_next(cp, n, i);
    int b0 = bo[i], b1 = bo[e];
    int got = bpe(t, s + b0, b1 - b0, out + total, cap - total);
    if (got < 0) { free(cp); free(bo); return -1; }
    total += got;
    i = e;
  }
  free(cp); free(bo);
  return total;
}

int mt_tok_decode(const mt_tok_t *t, const int *ids, int n, char *out, int cap) {
  (void)t;
  int w = 0;
  for (int i = 0; i < n; i++) {
    int id = ids[i];
    if (id < 0 || id >= TOK_VOCAB_SIZE) continue;
    if (TOK_IS_SPECIAL[id]) continue;
    uint32_t b0 = TOK_OFF[id], b1 = TOK_OFF[id + 1];
    for (uint32_t p = b0; p < b1; p++) {
      if (w >= cap - 1) return w;
      out[w++] = (char)TOK_BLOB[p];
    }
  }
  out[w] = 0;
  return w;
}
