#include "postproc.h"

/* 与 Python 参考实现等价（见 实验/audit_recase_numnorm.py 的 ruleB）：
 * 只有当前导**仅由**空白/标点/引号/括号/破折号/省略号构成时，才把第一个字母改大写；
 * 一旦先遇到**数字 / 大写字母 / CJK / 其它字符**就原样返回（避免 "2016 was"→"2016 Was"、
 * "5,000 yuan"→"5,000 Yuan" 这类把句子中间词大写的错误）。
 * 前导集合与 Python 版逐字一致 ⇒ 两个实现应给出一致的输出（已用 1000 句 diff=0 验证）。 */
static int skippable(unsigned char c) {
  switch (c) {
    case ' ': case '\t': case '\'': case '"': case '(': case '[': case '{':
    case '<': case '-': case '.': case ',': case ';': case ':': case '!':
    case '?': case '/': case '\\': case '*':
      return 1;
    default:
      return 0;
  }
}

/* 多字节"可跳过"字符：U+00AB/BB、U+2018/19/1C/1D、U+2026（与 Python SKIP 集合对应） */
static int skippable_mb(const unsigned char *p, int n_left, int *bytes) {
  if (n_left >= 2 && p[0] == 0xC2 && (p[1] == 0xAB || p[1] == 0xBB)) { *bytes = 2; return 1; }
  if (n_left >= 3 && p[0] == 0xE2 && p[1] == 0x80 &&
      (p[2] == 0x98 || p[2] == 0x99 || p[2] == 0x9C || p[2] == 0x9D || p[2] == 0xA6)) { *bytes = 3; return 1; }
  return 0;
}

static void recase(char *b, int n) {
  int i = 0;
  while (i < n) {
    unsigned char c = (unsigned char)b[i];
    if (c < 0x80) {
      if (skippable(c)) { i++; continue; }
      if (c >= 'a' && c <= 'z') b[i] = (char)(c - 32);
      return;                     /* 大写字母 / 数字 / 其它 ASCII ⇒ 不动 */
    }
    int mb = 0;
    if (skippable_mb((const unsigned char *)b + i, n - i, &mb)) { i += mb; continue; }
    return;                       /* CJK / 其它多字节 ⇒ 不动（保守：可能是未翻译残留） */
  }
}

/* 全角数字 U+FF10..U+FF19 = EF BC 90..99 → '0'..'9'（3 字节压成 1 字节，就地左移）。 */
static int numnorm(char *b, int n) {
  int w = 0;
  for (int r = 0; r < n;) {
    unsigned char c = (unsigned char)b[r];
    if (c == 0xEF && r + 2 < n && (unsigned char)b[r + 1] == 0xBC &&
        (unsigned char)b[r + 2] >= 0x90 && (unsigned char)b[r + 2] <= 0x99) {
      b[w++] = (char)('0' + ((unsigned char)b[r + 2] - 0x90));
      r += 3;
    } else {
      b[w++] = b[r++];
    }
  }
  return w;
}

int mt_postproc(char *buf, int n, int opts) {
  if (!buf || n <= 0 || !opts) return n;
  if (opts & MT_PP_NUMNORM) n = numnorm(buf, n);
  if (opts & MT_PP_RECASE) recase(buf, n);
  return n;
}
