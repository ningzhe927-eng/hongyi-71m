#ifndef MT_TOKENIZER_BPE_H
#define MT_TOKENIZER_BPE_H

#include <stdint.h>

/* GPT-2 byte-level BPE（HF tokenizers 口径）。
 * 表来自 gen_tok/tok_tables.c；本结构只放运行时哈希。 */
typedef struct {
  int32_t *mrank;   /* 开放寻址：(left<<20)|right → rank+1；0 = 空 */
  uint32_t mask;
} mt_tok_t;

int  mt_tok_init(mt_tok_t *t);
void mt_tok_free(mt_tok_t *t);

/* 文本(UTF-8) → token id（不含任何 special）。返回条数，-1 = 出错/超容量。 */
int  mt_tok_encode(const mt_tok_t *t, const char *text, int len, int *out, int cap);

/* token id → 文本字节（跳过 special）。返回写入字节数。 */
int  mt_tok_decode(const mt_tok_t *t, const int *ids, int n, char *out, int cap);

#endif /* MT_TOKENIZER_BPE_H */
