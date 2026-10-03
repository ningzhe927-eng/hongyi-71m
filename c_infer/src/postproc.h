#ifndef MT_POSTPROC_H
#define MT_POSTPROC_H

/* 输出侧后处理（只改文本，不碰数值）—— 2026-10-03
 * 🔴 **MT_PP_RECASE 已是部署默认**（main.c 里 `pp_opts = MT_PP_RECASE`）：
 *    ⇒ **与旧 golden（`mt_infer_int8_v19_cli_pgo`）做逐位比对时，两边都必须加 `--no_recase`**。
 *    ⇒ 新 golden 应基于"默认开 recase"的输出建立（别再拿旧 golden 直接 diff）。
 * 实测（345k 基座，16 格，见 实验/audit_recase_numnorm.py）：
 *   MT_PP_RECASE  首字母大写（规则 B）：均值 +0.184 BLEU（tatoeba zh→en **+1.26**、flores +0.08、opensub +0.10）；en→zh 全格 ±0.00
 *   MT_PP_NUMNORM 全角数字→半角：**≈0**（这些测试集里数字错误仅 ~1%）
 *   ❌ 不要做：加千分位（−0.10，且会把年份 2016 改成 2,016）、去千分位（−0.007）、全角标点→半角（−2.23） */
#define MT_PP_RECASE  1   /* 第一个 ASCII 字母（跳过前导引号/括号/省略号/多字节字符）改大写 */
#define MT_PP_NUMNORM 2   /* 全角数字（U+FF10-FF19）→ ASCII 数字 */

/* 就地处理 buf[0..n)，返回新长度（可能变短）。opts 为 0 时原样返回 n。 */
int mt_postproc(char *buf, int n, int opts);

#endif
