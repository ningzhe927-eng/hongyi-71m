#ifndef MT_CPU_TOPOLOGY_H
#define MT_CPU_TOPOLOGY_H

#include <stddef.h>

/* 真实可用 CPU 数（亲和掩码），不是 nproc。 */
int mt_cpu_available(void);

/* 物理核数；探测失败或平台不支持返回 -1（调用方须优雅降级 —— 例如忽略 --smt 0）。 */
int mt_cpu_physical(void);

/* 每个物理核一个代表 CPU 的字符串（"0,2,4,6"），供外部 taskset 用；失败返回 0。 */
int mt_cpu_phys_mask_str(char *buf, size_t n);

/* 最终线程数决策：threads_opt>0 则原样尊重；否则 --smt 0 且物理核可知 ⇒ 物理核数；
 * 否则 = 可用逻辑 CPU；最后按 MT_THREAD_CAP（默认 16）封顶。 */
int mt_threads_resolve(int threads_opt, int smt_opt);

#endif /* MT_CPU_TOPOLOGY_H */
