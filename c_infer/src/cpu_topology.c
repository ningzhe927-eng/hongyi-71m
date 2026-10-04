/* 真实可用的 CPU 资源探测。
 * 🔴 为什么不能直接信 nproc / 亲和掩码：本机 `nproc`=64、`sched_getaffinity` 也给 64，
 *    但 `/sys/devices/system/cpu/online` 只有 **0-7**，cgroup v1 `cpu.cfs_quota_us`=800000
 *    （周期 100000 ⇒ 恰好 8 核）⇒ 真能跑线程的只有 8。按 64 起 OMP 线程池会 8× 过订阅
 *    （`记忆/会话记忆.md` 踩坑 #1：历史上多次"读出假结论"就是这个）。
 *    ⇒ 可用 CPU = **亲和掩码 ∩ online**。
 * 🔴 为什么物理核要另外算：online=0-7 但 thread_siblings 是 0-1/2-3/4-5/6-7
 *    ⇒ **4 个物理核 + SMT**，而历史所有基准都用 --threads 8 ⇒ 一直在 SMT 上跑。
 * 🟢 可移植：Linux 走 sysfs + 亲和掩码，macOS 走 sysctlbyname，其它平台返回"不可用"，
 *    由调用方优雅降级 —— 探测失败绝不 abort、绝不覆盖用户显式指定的线程数。
 *    env 逃生舱：MT_CPU_AVAIL / MT_CPU_PHYS / MT_THREAD_CAP（也便于伪装多核验证策略）。
 * ⚠ `_GNU_SOURCE` 必须定义在**任何 include 之前**（否则 features.h 已被先前的 include 锁死，
 *    `CPU_SETSIZE`/`CPU_COUNT` 全不可用 —— 本工程用 -std=c11，踩过）。 */
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE 1
#endif

#include "cpu_topology.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#  include <sched.h>
#endif
#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

static int env_or(const char *k, int dflt) {
  const char *e = getenv(k);
  if (e && *e) { int v = atoi(e); if (v > 0) return v; }
  return dflt;
}

#if defined(__linux__) && defined(CPU_SETSIZE)
/* 解析 "0-7" / "0,2-4,8" 形式的 CPU 列表，把其中每个 CPU 写入 set。返回 1 成功。 */
static int parse_cpu_list(const char *path, cpu_set_t *set) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char line[256];
  int ok = 0;
  if (fgets(line, sizeof line, f)) {
    int a = -1, b = -1, dash = 0;
    for (const char *p = line; *p; p++) {
      if (*p >= '0' && *p <= '9') {
        int v = 0;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        p--;
        if (a < 0) a = v; else b = v;
      } else if (*p == '-') {
        dash = 1;
      } else {                       /* ',' '\n' 空格等：结束一段 */
        if (a >= 0) {
          int hi = (dash && b >= 0) ? b : a;
          for (int i = a; i <= hi && i < CPU_SETSIZE; i++) { CPU_SET(i, set); ok = 1; }
        }
        a = -1; b = -1; dash = 0;
      }
    }
    if (a >= 0) {
      int hi = (dash && b >= 0) ? b : a;
      for (int i = a; i <= hi && i < CPU_SETSIZE; i++) { CPU_SET(i, set); ok = 1; }
    }
  }
  fclose(f);
  return ok;
}

/* 可用 CPU 集合 = 亲和掩码 ∩ online。返回 1 成功（两个源都拿到），0 表示部分/完全失败，
 * 此时 *set 仍然被填成"尽力而为"的结果（亲和掩码 或 online）。 */
static int usable_cpu_set(cpu_set_t *set) {
  cpu_set_t aff, onl;
  CPU_ZERO(&aff); CPU_ZERO(&onl);
  int have_aff = (sched_getaffinity(0, sizeof aff, &aff) == 0);
  int have_onl = parse_cpu_list("/sys/devices/system/cpu/online", &onl);
  if (have_aff && have_onl) {
    CPU_ZERO(set);
    for (int i = 0; i < CPU_SETSIZE; i++) if (CPU_ISSET(i, &aff) && CPU_ISSET(i, &onl)) CPU_SET(i, set);
    return 1;
  }
  if (have_aff) { CPU_ZERO(set); CPU_OR(set, set, &aff); return 0; }
  if (have_onl) { CPU_ZERO(set); CPU_OR(set, set, &onl); return 0; }
  CPU_ZERO(set);
  return 0;
}
#endif

int mt_cpu_available(void) {
  const char *e = getenv("MT_CPU_AVAIL");
  if (e && *e) { int v = atoi(e); if (v > 0) return v; }
#if defined(__linux__) && defined(CPU_SETSIZE)
  {
    cpu_set_t set;
    usable_cpu_set(&set);
    int n = CPU_COUNT(&set);
    if (n > 0) return n;                       /* 本机 = 8（不是 nproc 的 64） */
  }
#endif
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
}

#if defined(__linux__) && defined(CPU_SETSIZE)
/* 枚举可用集合内的 SMT 兄弟组，把**每组的最小 CPU 号**写进 grp[]。
 * 返回组数；<=0 表示不可用（sysfs 被容器屏蔽等）。
 * 🔴 只遍历「亲和掩码 ∩ online」内的 CPU：128 核机器上不这么做会数出整槽的核。 */
static int enum_smt_groups(int *grp, int maxgrp) {
  cpu_set_t set;
  usable_cpu_set(&set);
  int any_read = 0, ngrp = 0;
  for (int c = 0; c < CPU_SETSIZE; c++) {
    if (!CPU_ISSET(c, &set)) continue;
    char path[192];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
    FILE *f = fopen(path, "r");
    if (!f) {
      if (!any_read) return -1;                 /* 第一个就读不到 ⇒ 该功能不可用 */
      continue;                                 /* 中途读不到 ⇒ 跳过，不致命 */
    }
    char line[128];
    int got = fgets(line, sizeof line, f) != NULL;
    fclose(f);
    if (!got) { any_read = 1; continue; }
    any_read = 1;
    int lo = -1;
    for (const char *p = line; *p; p++) {
      if (*p >= '0' && *p <= '9') {
        int v = 0;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        p--;
        if (lo < 0 || v < lo) lo = v;
      }
    }
    if (lo < 0) lo = c;
    if (lo == c) {                              /* 自己是组长 ⇒ 新的一组 */
      if (ngrp < maxgrp) grp[ngrp] = c;
      ngrp++;
    }
  }
  return any_read ? (ngrp > 0 ? ngrp : -1) : -1;
}
#endif

int mt_cpu_physical(void) {
  const char *e = getenv("MT_CPU_PHYS");
  if (e && *e) { int v = atoi(e); if (v > 0) return v; }
#if defined(__APPLE__)
  {
    int v = 0; size_t sz = sizeof v;
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
  }
  return -1;
#elif defined(__linux__) && defined(CPU_SETSIZE)
  {
    int grp[CPU_SETSIZE];
    int n = enum_smt_groups(grp, CPU_SETSIZE);
    return n > 0 ? n : -1;                      /* 本机 = 4 */
  }
#else
  return -1;
#endif
}

int mt_cpu_phys_mask_str(char *buf, size_t n) {
  if (!buf || n == 0) return 0;
  buf[0] = 0;
  /* env 覆写时本机 sysfs 的组信息与伪装值不符 ⇒ 不给绑核建议，免得误导 */
  const char *ea = getenv("MT_CPU_AVAIL"), *ep = getenv("MT_CPU_PHYS");
  if ((ea && *ea) || (ep && *ep)) return 0;
#if defined(__linux__) && defined(CPU_SETSIZE)
  {
    int grp[CPU_SETSIZE];
    int g = enum_smt_groups(grp, CPU_SETSIZE);
    if (g <= 0) return 0;
    size_t used = 0;
    for (int i = 0; i < g; i++) {
      int w = snprintf(buf + used, n - used, "%s%d", i ? "," : "", grp[i]);
      if (w <= 0 || (size_t)w >= n - used) break;
      used += (size_t)w;
    }
    return used > 0;
  }
#else
  return 0;
#endif
}

int mt_threads_resolve(int threads_opt, int smt_opt) {
  int avail = mt_cpu_available();
  int phys  = mt_cpu_physical();
  int T;
  if (threads_opt > 0)      T = threads_opt;              /* 显式优先，原样尊重 */
  else if (smt_opt == 0 && phys > 0) T = phys;            /* --smt 0 且探测成功 */
  else                      T = avail;
  /* 封顶：模型只有 71M，解码步含大量串行段（候选扩展 / KV 重排 / rmsnorm / rope），
   * 且最小 GEMM 只有 ((N&~3)/4) ≈ 152 个任务 ⇒ 超过 cap 的线程全是唤醒开销。
   * 128 核的 AMD pod 靠这条保命（否则 152 个任务摊给 128 个线程）。 */
  const int cap = env_or("MT_THREAD_CAP", 16);
  if (T > cap) T = cap;
  if (T < 1)   T = 1;
  return T;
}
