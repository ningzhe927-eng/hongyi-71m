#ifndef MT_ISA_SUFFIX_H
#define MT_ISA_SUFFIX_H

/* 同一份源码编译多遍（baseline / avx512）时的符号后缀。
 *
 * 由 CMake 传 `-DMT_ISA_SUFFIX=_scalar` 或 `-DMT_ISA_SUFFIX=_avx512`。
 * 未定义 ⇒ `MT_ISAN(sym)` 就是 sym 本身（= 旧行为，逐位不变）。
 *
 * 用法：把**对外符号**写成 `MT_ISAN(mt_gemm_q8)`，`static` 函数不用改。
 */
#ifdef MT_ISA_SUFFIX
#  define MT_JOIN2_(a, b) a##b
#  define MT_JOIN2(a, b) MT_JOIN2_(a, b)
#  define MT_ISAN(sym) MT_JOIN2(sym, MT_ISA_SUFFIX)
#else
#  define MT_ISAN(sym) sym
#endif

#endif /* MT_ISA_SUFFIX_H */
