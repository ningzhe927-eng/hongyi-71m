cmake_minimum_required(VERSION 3.16)
project(mt50m_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 权重目录：gen（fp32）或 gen_q（int8）
set(MT_GEN_DIR "gen" CACHE STRING "gen (fp32) or gen_q (int8)")
if(MT_GEN_DIR MATCHES "^gen_q")
  set(MT_QUANT8_DEFAULT ON)
else()
  set(MT_QUANT8_DEFAULT OFF)
endif()
set(MT_QUANT8 ${MT_QUANT8_DEFAULT} CACHE BOOL "权重量化为 int8（自定义 gen 目录时需显式开）")

# ⚠ 绝不加 -ffast-math：会破坏与 PyTorch 的逐 token 对齐
file(GLOB MT_GEN_WEIGHTS ${CMAKE_SOURCE_DIR}/${MT_GEN_DIR}/weights/*.c)
set(MT_TOK_SRC ${CMAKE_SOURCE_DIR}/gen_tok/tok_tables.c)

find_package(OpenMP)

# ⚫ 浮点标志（2026-10-01 实测，见 runs/local_eval/c_infer_bench/REPORT.md §七-b）
#   🔴 绝不加 -ffast-math / -funsafe-math-optimizations / -Ofast / -fassociative-math：
#      会重排浮点 ⇒ 破坏与 PyTorch 的逐 token 对齐（实测 300 句里有 20 句译文不同）。
#   ✅ 但下面这两个**不改 IEEE 语义**（只是声明不检查 errno、不触发陷阱），白拿 ~10%：
#      int8 profiler 2430 → 2212 ms，300 句输出 diff 0。
#   🔴 -mprefer-vector-width=512 也别加：本机 -march=native=icelake-server，gcc 默认 256 位是
#      为避免 512 位降频；实测 512 位慢 8%（2656 ms）且改数值（6/300 不同）。
set(MT_SAFE_FP_FLAGS -fno-math-errno -fno-trapping-math)

# PGO（两阶段）：-DMT_PGO=gen 构建 → 跑一遍代表性负载 → -DMT_PGO=use 重新 configure + 构建
#   实测 int8 档在 noerrno 之上再 +5%（2212 → ~2106 ms），300 句输出 diff 0。
#   ⚠ profile 只对"采集时那个形状"最优（本轮 = 200 句 / batch=1 / greedy 采集）
#      ⇒ 换主力负载（beam4 / batch=16）要用对应负载重新采集，否则可能反噬。
set(MT_PGO "" CACHE STRING "PGO: 空=关 | gen=采集 | use=使用")
set(MT_PGO_DIR "${CMAKE_BINARY_DIR}/pgo" CACHE STRING "PGO profile 目录")

# 公共：源文件 + 权重 + 编译选项
function(mt_common tgt)
  target_include_directories(${tgt} PRIVATE
    ${CMAKE_SOURCE_DIR}/${MT_GEN_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/gen_tok)
  target_compile_options(${tgt} PRIVATE -O3 -march=native -Wall -Wextra ${MT_SAFE_FP_FLAGS})
  if(MT_PGO STREQUAL "gen")
    target_compile_options(${tgt} PRIVATE -fprofile-generate -fprofile-dir=${MT_PGO_DIR})
    target_link_options(${tgt} PRIVATE -fprofile-generate -fprofile-dir=${MT_PGO_DIR})
  elseif(MT_PGO STREQUAL "use")
    # ⚠ -Wno-missing-profile：未采集到的 TU（如 bench_kernel）会刷一堆 [-Wmissing-profile]，
    #   -fprofile-correction 已经兜住（按 0 计数处理），不是错误。
    target_compile_options(${tgt} PRIVATE -fprofile-use -fprofile-correction -Wno-missing-profile
                                          -fprofile-dir=${MT_PGO_DIR})
    target_link_options(${tgt} PRIVATE -fprofile-use -fprofile-correction -fprofile-dir=${MT_PGO_DIR})
  elseif(NOT MT_PGO STREQUAL "")
    message(FATAL_ERROR "MT_PGO 只能是 空 / gen / use，收到 '${MT_PGO}'")
  endif()
  set_source_files_properties(${MT_GEN_WEIGHTS} PROPERTIES COMPILE_OPTIONS "-O0")
  target_link_libraries(${tgt} PRIVATE m)
  if(OpenMP_C_FOUND)
    target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
  endif()
  if(MT_QUANT8)
    target_compile_definitions(${tgt} PRIVATE MT_QUANT8=1)
    # 🔴 量化路径不调用 BLAS，且绝不链接 libopenblas：
    #    其构造器会按逻辑 CPU 数（64，无视 8 核 cgroup）起满线程池，每进程白烧 ~150ms
  else()
    find_library(OPENBLAS_LIB openblas REQUIRED)
    target_link_libraries(${tgt} PRIVATE ${OPENBLAS_LIB})
  endif()
endfunction()

add_executable(mt_infer
  src/main.c src/model.c src/ops.c src/blas.c src/rope.c src/quant.c src/vnni_gemm.c src/prof.c
  src/tokenizer_bpe.c src/decode.c src/postproc.c ${MT_GEN_WEIGHTS} ${MT_TOK_SRC})
mt_common(mt_infer)

set_source_files_properties(src/decode.c PROPERTIES COMPILE_OPTIONS "-g")

# ⚠ 必须带 prof.c：vnni_gemm.c 现在引用 mt_prof_on/mt_prof_add（激活量化探针）
add_executable(bench_kernel tests/bench_kernel.c src/blas.c src/vnni_gemm.c src/quant.c src/ops.c src/prof.c
               ${MT_GEN_WEIGHTS})
mt_common(bench_kernel)
