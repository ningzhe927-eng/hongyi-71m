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

# ⚫ ISA 目标（2026-10-04 新增，解 SIGILL）
#   portable（默认）= 全工程 `-march=x86-64-v3`（AVX2 基线，覆盖 2013+ x86）
#                    + `vnni_gemm.c` 编译两遍（标量 / AVX-512 VNNI）+ 启动时运行时分派
#                    ⇒ 无 AVX-512 的 CPU（AMD Zen1/2/3、老 Xeon、部分云 SKU）**不再 SIGILL**
#   native          = 旧行为（`-march=native`，单条路，绑死构建机 ISA）—— 仅本机自测用
set(MT_ISA "portable" CACHE STRING "portable(基线+运行时分派) | native(-march=native，旧行为)")
# `-mtune` **只影响指令调度、不产生新指令**（对可移植性无影响）；默认 generic，比 native 慢 6~14%。
# 想要"构建机最优调度"用 native；想要可复现/中性调度用 generic。
set(MT_TUNE "native" CACHE STRING "generic | native（-mtune；改调度不改 ISA）")
if(MT_ISA STREQUAL "portable")
  set(MT_MARCH "-march=x86-64-v3" "-mtune=${MT_TUNE}")   # 列表 ⇒ 展开成两个参数
elseif(MT_ISA STREQUAL "native")
  set(MT_MARCH "-march=native")
else()
  message(FATAL_ERROR "MT_ISA 只能是 portable / native，收到 '${MT_ISA}'")
endif()
set(MT_MARCH_V3 "-march=x86-64-v3" "-mtune=${MT_TUNE}")  # 变体 TU 用（恒定 x86-64-v3 基线）

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
  target_compile_options(${tgt} PRIVATE -O3 ${MT_MARCH} -Wall -Wextra ${MT_SAFE_FP_FLAGS})
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

# ---- 含 SIMD 的 TU 的 ISA 变体（2026-10-04）----
#   portable：把 `src/vnni_gemm.c`（int8 GEMM）与 `src/ops.c`（逐元素/归一化）
#             各编译两遍（同名源，用 -DMT_ISA_SUFFIX 改对外符号）
#             · _scalar = 基线 x86-64-v3 ⇒ 走文件内标量回退 / 标量 expf
#             · _avx512 = +AVX-512 VNNI  ⇒ 走 vpdpbusd / 位精确 SIMD expf
#   全部对外符号由 `src/isa_dispatch.c` 在启动时按 CPU 选路（调用点零改动）。
if(MT_ISA STREQUAL "portable")
  set(MT_ISA_VARIANTS
      mt_isa_vnni_scalar:src/vnni_gemm.c:_scalar
      mt_isa_vnni_avx2:src/vnni_gemm.c:_avx2
      mt_isa_vnni_avx512:src/vnni_gemm.c:_avx512
      mt_isa_ops_scalar:src/ops.c:_scalar
      mt_isa_ops_avx2:src/ops.c:_avx2
      mt_isa_ops_avx512:src/ops.c:_avx512)
  set(MT_ISA_OBJS "")
  foreach(_spec ${MT_ISA_VARIANTS})
    string(REPLACE ":" ";" _p "${_spec}")
    list(GET _p 0 _lib)
    list(GET _p 1 _src)
    list(GET _p 2 _sfx)
    add_library(${_lib} OBJECT ${_src})
    target_include_directories(${_lib} PRIVATE
      ${CMAKE_SOURCE_DIR}/${MT_GEN_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/gen_tok)
    target_compile_options(${_lib} PRIVATE -O3 ${MT_MARCH_V3} -Wall -Wextra
                           ${MT_SAFE_FP_FLAGS} -DMT_ISA_SUFFIX=${_sfx})
    if(MT_PGO STREQUAL "gen")
      target_compile_options(${_lib} PRIVATE -fprofile-generate -fprofile-dir=${MT_PGO_DIR})
    elseif(MT_PGO STREQUAL "use")
      target_compile_options(${_lib} PRIVATE -fprofile-use -fprofile-correction
                            -Wno-missing-profile -fprofile-dir=${MT_PGO_DIR})
    endif()
    if(MT_QUANT8)
      target_compile_definitions(${_lib} PRIVATE MT_QUANT8=1)
    endif()
    if(OpenMP_C_FOUND)
      target_link_libraries(${_lib} PRIVATE OpenMP::OpenMP_C)
    endif()
    list(APPEND MT_ISA_OBJS $<TARGET_OBJECTS:${_lib}>)
  endforeach()
  target_compile_options(mt_isa_vnni_avx2 PRIVATE -DMT_IMPL_AVX2=1)
  target_compile_options(mt_isa_ops_avx2 PRIVATE -DMT_EXPF8_AVX2=1)
  foreach(_lib mt_isa_vnni_avx512 mt_isa_ops_avx512)
    target_compile_options(${_lib} PRIVATE
      -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni)
  endforeach()
  set(MT_VNNI_SRC src/isa_dispatch.c ${MT_ISA_OBJS})
  set(MT_OPS_SRC "")            # ops 的两个变体已在 MT_ISA_OBJS 里
else()
  set(MT_VNNI_SRC src/vnni_gemm.c)
  set(MT_OPS_SRC src/ops.c)
endif()

add_executable(mt_infer
  src/main.c src/model.c ${MT_OPS_SRC} src/blas.c src/rope.c src/quant.c ${MT_VNNI_SRC} src/prof.c
  src/tokenizer_bpe.c src/decode.c src/postproc.c src/cpu_topology.c ${MT_GEN_WEIGHTS} ${MT_TOK_SRC})
mt_common(mt_infer)

set_source_files_properties(src/decode.c PROPERTIES COMPILE_OPTIONS "-g")

# ⚠ 必须带 prof.c：vnni_gemm.c 现在引用 mt_prof_on/mt_prof_add（激活量化探针）
add_executable(bench_kernel tests/bench_kernel.c src/blas.c ${MT_VNNI_SRC} src/quant.c ${MT_OPS_SRC} src/prof.c
               ${MT_GEN_WEIGHTS})
mt_common(bench_kernel)
