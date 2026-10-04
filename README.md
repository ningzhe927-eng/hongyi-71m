# HongYi-71M（鸿译）— zh↔en Preview

A compact 71M-parameter Chinese↔English translation model with a **self-contained C inference engine** (int8, bit-exact SIMD, zero GPU, zero Python).

English summary below — 中文说明为主。

## Highlights / 亮点

- **快**：CPU-only 推理，greedy batch=1 **165.5 句/s**（同机 PyTorch fp32 的 **10.5×**）；batch16 183.6 句/s；beam4 51.7 句/s（Intel icelake, 8 threads）
- **可移植**：二进制为 **x86-64-v3 基线 + 运行时 ISA 分派**（`avx512vnni / avx2 / scalar` 三路，启动时按 CPU 自动选路，输出逐位一致）——在无 AVX-512 的 CPU（AMD Zen1/2/3、老 Xeon）上不再 SIGILL，avx2 档比标量快 1.4~1.5×
- **小**：int8 权重直接链入可执行文件（~73MB），运行内存 **65~78 MB**，运行时依赖仅 `libm / libgomp / libc`
- **干净**：无 Python、无模型框架依赖；单文件二进制即可用
- 质量（sacrebleu，beam4 lp0.6 cap128，zh→en 用 `13a`、en→zh 用 `zh` 分词）：

| 测试集 | zh→en | en→zh |
|---|---|---|
| FLORES-200 devtest (1012 句) | 25.8 | **42.0** |
| IWSLT / TED (1000 句) | 21.7 | 29.1 |
| OpenSubtitles (1000 句) | **22.1** | **22.7** |
| Tatoeba (1000 句) | 33.8 | 40.3 |

与 Firefox Translations（同评测口径，beam4）对比：**en→zh 四个测试集全部领先（+2.8 ~ +3.2）**；zh→en 在字幕域（OpenSubtitles）领先 +2.2，正式书面域落后 ~3 BLEU。

## Quick start / 快速上手

预编译 Linux x64 二进制已包含在仓库（权重已链入）：

```sh
# 中文 → 英文（方向按文本自动判定）
cat input_zh.txt | head -100 > src.txt
c_infer/bin/mt_infer_linux_x64 --src_file src.txt --batch 1 --threads 8 \
    --cap 128 --max_new 128 --beam 4 --lp_alpha 0.6 > out.txt
```

- `--recase`（句首大写，规则 B）**默认开启**；`--no_recase` 关闭
- `--split_sent`：多句/段落源先分句再译（长段落 zh→en 可 +0.6~0.7 BLEU），opt-in
- `--nbest N`：额外输出 N 条候选（供外部重排），opt-in
- 翻译方向：按输入语言自动判定；也可显式 `--en2zh`

从源码构建（需要 CMake ≥3.16 + gcc ≥9，权重以 C 源码形式提供，见 Release 附件 `gen_int8_345k.tar.zst`）：

```sh
tar xf gen_int8_345k.tar.zst   # 得到 gen_q/（int8 权重 C 源码）
cd c_infer
cmake -B build -DMT_GEN_DIR=.. -DMT_QUANT8=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8        # 产物 build/mt_infer
```

- `MT_ISA=portable|native`（默认 `portable`）：`portable` = x86-64-v3 基线 + 启动时三路运行时分派；`native` = 按构建机 ISA 直编（仅自用）
- `MT_TUNE=<arch>`：可选 `-mtune` 微调（只影响调度不改指令集）
- 运行时 `MT_ISA_FORCE=scalar|avx2|avx512` 可强制选路（启动行 `[isa] ...` 会打印实际分派；逐位一致，可作验证）

## Model / 模型

| 项 | 值 |
|---|---|
| 架构 | 编码器-解码器 Transformer 变体（7 enc + 3 dec），gated attention input + QK-RMSNorm，tied embedding |
| 参数量 | 71.0M |
| 词表 | 30,008 BPE（自训，zh+en 联合） |
| 推理引擎 | 自研 C（int8 VNNI GEMM，逐位一致 SIMD 超越函数，PGO） |
| PyTorch 权重 | Release 附件 `ckpt_slim.pt`（fp16 存储的研究版权重；preview 未附 Python 加载器，C 引擎为主要使用路径） |

训练数据 ≈ 40M 中英平行对（网络挖掘平行语料 + 字幕域数据 + 约 28.5% 由 Hy-MT2-7B（Apache 2.0）合成的 wiki 域翻译）。

## License / 许可

- **代码**（`c_infer/`）：[MIT](LICENSE-CODE)
- **模型权重**（预编译二进制、`ckpt_slim.pt`、`gen_q` 权重）：[CC-BY-NC-4.0](LICENSE-WEIGHTS)
  - 训练语料含 CCMatrix（LDC 许可，限制商用）成分，故权重采用非商业许可（与 NLLB 等先例一致）
  - 后续版本（200M）计划以完全可商用语料重建训练集

## Known limitations / 已知局限

- zh→en 正式书面域（FLORES/IWSLT/Tatoeba）落后于更大的公开系统 ~3 BLEU；字幕域已持平或领先
- 多句/段落源会被压缩（翻译偏短）——建议开 `--split_sent`
- 罕见专名/学名可能保留英文原文
- 输入超 128 token 会被截断（stderr 会提示）

---

### English summary

HongYi-71M: 71M-param zh↔en NMT with a self-contained C inference engine (int8 VNNI, PGO): **165.5 sent/s greedy on 8 CPU threads (10.5× PyTorch)**, 65–78 MB RAM, single ~73 MB binary, no Python. en→zh beats Firefox Translations on all four test sets (+2.8~+3.2 BLEU); zh→en trails on formal domains (~3 BLEU) but leads on subtitles. Code: MIT. Weights: CC-BY-NC-4.0 (CCMatrix-derived training data).
