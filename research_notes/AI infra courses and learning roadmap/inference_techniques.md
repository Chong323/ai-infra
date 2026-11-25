# LLM Inference & Serving Optimization: Concept Map, Canonical Papers, Explainers, Handbooks and Exercises (state as of 2026-11-24)

*How this was verified (2026-11-24):* every URL below was HTTP-checked from this environment unless flagged; dates come from page metadata, URL slugs or the arXiv API (published / latest-version dates); GitHub stars and "last push" come from the ungh.cc GitHub API proxy (the session's own GitHub API access was blocked), so star counts are approximate snapshots. "From memory" marks content I could not re-verify in this session. Domain moves seen: blog.vllm.ai now redirects to vllm.ai/blog; lmsys.org to www.lmsys.org; hao-ai-lab.github.io to haoailab.com; bentoml.com/llm to handbook.modular.com.

Levels: B = beginner, I = intermediate, A = advanced. Language is English unless noted.

---

## Q1. What fundamentals must a beginner understand first, and which resources explain them best?

### Takeaway
The non-negotiable foundations are: (1) prefill is a compute-heavy parallel pass over the prompt, while decode generates one token per step and is memory-bandwidth-bound at small batch; (2) the roofline / arithmetic-intensity argument tells you how large a batch must get before decode stops being bandwidth-bound (about 200 to 300 tokens on A100/TPU v5e/H100-class hardware); (3) KV-cache size per token = 2 (K and V) x layers x KV-heads x head_dim x bytes, which, with weights, decides concurrency; (4) the metrics TTFT, ITL/TPOT, E2E latency, throughput, and goodput under SLOs. The best starting set is still kipply's "Transformer Inference Arithmetic" (2022) plus the "How to Scale Your Model" roofline and inference chapters (2025, maintained). Add Hugging Face's "Continuous batching from first principles" (Nov 2025) and the Modular (formerly BentoML) LLM Inference Handbook as the up-to-date map. The NVIDIA (2023) and Lilian Weng (2023) overviews are still good to read but are dated.

### Cited Findings

**Core concepts, with the sources that explain them best**
- Prefill and generation have very different bottlenecks. The scaling book stresses "how different the answers will be for prefill versus generation" when you ask whether a step is compute- or memory-bound — [How to Scale Your Model: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)
- Bandwidth-bound decode, with numbers: on an A100, kipply derives a FLOPs-to-bandwidth "ratio here of 208". Computing K/V for one token "will take the same amount of time to compute for up to 208 tokens! Anything below, we're memory bandwidth bound" — [kipply, Transformer Inference Arithmetic (2022-03-30)](https://kipp.ly/transformer-inference-arithmetic/)
- Roofline and arithmetic intensity: for a batch of B tokens multiplied by D x F weights, the step is compute-bound when 2BDF / (2BD + 2DF + 2BF) is at least the hardware intensity. For TPU v5e that intensity is 1.97e14 / 8.2e11 = 240 FLOPs/byte, and the condition simplifies to B of roughly 240 when D and F are much larger than B — [scaling book, inference chapter](https://jax-ml.github.io/scaling-book/inference/); background in [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
- FLOP:byte ratios of real hardware: Ryzen 7950X has 67 GB/s and 2,735 GFLOPS (40:1). RTX 4090 has 1,008 GB/s and 83 TFLOPS (82:1). H100 SXM has 3,350 GB/s of memory bandwidth, but tensor cores make matmul-shaped work far more compute-dense — [zeux, LLM inference speed of light (2024-03-15)](https://zeux.io/2024/03/15/llm-inference-sol/)
- KV-cache sizing formula: "Per token, the number of bytes we store is 2·2·n_layers·n_heads·d_head". The first 2 counts K and V; the second is bytes per fp16 element. This is the MHA form; for GQA, substitute KV heads (see Q5) — [kipply](https://kipp.ly/transformer-inference-arithmetic/)
- Overhead is the third regime, alongside compute-bound and bandwidth-bound — [Horace He, Making Deep Learning Go Brrrr From First Principles (2022)](https://horace.io/brrr_intro.html). In practice, Hazy Research found that vLLM and SGLang used "at most 50% of available GPU bandwidth" for a Llama-1B low-batch decode on H100. The cause: a forward pass is split into "around a hundred separate kernels" — [Hazy Research, Look Ma, No Bubbles! (2025-05-27)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Standard metric definitions: Time to First Token, End-to-End Request Latency, Inter-token Latency, Tokens Per Second, and Requests Per Second (doc version 2.0.0, updated 2026-10-05) — [NVIDIA NIM Benchmarking: Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- Goodput and SLOs: the DistServe authors argue "Throughput is Not All You Need". They optimize goodput, meaning requests served per second that meet TTFT and TPOT SLOs — [Hao AI Lab blog (2024-03-17)](https://hao-ai-lab.github.io/blogs/distserve/); paper [DistServe, OSDI 2024 (arXiv 2401.09670)](https://arxiv.org/abs/2401.09670)
- Critique of TTFT/TPOT: Etalon proposes holistic metrics for streaming user experience, the "fluidity-index" (from memory) — [Etalon (arXiv 2407.07000, Jul 2024; v2 Aug 2024)](https://arxiv.org/abs/2407.07000)
- Reproducible benchmarking methodology (LLMPerf) — [Anyscale, Reproducible Performance Metrics for LLM inference (2023-11-01)](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference)
- The theory behind partitioning and latency/cost trade-offs for large-model inference — [Pope et al., Efficiently Scaling Transformer Inference (arXiv 2211.05102, Nov 2022; MLSys 2023 from memory)](https://arxiv.org/abs/2211.05102)

**Fundamentals resource records** (ordered roughly in suggested reading order)

| Resource | Author / maintainer | Date / last update | Format · Level · Lang | Prereqs | Key topics | Why it stands out / caveats |
|---|---|---|---|---|---|---|
| [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | Horace He | 2022 | blog · B · EN | basic PyTorch | compute vs memory-bandwidth vs overhead regimes; fusion | Best mental model of GPU bottlenecks; not LLM-specific |
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (now at kipp.ly/p/...) | kipply | 2022-03-30 (page re-rendered 2026) | blog · B→I · EN | transformer basics, FLOP counting | KV cache bytes/token, memory- vs compute-bound (A100 ratio 208), latency napkin math, batching | The canonical napkin-math primer. Caveat: A100-era numbers; MHA formula predates GQA ubiquity |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): [Rooflines](https://jax-ml.github.io/scaling-book/roofline/), [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/), [Serving LLaMA 3-70B on TPUs](https://jax-ml.github.io/scaling-book/applied-inference/) | Google DeepMind (Austin et al.); [repo jax-ml/scaling-book](https://github.com/jax-ml/scaling-book) ~1.5k★, last push 2026-10-19 | 2025 (release date from memory); maintained | online textbook with worked problems · I→A · EN | linear algebra; JAX/TPU helpful | rooflines, prefill vs generation, KV caches, batching, sharding for inference, latency bounds | Most rigorous free treatment, with exercises and answers. TPU-centric numbers, but the concepts transfer to GPUs |
| [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching) | Rémi Ouazan Reboul, Arthur Zucker, Luc Georges (Hugging Face) | 2025-11-25 | blog · B · EN | attention basics | attention, then KV cache, then chunked prefill, then continuous batching; ragged batching | Newest, gentlest derivation of the serving-engine core |
| [LLM Inference Handbook](https://handbook.modular.com/) (formerly BentoML's) | BentoML team, now Modular; [repo modular/llm-inference-handbook](https://github.com/modular/llm-inference-handbook) ~0.5k★, last push 2026-11-24 | continuously updated | web handbook (Markdown via `.md` / [llms.txt](https://handbook.modular.com/llms.txt)) · B→I · EN | none | metrics, GPU memory calculation, batching, PagedAttention, prefix caching, spec decoding, P/D disaggregation, KV offload, routing, parallelism, FlashAttention, GPU architecture | Best single practitioner map. The old bentoml.com/llm URL returned 502 on 2026-11-24 |
| [Inference Engineering (book)](https://www.baseten.co/inference-engineering/) | Philip Kiely (Baseten) | 2026 (exact launch date not pinned) | book: free PDF/online/audiobook, paperback · B→I · EN | "Anyone can read the first twenty pages"; then basic coding/CS | Ch. 0 Inference, 1 Prerequisites, 2 Models, 3 Hardware, 4 Software, 5 Techniques, 6 Modalities, 7 Production | 256 pages (~46k words). Broad and current production view; focused on NVIDIA GPUs and the datacenter |
| [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | NVIDIA Technical Blog | 2023-11-17 (metadata modified 2025-12-27) | blog · B→I · EN | basics | batching, KV cache, model parallelism, MQA/GQA, FlashAttention, PagedAttention, quantization, sparsity, distillation, speculative inference (from memory) | Compact overview. Dated: no FP4, disaggregation, wide EP or MTP |
| [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | Lilian Weng | 2023-01-10 | blog · I · EN | ML background | distillation, quantization, pruning, sparsity, MoE, efficient architectures (from memory) | Strong on model-compression literature. Outdated on serving systems (predates vLLM) |
| [A guide to LLM inference and performance](https://www.baseten.co/blog/llm-transformer-inference-guide/) | Baseten | 2023-11-17 (modified 2026-11-02) | blog · B · EN | basics | ops:byte, memory- vs compute-bound on real GPUs (from memory) | Practical GPU-sizing intuition |
| [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) | Databricks/MosaicML | 2023-10-12 | blog · I · EN | basics | TTFT/TPOT/latency/throughput, model bandwidth utilization (MBU), batching, TP, quantization (from memory) | Introduced the MBU framing. Metrics still valid |
| [Achieve 23x LLM Inference Throughput & Reduce p50 Latency](https://www.anyscale.com/blog/continuous-batching-llm-inference) | Anyscale | 2023-06-22 | blog · B · EN | basics | static vs continuous batching | Classic continuous-batching explainer |
| [LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/) | Arseny Kapoulkine | 2024-03-15 | blog · B→I · EN | basics | bandwidth-bound decode math, FLOP:byte ratios | Teaches "speed-of-light" estimation |
| [Dissecting Batching Effects in GPT Inference](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) | Lequn Chen (Punica author) | 2023-05-13 | blog · I · EN | basics | how batching changes per-op cost | Empirical intuition for batching |
| [Prefill and Decode for Concurrent Requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests) | TNG Technology Consulting (HF community blog) | 2025-04-16 | blog · I · EN | basics | prefill/decode behavior under concurrency | Concrete, recent measurements |
| [Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops) | Adam Casson | 2023-05-16 | blog · B→I · EN | basics | FLOP counting | Good companion to kipply |
| [GPU Glossary](https://modal.com/gpu-glossary) | Modal; [repo](https://github.com/modal-labs/gpu-glossary) ~0.8k★, last push 2026-09-04 | maintained | web glossary · B · EN | none | SMs, tensor cores, memory hierarchy, roofline terms | Hardware literacy for inference |
| [NVIDIA NIM: Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) | NVIDIA | updated 2026-10-05 | docs · B · EN | none | TTFT, E2E, ITL, TPS, RPS | Precise metric definitions |
| [Throughput is Not All You Need (DistServe blog)](https://hao-ai-lab.github.io/blogs/distserve/) | Hao AI Lab (UCSD) | 2024-03-17 | blog · I · EN | metrics | goodput, SLOs, P/D interference | Origin of the goodput framing |
| [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm) (also on the [vLLM blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)) | Aleksa Gordić | 2025-08-29 (vLLM blog 2025-09-05) | long-form blog · I→A · EN | Python, basics above | paged attention, continuous batching, prefix caching, spec decoding, multi-GPU/multi-node serving | Best end-to-end walk through a real engine (vLLM V1) |
| [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Horace He et al. (Thinking Machines) | 2025-09-10 | blog · I→A · EN | kernels basics | floating-point non-associativity, batch invariance (RMSNorm/matmul/attention) | Explains why batching changes outputs. Foundational for RL-era serving |

### Inferences
- Suggested beginner order: Horace He (regimes), then kipply (napkin math), then scaling-book rooflines and inference chapters (do the problems), then the HF continuous-batching post, then the Modular handbook (as a map), then the NIM metrics page plus the DistServe goodput post, then Aleksa's vLLM anatomy, then the Baseten book for a production overview. After that, move to the Q2 deep dives.
- kipply's MHA formula overstates KV for modern GQA models by n_heads/n_kv_heads; for Llama-3.1 that is 4x for 8B and 8x for 70B (configs in Q5). Learners should be told this explicitly.
- Lilian Weng (2023) and NVIDIA "Mastering" (2023) are now "history and compression context" rather than "how serving works in 2026". The Modular handbook and the Baseten book are the current equivalents.

### Gaps
- I could not confirm the exact first-publication date of "How to Scale Your Model" (believed Feb 2025) or the exact launch date of Baseten's "Inference Engineering" (2026; the site shows © 2026 and Baseten announced a "book launch").
- Etalon's "fluidity-index" naming and the Databricks MBU content are from memory; liveness and dates are verified.

---

## Q2. Concept map: each major technique — explanation, canonical papers, 2026 production status, best explainers

### Takeaway
By late 2026 the "engine core" is settled and standard everywhere: paged KV cache, continuous batching with chunked prefill, prefix caching, FlashAttention-class and FlashInfer kernels, CUDA graphs with torch.compile, GQA/MLA-aware kernels, FP8 (and NVFP4 on Blackwell), structured output, multi-LoRA, and MTP/EAGLE-3 speculative decoding. Cluster-scale techniques are now also standard at frontier scale: P/D disaggregation, wide expert parallelism with DP-attention, KV tiering/offload, and KV-aware routing. The active frontier is attention–FFN disaggregation, FP4 attention math and 4-bit KV, parallel drafting with adaptive verification, megakernels, and model-native compressed or sparse attention (DeepSeek V3.2/V4/V4.1, hybrid linear attention) co-designed with serving.

### Cited Findings

#### A. Batching and scheduling

**A1. Continuous batching (iteration-level scheduling)**
- *What:* schedule at the granularity of one model iteration, so finished sequences leave and new ones join every step instead of waiting for a whole static batch. Orca also introduced selective batching (from memory) — [Orca, OSDI 2022 (Yu et al.)](https://www.usenix.org/conference/osdi22/presentation/yu)
- *Status 2026:* standard. "Continuous batching, the technique first proposed by Orca and later popularized by vLLM, by default colocates prefill and decode together" — [Hao AI Lab, Disaggregated Inference: 18 Months Later (2025-11-03)](https://hao-ai-lab.github.io/blogs/distserve-retro/)
- *Explainers:* [HF Continuous batching from first principles (2025-11-25; B)](https://huggingface.co/blog/continuous_batching); [Anyscale (2023-06-22; B)](https://www.anyscale.com/blog/continuous-batching-llm-inference); [Modular handbook: static, dynamic and continuous batching (B)](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching.md); [Aleksa Gordić, Inside vLLM (2025; I→A)](https://www.aleksagordic.com/blog/vllm)

**A2. Chunked prefill and mixed prefill/decode batches**
- *What:* split long prompts into chunks and co-schedule them with ongoing decodes under a per-step token budget. This is "stall-free" scheduling: decodes are not blocked by long prefills, and the GPU stays saturated — [SARATHI (arXiv 2308.16369, Aug 2023)](https://arxiv.org/abs/2308.16369); [Sarathi-Serve, "Taming Throughput-Latency Tradeoff..." (arXiv 2403.02310; OSDI 2024 from memory)](https://arxiv.org/abs/2403.02310); [DeepSpeed-FastGen / Dynamic SplitFuse (arXiv 2401.08671, Jan 2024)](https://arxiv.org/abs/2401.08671)
- *Kernel-level overlap of prefill and decode:* [POD-Attention (arXiv 2410.18038; ASPLOS 2025)](https://arxiv.org/abs/2410.18038)
- *Status 2026:* standard; a documented tuning knob in vLLM — [vLLM docs: Optimization and Tuning (updated 2026-11-22)](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- *Explainers/code:* the HF post above has a chunked-prefill section; [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve) (~0.5k★, last push 2026-01-08)

#### B. KV-cache memory management

**B1. PagedAttention (paged KV cache)**
- *What:* store KV in fixed-size blocks indexed by a per-sequence block table, like OS virtual memory. This removes fragmentation and enables sharing and copy-on-write across sampled sequences — [Kwon et al., SOSP 2023 (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180)
- *Alternative:* keep KV virtually contiguous using CUDA virtual-memory APIs — [vAttention (arXiv 2405.04437; ASPLOS 2025)](https://arxiv.org/abs/2405.04437)
- *Status 2026:* standard (the default KV manager in vLLM/SGLang-class engines) — [Modular handbook: PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention.md)
- *Explainers:* [vLLM launch blog (2023-06-20; B)](https://blog.vllm.ai/2023/06/20/vllm.html); [Aleksa Gordić (I→A)](https://www.aleksagordic.com/blog/vllm)

**B2. Prefix caching / RadixAttention**
- *What:* reuse the KV of shared prefixes (system prompts, few-shot examples, multi-turn history, agent trajectories). SGLang keeps a radix tree of cached token sequences with eviction and cache-aware scheduling; vLLM's Automatic Prefix Caching hashes full KV blocks — [SGLang (arXiv 2312.07104; NeurIPS 2024)](https://arxiv.org/abs/2312.07104); [vLLM design doc: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- *Related:* [Prompt Cache (arXiv 2311.04934; MLSys 2024)](https://arxiv.org/abs/2311.04934); shared-prefix attention kernels [Hydragen (arXiv 2402.05099)](https://arxiv.org/abs/2402.05099) and [FlashInfer Cascade Inference (2024-02-02)](https://flashinfer.ai/2024/02/02/cascade-inference.html)
- *Status 2026:* standard. Agentic workloads make it critical: in a Qwen3-Coder-480B coding-agent scenario "dialogues often stretched past 25K tokens around 8 turns per session. Without full KV cache retention, nearly every request required costly re-computation" — [SGLang HiCache (2025-09-10)](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- *Explainers:* [LMSYS RadixAttention blog (2024-01-17; B→I)](https://lmsys.org/blog/2024-01-17-sglang/); [Modular handbook: prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching.md)

**B3. KV-cache offloading, tiering and sharing (LMCache, Mooncake, HiCache)**
- *What:* extend KV capacity beyond HBM into CPU DRAM, local NVMe and distributed storage. KV can also be shared across engine instances and moved between prefill and decode nodes.
- *Canonical:* [FlexGen (arXiv 2303.06865)](https://arxiv.org/abs/2303.06865) (offloading for throughput); [CacheGen (arXiv 2310.07240; SIGCOMM'24)](https://arxiv.org/abs/2310.07240); [CacheBlend (arXiv 2405.16444)](https://arxiv.org/abs/2405.16444) (KV fusion for non-prefix RAG reuse); [Mooncake (arXiv 2407.00079; latest v5 2026-10-08)](https://arxiv.org/abs/2407.00079), the KVCache-centric architecture serving Kimi; [LMCache (arXiv 2510.09665; Oct 2025, v2 Dec 2025)](https://arxiv.org/abs/2510.09665)
- *Status 2026:* standard for multi-turn/agentic serving at scale. Evidence:
  - [vLLM KV Offloading Connector (2026-01-08)](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
  - [vLLM x Mooncake Store for agentic workloads (2026-05-06)](https://vllm.ai/blog/2026-05-06-mooncake-store)
  - [Tiered KV offloading across host memory, filesystems, object stores and remote peers (2026-09-10)](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading)
  - On MI355X, an NVMe KV pool "matched a DRAM pool of the same size within 1%" for Kimi-K3 — [vLLM, Skip the DRAM Tier (2026-11-23)](https://vllm.ai/blog/2026-11-23-skip-the-dram-tier)
- *Frontier:* agent-directed KV management for edited or retried trajectories — [Leyline (arXiv 2606.01065, May 2026)](https://arxiv.org/abs/2606.01065)
- *Explainers:* [Modular handbook: KV cache offloading](https://handbook.modular.com/inference-optimization/kv-cache-offloading.md); [LMCache blog](https://blog.lmcache.ai/); [SGLang HiCache (2025-09-10)](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- *Repos:* [LMCache](https://github.com/LMCache/LMCache) ~12.5k★ (active 2026-11-24); [Mooncake](https://github.com/kvcache-ai/Mooncake) ~7.1k★ (active); [DeepSeek 3FS](https://github.com/deepseek-ai/3FS) ~10.5k★ (active)

#### C. Attention kernels

**C1. FlashAttention (v1 to v4)**
- *What:* exact attention computed tile-by-tile in on-chip SRAM with an online softmax, so the N x N score matrix never goes to HBM ("IO-aware") — [FA1 (arXiv 2205.14135, May 2022)](https://arxiv.org/abs/2205.14135). Later versions:
  - FA2: better parallelism and work partitioning — [arXiv 2307.08691](https://arxiv.org/abs/2307.08691)
  - FA3: Hopper asynchrony and FP8 — [arXiv 2407.08608](https://arxiv.org/abs/2407.08608)
  - FA4: Blackwell co-design for "asymmetric hardware scaling", where tensor-core throughput doubles but exponential units and shared-memory bandwidth do not — [FlashAttention-4 (arXiv 2603.05451, 2026-03-05; in the MLSys 2026 proceedings)](https://arxiv.org/abs/2603.05451), [MLSys 2026 PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf)
- *Numbers:*
  - FA3 is "1.5-2.0x faster than FlashAttention-2 with FP16, up to 740 TFLOPS, i.e., 75% utilization of H100", and close to 1.2 PFLOPS in FP8 — [Tri Dao, FA3 blog (Jul 2024)](https://tridao.me/blog/2024/flash3/)
  - FA4 reaches up to 1.3x over cuDNN 9.13 and 2.7x over Triton on B200 (BF16), up to 1,613 TFLOPs/s (71% utilization). It is written in CuTe-DSL with 20–30x faster compile times — [arXiv 2603.05451](https://arxiv.org/abs/2603.05451)
- *Status 2026:* standard (FA2/FA3/FA4/cuDNN-class fused attention). FP4 attention math is still frontier: "Direct-P" reaches up to 2.13x BF16 forward throughput on GB200 — [Hardware-Aware FP4 FlashAttention-4 (arXiv 2609.04105, Sep 2026)](https://arxiv.org/abs/2609.04105)
- *Explainers:*
  - [UW CSE 599M notes "From Online Softmax to FlashAttention" (2023; PDF; I→A; author Zihao Ye from memory)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
  - [Modal, We reverse-engineered Flash Attention 4 (2025-09-26; A)](https://modal.com/blog/reverse-engineer-flash-attention-4)
  - [Lambda FA4 explainer (2026-04-27; I)](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet)
  - [Modular handbook: FlashAttention (B→I)](https://handbook.modular.com/kernel-optimization/flashattention.md)
  - [PyTorch FlexAttention (2024-08-07; I)](https://pytorch.org/blog/flexattention/) for programmable variants
  - Code: [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal), "~100 lines of CUDA" forward pass (~1.2k★, last push 2024-12-30)
  - Flag: Aleksa Gordić's "ELI5: FlashAttention" is on Medium ([link](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)). It returned a Cloudflare 403 to automated fetching, so it was not verified, and Medium may be metered.
- *Repo:* [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) ~25.3k★ (active 2026-11-23)

**C2. FlashDecoding (split-KV decode)**
- *What:* for long-context, small-batch decode, parallelize attention across the KV sequence dimension and then reduce, so all SMs stay busy — [Stanford CRFM, Flash-Decoding (2023-10-12)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html). Follow-up: [FlashDecoding++ (arXiv 2311.01282; v4 Jan 2024)](https://arxiv.org/abs/2311.01282)
- *Status:* standard; split-KV decode paths exist in FlashAttention/FlashInfer-class kernels (inference from the kernel libraries above).

**C3. FlashInfer**
- *What:* attention engine and kernel library for serving. It provides a unified block-sparse KV format for paged and ragged layouts, JIT-customizable attention variants, and load-balanced scheduling (from memory) — [FlashInfer (arXiv 2501.01005; MLSys 2025)](https://arxiv.org/abs/2501.01005)
- *Status 2026:* standard in vLLM/SGLang. vLLM's Blackwell wide-EP uses "FlashInfer's TRTLLM-Gen GEMM kernels optimized for GB200's FP4 tensor cores" — [vLLM Blackwell Part I (2026-02-03)](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1). Vera Rubin support used "Rubin-tuned FlashInfer kernels" — [vLLM (2026-10-09)](https://vllm.ai/blog/2026-10-09-vera-rubin-preview)
- *Explainers:* [Cascade Inference (2024-02-02)](https://flashinfer.ai/2024/02/02/cascade-inference.html); [FlashInfer 0.2 release (2024-12-16)](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html); [repo](https://github.com/flashinfer-ai/flashinfer) ~7.0k★ (active)

**C4. FlashMLA**
- *What:* DeepSeek's MLA decode kernels, released during DeepSeek's Feb 2025 Open-Source Week. They were later extended to sparse attention for V3.2 — [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) (~13.3k★, last push 2026-10-27); [open-infra-index](https://github.com/deepseek-ai/open-infra-index) (~8.1k★)
- *Status:* standard for DeepSeek-family serving. vLLM's DeepSeek-V3.2-Exp support uses "lightning indexer caches, separate prefill and decode paths, FlashMLA sparse attention, DeepGEMM" — [vLLM (2025-09-29)](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)

#### D. Attention architectures that shrink the KV cache

**D1. MQA / GQA / MLA**
- *What:* each design cuts the KV stored per token:
  - MQA shares one K/V head across all query heads — [Shazeer, Fast Transformer Decoding: One Write-Head is All You Need (arXiv 1911.02150, 2019)](https://arxiv.org/abs/1911.02150)
  - GQA shares K/V heads within groups of query heads, and MHA checkpoints can be "uptrained" into it — [Ainslie et al., EMNLP 2023 (arXiv 2305.13245)](https://arxiv.org/abs/2305.13245)
  - MLA caches a low-rank joint K/V latent plus a small decoupled RoPE key. DeepSeek-V2 reports a 93.3% KV-cache reduction and 5.76x max generation throughput vs DeepSeek 67B (numbers from memory of the abstract) — [DeepSeek-V2 (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434)
- *Status 2026:* standard. Verified configs: Llama-3.1-8B/70B, Qwen3-32B and gpt-oss-120b all use 8 KV heads (GQA); DeepSeek-V3 uses MLA with kv_lora_rank 512 and rope dim 64 — [Llama-3.1-8B config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/raw/main/config.json), [Llama-3.1-70B config](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/raw/main/config.json), [Qwen3-32B config](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json), [gpt-oss-120b config](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json), [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)
- *Explainers:*
  - [Sebastian Raschka, The Big LLM Architecture Comparison (2025-07-19; page shows "last updated" Apr 2, 2026; B→I)](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
  - Jianlin Su's Chinese-language deep dive from MHA/MQA/GQA to MLA, [kexue.fm/archives/10091](https://kexue.fm/archives/10091) (I→A; Chinese). It returned 403 to automated fetch, so liveness was not verified; title and date (May 2024) from memory.
  - Production case study of KV-centric architecture changes: [Character.AI, Optimizing AI Inference (2024-06-20)](https://blog.character.ai/optimizing-ai-inference-at-character-ai/). It credits "innovations around Transformer architecture and 'attention KV cache'" and "inter-turn caching" for serving at "less than one cent per hour of conversation". The specifics (MQA, hybrid local/global attention, cross-layer KV sharing, int8) are from memory, and the old research.character.ai URL now redirects to the blog homepage.

**D2. Long-context, sparse, and hybrid/linear attention**
- *Training-free KV eviction/selection:*
  - [StreamingLLM / attention sinks (arXiv 2309.17453; ICLR 2024)](https://arxiv.org/abs/2309.17453)
  - [H2O heavy-hitter oracle (arXiv 2306.14048)](https://arxiv.org/abs/2306.14048)
  - [SnapKV (arXiv 2404.14469)](https://arxiv.org/abs/2404.14469)
  - [Quest query-aware sparsity (arXiv 2406.10774; ICML 2024)](https://arxiv.org/abs/2406.10774)
  - [MInference dynamic sparse prefill (arXiv 2407.02490; NeurIPS 2024 Spotlight)](https://arxiv.org/abs/2407.02490)
- *Natively trained sparse/compressed attention:*
  - [NSA: Native Sparse Attention (arXiv 2502.11089, Feb 2025)](https://arxiv.org/abs/2502.11089)
  - [MoBA (Moonshot; arXiv 2502.13189)](https://arxiv.org/abs/2502.13189)
  - DeepSeek Sparse Attention with a lightning indexer — [DeepSeek-V3.2 (arXiv 2512.02556, Dec 2025)](https://arxiv.org/abs/2512.02556)
  - DeepSeek-V4's "hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)". At 1M-token context, V4-Pro needs "only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2". V4-Pro is 1.6T params (49B active) and V4-Flash is 284B (13B active) — [DeepSeek-V4 (arXiv 2606.19348, 2026-04-26)](https://arxiv.org/abs/2606.19348)
  - DeepSeek-V4.1-Flash combines "cross-layer KV cache reuse in ... CSA2 with FP4 KV caching", bringing the global KV footprint to 890 bytes per token (about 1/4 of V4-Flash). "SWA Bounded Replay" cuts the persistent SSD/host KV to about 1/8, and decode activates 16B params vs 8B in prefill — [DeepSeek-V4.1-Flash (arXiv 2609.19969, 2026-09-17)](https://arxiv.org/abs/2609.19969)
- *Linear/hybrid attention:* [Kimi Linear (arXiv 2510.26692, Oct 2025)](https://arxiv.org/abs/2510.26692). Qwen3.8-2.4T-A95B's "hybrid attention architecture puts pressure on most of the assumptions a se[rving engine makes]", and SGLang's cache controller coordinates KV and Gated-DeltaNet state across tiers — [LMSYS, Qwen3.8 Day-0 (2026-08-12)](https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/). See also [vLLM disaggregated serving for hybrid SSM models (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg)
- *Status 2026:* model-native sparse, compressed and hybrid attention ships in production models with day-0 engine support (DeepSeek V3.2/V4/V4.1, Qwen3.8, Kimi Linear). Training-free eviction (H2O/SnapKV/Quest) remains mostly research or optional (inference; see Q3).
- *Explainers:*
  - [vLLM, DeepSeek V4 in vLLM: Efficient Long-context Attention, "a first-principles walkthrough" (2026-04-24; I→A)](https://vllm.ai/blog/2026-04-24-deepseek-v4)
  - [vLLM, DeepSeek-V3.2-Exp sparse attention (2025-09-29)](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)
  - [SGLang Day-0 DeepSeek-V3.2 (2025-09-29)](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/)
  - [NVIDIA, Build with DeepSeek V4 (2026-04-24)](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)
  - Reading lists: [Xnhyacinth/Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) (~2.2k★, active 2026-11-13); [Zefan-Cai/Awesome-LLM-KV-Cache](https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache) (~0.5k★, last push 2026-06-17)
- *Repos:* [streaming-llm](https://github.com/mit-han-lab/streaming-llm) ~7.3k★ (last push 2024-07, archival); [MInference](https://github.com/microsoft/MInference) ~1.2k★ (2026-10); [MoBA](https://github.com/MoonshotAI/MoBA) ~2.2k★ (2025-04); [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) ~6.1k★ (active)

#### E. Quantization (weights, activations, KV cache; FP8/FP4)
- *What:* lower precision cuts memory and bandwidth and unlocks low-precision tensor cores:
  - Weight-only methods help bandwidth-bound decode: GPTQ, a second-order one-shot method — [arXiv 2210.17323; ICLR 2023](https://arxiv.org/abs/2210.17323); AWQ, activation-aware scaling — [arXiv 2306.00978; "MLSys 2024 Best Paper Award"; v6 updated 2026-04-25](https://arxiv.org/abs/2306.00978)
  - W8A8 helps compute-bound prefill: [LLM.int8() (NeurIPS 2022; arXiv 2208.07339)](https://arxiv.org/abs/2208.07339); [SmoothQuant (ICML 2023; arXiv 2211.10438)](https://arxiv.org/abs/2211.10438); [FP8 formats E4M3/E5M2 (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433)
  - Rotations remove outliers for 4-bit: [QuaRot (arXiv 2404.00456)](https://arxiv.org/abs/2404.00456); [SpinQuant (ICLR 2025; arXiv 2405.16406)](https://arxiv.org/abs/2405.16406)
  - Systems co-design: [QServe W4A8KV4 (MLSys 2025; arXiv 2405.04532)](https://arxiv.org/abs/2405.04532); [Marlin mixed-precision kernel (arXiv 2408.11743)](https://arxiv.org/abs/2408.11743)
  - Microscaling block formats: [MX formats (arXiv 2310.10537)](https://arxiv.org/abs/2310.10537); [NVIDIA NVFP4 (2025-06-24)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/); [Pretraining LLMs with NVFP4 (arXiv 2509.25149; v2 2026-03-04)](https://arxiv.org/abs/2509.25149)
  - KV-cache quantization: [KIVI 2-bit (ICML 2024; arXiv 2402.02750)](https://arxiv.org/abs/2402.02750); [KVQuant (NeurIPS 2024; arXiv 2401.18079)](https://arxiv.org/abs/2401.18079)
- *Status 2026:*
  - FP4 MoE experts are standard at frontier scale on Blackwell. vLLM's GB200 wide-EP used FP4 quantization of MoE expert weights and output projections — [vLLM (2026-02-03)](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1). SGLang GB200 used "FP8 attention, NVFP4 MoE" — [LMSYS Part II (2025-09-25)](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/). Other examples: [SGLang DeepSeek R1-NVFP4 on GB300 (2026-02-19)](https://www.lmsys.org/blog/2026-02-19-gb300-longctx) and [vLLM DeepSeek-V3.2 on GB300 with NVFP4 (2026-02-13)](https://vllm.ai/blog/2026-02-13-gb300-deepseek)
  - SGLang states that "Quantization has moved from an advanced feature to an essential part of high-throughput LLM serving" — [LMSYS (2026-07-28)](https://www.lmsys.org/blog/2026-07-31-cleaner-quantization-stack/)
  - FP8 KV cache is standard; see the validation across Hopper and Blackwell — [vLLM, The State of FP8 KV-Cache and Attention Quantization (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
  - NVFP4 KV cache is emerging: "up to 50%" smaller than FP8, but "the current implementation ... requires that values be dequantized from NVFP4 to FP8 before attention" — [NVIDIA (2025-12-08)](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
  - 4-bit KV schemes are still being evaluated — [vLLM TurboQuant study (2026-05-11)](https://vllm.ai/blog/2026-05-11-turboquant)
  - New quantized-kernel library — [Humming, Ant Group + vLLM (2026-10-21)](https://vllm.ai/blog/2026-10-21-humming)
- *Explainers:*
  - [Maarten Grootendorst, A Visual Guide to Quantization (2024-07-22; B)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
  - [NVIDIA NVFP4 intro (2025-06-24; I)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
  - [NVIDIA NVFP4 KV cache (2025-12-08; I)](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
  - [vLLM FP8 KV state (2026-04-22; I→A)](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
  - [MIT 6.5940 TinyML and Efficient DL, Fall 2024 (Song Han; lectures on quantization, LLM efficiency; I)](https://hanlab.mit.edu/courses/2024-fall-65940); the Fall 2025 URL returned 404
  - Red Hat's "half a million evaluations of quantized LLMs" (Oct 2024) at [developers.redhat.com](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms) returned 403 to automated fetch; not verified.
- *Tools/repos:* [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) ~4.0k★ (active); [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) ~5.7k★ (active). Paper repos: [llm-awq](https://github.com/mit-han-lab/llm-awq) ~3.7k★ (last push 2025-07); [gptq](https://github.com/IST-DASLab/gptq) ~2.4k★ (2024-03); [marlin](https://github.com/IST-DASLab/marlin) ~1.2k★ (2024-09); [smoothquant](https://github.com/mit-han-lab/smoothquant) ~1.7k★ (2024-07); [omniserve/QServe](https://github.com/mit-han-lab/omniserve) ~0.9k★ (2025-03); [KIVI](https://github.com/jy-yuan/KIVI) ~0.4k★ (2025-11). Paper repos are archival; production implementations live in the engines and tools.

#### F. Speculative decoding and multi-token prediction (MTP)
- *What:* a cheap drafter proposes k tokens, and the target model verifies them in one forward pass. A modified rejection-sampling step keeps the output distribution unchanged (lossless) — [Leviathan et al. (arXiv 2211.17192; ICML 2023 Oral)](https://arxiv.org/abs/2211.17192); [Chen et al., DeepMind (arXiv 2302.01318)](https://arxiv.org/abs/2302.01318). Drafter families:
  - Tree verification: [SpecInfer (ASPLOS'24; arXiv 2305.09781)](https://arxiv.org/abs/2305.09781)
  - Extra heads: [Medusa (arXiv 2401.10774)](https://arxiv.org/abs/2401.10774)
  - Feature-level drafting: [EAGLE (arXiv 2401.15077)](https://arxiv.org/abs/2401.15077); dynamic trees in [EAGLE-2 (arXiv 2406.16858)](https://arxiv.org/abs/2406.16858); token-level drafting with multi-layer features, trained with "training-time test", in [EAGLE-3 (arXiv 2503.01840)](https://arxiv.org/abs/2503.01840). Venues ICML'24, EMNLP'24 and NeurIPS'25 per the [EAGLE repo](https://github.com/SafeAILab/EAGLE)
  - Jacobi-style decoding: [Lookahead decoding (arXiv 2402.02057)](https://arxiv.org/abs/2402.02057)
  - Native MTP heads: [Meta, Multi-token prediction (arXiv 2404.19737)](https://arxiv.org/abs/2404.19737); the [DeepSeek-V3 MTP module (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437). The V3 report's 85–90% second-token acceptance is from memory.
  - Suffix trees for agentic workloads: [SuffixDecoding (arXiv 2411.04975; NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2411.04975)
- *2026 parallel drafters:* "Speculators and vLLM now support P-EAGLE, DFlash, and DSpark — three parallel drafting algorithms" — [vLLM, Parallel All the Way Down (2026-07-28)](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting). See also:
  - [P-EAGLE (2026-03-13)](https://vllm.ai/blog/2026-03-13-p-eagle)
  - [EAGLE 3.1 (2026-05-26)](https://vllm.ai/blog/2026-05-26-eagle-3-1)
  - [DFlash speculator for Laguna XS.2 (2026-05-28)](https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor)
  - Adaptive verification: [DSpark confidence-scheduled verification (2026-08-14)](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification); [Adaptive Verification Without a Confidence Head (2026-10-14)](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator)
- *Reality check:*
  - The first systematic study on a production engine (vLLM) covered n-gram, EAGLE/EAGLE-3, draft model and MTP. "Verification by the target model dominates the execution", acceptance "varies markedly", and there are "substantial gaps" to theoretical upper bounds — [Speculative Decoding: Performance or Illusion? (arXiv 2601.11580; Liu, ..., Stoica et al.; 2025-12-31, v2 2026-03-18)](https://arxiv.org/abs/2601.11580)
  - At high load, "drafting consumes up to 40% of step time, of which a majority is discarded" — [ForeSpark (arXiv 2611.13442, Nov 2026)](https://arxiv.org/abs/2611.13442)
  - [Energy accounting of SD under latency targets (arXiv 2610.40287)](https://arxiv.org/abs/2610.40287)
- *Status 2026:* standard. NVIDIA's NIM docs give a selection order: use MTP when the base model ships MTP heads, else EAGLE3 when a draft exists — [NIM speculative decoding docs](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/speculative-decoding.html) (from search summary). Production numbers:
  - SGLang Qwen3.8 NVFP4 on B300 TP8: 346 tok/s at batch 1 with MTP (accept length 3.3) and 378 tok/s with DSpark (accept length 4) — [LMSYS (2026-08-12)](https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/) (via search summary)
  - "MTP drives 1.87X TPS/User" on GB300 long-context — [LMSYS (2026-02-19)](https://www.lmsys.org/blog/2026-02-19-gb300-longctx) (via search summary)
  - Training toolchains: [SpecForge (LMSYS, 2025-07-25)](https://www.lmsys.org/blog/2025-07-25-spec-forge/); [Speculators v0.3.0, end-to-end EAGLE-3 training (2025-12-13)](https://vllm.ai/blog/2025-12-13-speculators-v030); [Speculators v0.5.0 (2026-05-28)](https://vllm.ai/blog/2026-05-28-speculators-v050)
- *Explainers:*
  - [Jay Mody, Speculative Sampling (2023-02-08; B→I; with code)](https://jaykmody.com/blog/speculative-sampling/)
  - [Google Research, Looking back at speculative decoding (2024-12-06; Leviathan, Kalman, Matias; B)](https://research.google/blog/looking-back-at-speculative-decoding/)
  - [HF Assisted Generation (2023-05-11; B)](https://huggingface.co/blog/assisted-generation)
  - [vLLM, How Speculative Decoding Boosts vLLM Performance by up to 2.8x (2024-10-17; I)](https://blog.vllm.ai/2024/10/17/spec-decode.html)
  - [vLLM, Exploring Speculative Decoding on AMD GPUs: a practical guide covering MTP, EAGLE-3, DFlash, DSpark, configuration and tuning (2026-08-23; I)](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
  - [Modular handbook: speculative decoding (B)](https://handbook.modular.com/inference-optimization/speculative-decoding.md)
  - [Together ATLAS, runtime-learning speculators (2025-10-10)](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)
  - Surveys: [Xia et al., ACL Findings 2024 (arXiv 2401.07851)](https://arxiv.org/abs/2401.07851) with list [hemingkx/SpeculativeDecodingPapers](https://github.com/hemingkx/SpeculativeDecodingPapers) (~1.3k★, last push 2026-06-27); [Ryu & Kim survey (arXiv 2411.13157)](https://arxiv.org/abs/2411.13157)
- *Repos:* [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) ~2.6k★ (last push 2026-02-20); [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) ~2.8k★ (last push 2024-06, stale); [LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) ~1.3k★ (2025-03); [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) ~1.3k★ (active); [vllm-project/speculators](https://github.com/vllm-project/speculators) ~1.0k★ (active); [snowflakedb/ArcticInference](https://github.com/snowflakedb/ArcticInference) ~0.5k★ (2026-09)

#### G. Disaggregation: prefill/decode (P/D), encoder (EPD), attention–FFN (AFD)
- *What:* run prefill and decode on separate GPU pools so each phase gets its own parallelism and batching, and ship KV between them over NVLink/RDMA. This removes prefill-decode interference and maximizes goodput under TTFT/TPOT SLOs — [DistServe (OSDI 2024; arXiv 2401.09670)](https://arxiv.org/abs/2401.09670). Related systems:
  - [Splitwise (arXiv 2311.18677)](https://arxiv.org/abs/2311.18677)
  - [TetriInfer (arXiv 2401.11181)](https://arxiv.org/abs/2401.11181)
  - [Mooncake (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079)
  - Migration-based scheduling: [Llumnix (OSDI'24; arXiv 2406.03243)](https://arxiv.org/abs/2406.03243)
  - NVIDIA's "first systematic study of disaggregated inference at scale", which notes practical deployment "remains limited due to the complexity" of the design space — [Beyond the Buzz (arXiv 2506.05508, Jun 2025)](https://arxiv.org/abs/2506.05508)
- *Status 2026:* standard for large-scale serving. "Today, almost every production-grade LLM serving framework – NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake – runs on disaggregation". NVIDIA Rubin CPX "fully embraces p/d disaggregation" in hardware — [Hao AI Lab retrospective (2025-11-03)](https://hao-ai-lab.github.io/blogs/distserve-retro/). Recent work:
  - Reusing decode-node KV on later turns — [vLLM bi-directional KV transfer (2026-10-12)](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload)
  - Encoder disaggregation for multimodal — [vLLM EPD (2025-12-15)](https://vllm.ai/blog/2025-12-15-vllm-epd), [update (2026-11-11)](https://vllm.ai/blog/2026-11-11-encoder-disaggregation-epd)
- *Frontier, AFD:* the retrospective calls it "The Next Frontier". Examples:
  - [MegaScale-Infer (ByteDance; arXiv 2504.02263)](https://arxiv.org/abs/2504.02263)
  - [vLLM AFD Plugin (2026-07-23)](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin)
  - On GB200 NVL72, AFD is "up to 1.17x over wide EP on Qwen3 and 1.16x on Qwen3.5, and matches it on MiniMax-M2.5" — [FastAFD (arXiv 2611.07149, Nov 2026)](https://arxiv.org/abs/2611.07149)
  - [OpWeave operator-level disaggregation (arXiv 2609.14237)](https://arxiv.org/abs/2609.14237)
- *Explainers:* [Hao AI Lab DistServe blog (2024-03-17; I)](https://hao-ai-lab.github.io/blogs/distserve/); [Disaggregated Inference: 18 Months Later (2025-11-03; I)](https://hao-ai-lab.github.io/blogs/distserve-retro/); [vLLM, Taking vLLM Apart: A Practical Guide to Disaggregated Serving (2026-09-29; I, hands-on)](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide); [Modular handbook: P/D disaggregation (B→I)](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation.md); [NVIDIA Dynamo intro (2025-03-18)](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- *Repos:* [LLMServe/DistServe](https://github.com/LLMServe/DistServe) ~0.8k★ (last push 2025-04, archival); [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) ~8.6k★; [ai-dynamo/nixl](https://github.com/ai-dynamo/nixl) ~1.4k★ (KV transfer library); [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) ~7.1k★. Kubernetes and llm-d are out of scope here.

#### H. MoE serving: expert parallelism, DeepEP, EPLB, DeepSeek's inference system, wide EP
- *What:* shard experts across GPUs (EP) and use token all-to-all dispatch and combine. Attention usually runs data-parallel (DP-attention / "DEP"). Load is balanced with redundant experts and EPLB, and communication is hidden behind compute with micro-batch overlap. Supporting kernels: DeepEP for all-to-all, DeepGEMM for FP8 grouped GEMMs.
- *Canonical:* [DeepSpeed-MoE (ICML 2022; arXiv 2201.05596)](https://arxiv.org/abs/2201.05596); [MegaBlocks (arXiv 2211.15841)](https://arxiv.org/abs/2211.15841); [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437); [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware (arXiv 2505.09343; v2 2025-12-23)](https://arxiv.org/abs/2505.09343)
- *DeepSeek's production design, verified from the doc:*
  - Prefill: "Routed Expert EP32, MLA/Shared Expert DP32". Each unit spans 4 nodes with 32 redundant routed experts, and each GPU handles 9 routed experts plus 1 shared expert.
  - Decode: "EP144 ... DP144" over 18 nodes, with 2 routed plus 1 shared expert per GPU.
  - A "dual-batch overlap strategy" splits a batch into two microbatches; decode uses "a 5-stage pipeline".
  - Each H800 node averages "~73.7k tokens/s input (including cache hits) during prefilling or ~14.8k tokens/s output during decoding".
  - Source: [DeepSeek Open-Source Week (Feb 2025), Day 6: Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md). The doc's "545% cost profit margin" figure is from memory and was not re-checked.
- *Status 2026:* standard at frontier scale:
  - [SGLang PD + large-scale EP on 96 H100s (2025-05-05)](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
  - [SGLang GB200 NVL72 Part I, 2.7x decode (2025-06-16)](https://www.lmsys.org/blog/2025-06-16-gb200-part-1/)
  - [SGLang GB200 NVL72 Part II, 3.8x prefill and 4.8x decode (2025-09-25)](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/)
  - [vLLM, DeepSeek @ 2.2k tok/s/H200 with Wide-EP (2025-12-17)](https://vllm.ai/blog/2025-12-17-large-scale-serving)
  - [vLLM WideEP on Blackwell, 26.2K prefill and 10.1K decode tokens/GPU/s on GB200 for 2K/2K (2026-02-03)](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) (figures via search summary)
  - [Red Hat, scaling DeepSeek-style MoEs with vLLM and llm-d using wide EP (2025-09-08)](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep); seen in search, but 403 to automated fetch
  - DeepSeek-V4.1-Flash on GB300 NVL72 with P/D and Wide-EP: "954K tokens/s per GPU" on SemiAnalysis AgentX — [vLLM (2026-10-23)](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)
  - DeepSeek "open-sourced DeepInfer, the inference engine behind its own service, along with the kernels it runs on". SGLang benchmarked against it on DeepSeek-V4.1-Flash using "DEP4 for prefill and DEP64 for decode", Mooncake/NIXL/MORI KV transfer, EPLB and HiCache — [LMSYS, More Than DeepInfer (2026-10-24)](https://www.lmsys.org/blog/2026-10-24-dsv41-deepinfer)
  - Elastic EP is new: [Elastic Expert Parallelism (2026-05-14)](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism); [near-zero-downtime scaling (2026-10-19)](https://vllm.ai/blog/2026-10-19-elastic-ep-near-zero-downtime)
- *Explainers:* the DeepSeek Day-6 doc (I→A); the LMSYS large-scale EP post (I→A); [Maarten Grootendorst, A Visual Guide to MoE (2024-10-07; B)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts); [HF, Mixture of Experts Explained (2023-12-11; B)](https://huggingface.co/blog/moe); [Modular handbook: data/tensor/pipeline/expert/hybrid parallelism](https://handbook.modular.com/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism.md)
- *Repos:* [DeepEP](https://github.com/deepseek-ai/DeepEP) ~10.5k★ (active 2026-11-20); [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) ~9.2k★ (active); [EPLB](https://github.com/deepseek-ai/EPLB) ~1.5k★ (last push 2025-03); [profile-data](https://github.com/deepseek-ai/profile-data) ~1.2k★ (overlap traces); [perplexityai/pplx-kernels](https://github.com/perplexityai/pplx-kernels) ~0.6k★ (2025-11); [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) ~19.7k★ (heterogeneous CPU/GPU MoE inference; active)

#### I. Parallelism for inference (TP / PP / DP / EP / DP-attention / context parallelism)
- *What:*
  - TP splits each layer's matmuls (an all-reduce per layer, best within an NVLink domain) to cut per-token latency — [Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053). Inference-specific partitioning analysis: [Pope et al. (arXiv 2211.05102)](https://arxiv.org/abs/2211.05102)
  - PP splits layers. DP replicates the model. EP serves MoE (section H).
  - DP-attention avoids replicating MLA KV across TP ranks. It appears in [SGLang v0.4 (2024-12-04)](https://lmsys.org/blog/2024-12-04-sglang-v0-4/); the DP-attention detail is from memory, while the title lists the zero-overhead scheduler and cache-aware load balancer.
  - Context parallelism shards long sequences for prefill — [Ring Attention (arXiv 2310.01889)](https://arxiv.org/abs/2310.01889); [Context Parallelism for Scalable Million-Token Inference (Meta; arXiv 2411.01783)](https://arxiv.org/abs/2411.01783)
  - Helix gives attention and FFN different sharding for multi-million-token decode — [Helix Parallelism (NVIDIA; arXiv 2507.07120, Jul 2025)](https://arxiv.org/abs/2507.07120)
- *Status 2026:* TP/PP/DP/EP standard; DP-attention standard for MLA-class MoE models. Decode and prefill context parallelism are now shipping:
  - DCP "shards KV cache across GPUs by sequence dimension, enabling 3× higher throughput on long-context agentic workloads" — [vLLM DCP (2026-08-07)](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)
  - PCP and DCP serve GLM-5.3 at 1M-token contexts on 16 H200s, with "1.7–3.2× the per-user speed of data-parallel" — [vLLM (2026-10-23)](https://vllm.ai/blog/2026-10-23-glm53-part2-pcp-dcp-hisparse)
- *Explainers:* the scaling book (sharding and inference chapters; I→A); [HF Ultra-Scale Playbook (training-oriented but the clearest visuals of TP/PP/CP/EP; I)](https://huggingface.co/spaces/nanotron/ultrascale-playbook); [Modular handbook: parallelism](https://handbook.modular.com/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism.md)

#### J. Structured output (constrained decoding)
- *What:* at each step, mask the logits to tokens allowed by a regex, JSON Schema or grammar.
  - FSM indexing of the vocabulary: [Outlines, "Efficient Guided Generation for LLMs" (arXiv 2307.09702)](https://arxiv.org/abs/2307.09702)
  - Pushdown-automaton CFG engine: [XGrammar (MLSys '25; arXiv 2411.15100)](https://arxiv.org/abs/2411.15100)
  - Jump-forward decoding: [LMSYS compressed FSM (2024-02-05)](https://lmsys.org/blog/2024-02-05-compressed-fsm/)
  - Benchmark: [JSONSchemaBench (arXiv 2501.10868)](https://arxiv.org/abs/2501.10868)
- *Status 2026:* standard. Engines integrate XGrammar, llguidance and Outlines backends — [vLLM, Structured Decoding in vLLM: a gentle introduction (2025-01-14; B→I)](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- *Explainers:* the vLLM gentle intro; [MLC XGrammar blog (2024-11-22; I)](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar); the LMSYS compressed-FSM post (I)
- *Repos:* [outlines](https://github.com/dottxt-ai/outlines) ~16.0k★ (last push 2026-09-21); [xgrammar](https://github.com/mlc-ai/xgrammar) ~2.0k★ (active); [llguidance](https://github.com/guidance-ai/llguidance) ~0.9k★ (active)

#### K. Multi-LoRA serving
- *What:* serve many LoRA adapters over one base model in the same batch:
  - Segmented-gather matmul kernels for batching across adapters — [Punica (arXiv 2310.18547)](https://arxiv.org/abs/2310.18547)
  - Unified paging of adapter weights and KV, plus custom kernels — [S-LoRA (arXiv 2311.03285)](https://arxiv.org/abs/2311.03285); venues for both (MLSys 2024) from memory
- *Status 2026:* standard. It now extends to MoE models with "fused MoE LoRA kernels, Triton compiler fixes, Split-K and CTA swizzling" — [vLLM multi-LoRA (2026-02-26)](https://vllm.ai/blog/2026-02-26-multi-lora)
- *Explainers:* [LMSYS, Recipe for Serving Thousands of Concurrent LoRA Adapters (2023-11-15; I)](https://lmsys.org/blog/2023-11-15-slora/); [DeepLearning.AI, Efficiently Serving LLMs (2024; B; includes LoRA/multi-LoRA serving, from memory)](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)
- *Repos (archival):* [S-LoRA](https://github.com/S-LoRA/S-LoRA) ~1.9k★ (last push 2024-01); [punica](https://github.com/punica-ai/punica) ~1.2k★ (2024-05)

#### L. KV-aware routing and request scheduling
- *What:* route each request to the replica that already holds its prefix KV, balanced against queue depth and KV utilization. Schedule with policies beyond FCFS:
  - Preemptive skip-join MLFQ (from memory) — [FastServe (arXiv 2305.05920)](https://arxiv.org/abs/2305.05920)
  - Length-ranking — [Learning to Rank (arXiv 2408.15792)](https://arxiv.org/abs/2408.15792)
  - Fairness — [Fairness in Serving LLMs (arXiv 2401.00588)](https://arxiv.org/abs/2401.00588)
  - Distributed prompt scheduling — [Preble (arXiv 2407.00023)](https://arxiv.org/abs/2407.00023)
  - Live migration — [Llumnix (arXiv 2406.03243)](https://arxiv.org/abs/2406.03243)
- *Status 2026:* standard. Examples: the cache-aware load balancer in [SGLang v0.4 (2024-12-04)](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) and the [vLLM router release (2025-12-13)](https://vllm.ai/blog/2025-12-13-vllm-router-release). Agent- and session-aware routing is emerging — [vLLM Semantic Router: Session-Aware Agentic Routing (2026-06-02)](https://vllm.ai/blog/2026-06-02-session-aware-agentic-routing)
- *Explainer:* [Modular handbook: Inference routing ("cache locality, queue depth, KV cache ...")](https://handbook.modular.com/inference-optimization/inference-routing.md). Gateway and Kubernetes layers are out of scope.

#### M. CUDA graphs, torch.compile, megakernels, determinism
- *What:* CPU launch overhead dominates small-batch decode. The tools that address it:
  - CUDA graphs replay captured kernel sequences.
  - torch.compile fuses ops and generates kernels.
  - Megakernels or persistent kernels run the whole forward pass in one kernel.
  - Batch-invariant kernels give determinism, which matters for RL train/inference consistency.
- *Key sources:*
  - torch.compile, CUDA graphs, int8/int4 and speculative decoding in a tiny codebase — [PyTorch, Accelerating Generative AI with PyTorch II: GPT, Fast (2023-11-30)](https://pytorch.org/blog/accelerating-generative-ai-2/)
  - "TorchDynamo graph capture, TorchInductor code generation, custom compiler passes, graph breaks" — [vLLM, Introduction to torch.compile and How It Works with vLLM (2025-08-20)](https://vllm.ai/blog/2025-08-20-torch-compile)
  - [Hazy Research megakernel (2025-05-27)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); [Mirage Persistent Kernel: "Compiling LLMs into a MegaKernel"](https://github.com/mirage-project/mirage) (~2.6k★, active)
  - "GPU-native input preparation, stable persistent batching, async-first scheduling" — [vLLM Model Runner V2 (2026-03-24)](https://vllm.ai/blog/2026-03-24-mrv2)
  - Agent-assisted CuTe DSL optimizations for Kimi K3 decode on GB300 — [vLLM (2026-10-16)](https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel)
  - [FlyDSL MonoKernels on AMD Instinct (2026-11-18)](https://vllm.ai/blog/2026-11-18-flydsl-monokernels)
  - Determinism: [Thinking Machines (2025-09-10)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); [SGLang deterministic inference (2025-09-22)](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/); [vLLM bitwise-consistent on-policy RL (2025-11-10)](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference)
- *Status 2026:* CUDA graphs and torch.compile standard; deterministic/batch-invariant modes available; megakernels frontier, though now applied model-by-model in production engines.
- *Repos:* [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) ~6.3k★ (last push 2025-08-22); [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) ~3.8k★; [tile-ai/tilelang](https://github.com/tile-ai/tilelang) ~9.0k★

#### N. Benchmarking methodology and tools
- [vllm-project/guidellm](https://github.com/vllm-project/guidellm) ~1.8k★ (active) and [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf) ~0.9k★ (active) are maintained load generators.
- SemiAnalysis InferenceMAX now redirects to [inferencex.semianalysis.com](https://inferencex.semianalysis.com/) ("Compare AgentX long-context, multi-turn coding inference with fixed-sequence benchmarks across chips and frameworks"). Repo [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) has ~2.0k★ (active).
- Engines now tune against AgentX: KV management, parallelism, scheduling and P/D for agentic contexts — [vLLM x AgentX (2026-09-08)](https://vllm.ai/blog/2026-09-08-vllm-agentx)
- MLPerf Inference Datacenter is at v6.1 (page updated 2026-09-21) — [MLCommons](https://mlcommons.org/benchmarks/inference-datacenter/)
- Benchmarking explainer: [Modular handbook: LLM performance benchmarks](https://handbook.modular.com/inference-optimization/llm-performance-benchmarks.md)

### Inferences

**Concept map, as a dependency graph (learn top to bottom)**
```
L0 Foundations: transformer FLOPs -> GPU memory hierarchy & roofline -> prefill vs decode -> KV cache sizing -> metrics (TTFT, ITL/TPOT, E2E, throughput, goodput@SLO)
L1 Single-GPU engine core: paged KV -> continuous batching -> chunked prefill -> prefix caching (radix/APC)
   -> attention kernels (FlashAttention, split-KV FlashDecoding, FlashInfer, FlashMLA) -> CUDA graphs/torch.compile
   -> sampling & structured output -> quantization (W4A16 -> W8A8/FP8 -> FP8 KV -> NVFP4/MXFP4)
L2 Model-architecture awareness: MQA/GQA/MLA -> sliding-window, sparse (DSA/NSA/MoBA/CSA/HCA), hybrid linear attention
   -> speculative decoding (draft model -> n-gram/suffix -> Medusa -> EAGLE-1/2/3 -> native MTP -> parallel drafters + adaptive verification)
L3 Multi-GPU: TP/PP/DP -> MoE: EP + DP-attention, DeepEP all-to-all, EPLB, micro-batch overlap -> context parallelism (prefill CP, decode CP, Helix)
L4 Cluster serving: P/D disaggregation + KV transfer (NIXL/Mooncake) -> KV tiering/offload (LMCache, HiCache, Mooncake Store, NVMe)
   -> KV-aware routing & SLO-aware scheduling -> multi-LoRA -> elastic EP -> encoder disaggregation (EPD)
L5 Frontier: attention-FFN disaggregation, FP4 attention / 4-bit KV, megakernels, agent-directed KV, determinism for RL, diffusion LLMs
```

**Status classification (my synthesis from the evidence above)**

| Technique | 2026 status | Basis |
|---|---|---|
| Continuous batching, paged KV, prefix caching, chunked prefill | Standard everywhere | Hao AI Lab retro; vLLM/SGLang docs and posts |
| FlashAttention-class kernels, FlashInfer, FlashMLA, split-KV decode | Standard (FA4/cuDNN-class on Blackwell) | FA4 paper; vLLM Blackwell/Rubin posts |
| GQA (and MLA for DeepSeek-family) | Standard | Model configs |
| FP8 W8A8 and FP8 KV | Standard on Hopper/Blackwell | vLLM FP8 KV post |
| NVFP4/MXFP4 (esp. MoE experts) | Standard at frontier scale on Blackwell | vLLM/SGLang GB200/GB300 posts |
| INT4 weight-only (AWQ/GPTQ + Marlin-class kernels) | Standard for cost-sensitive/smaller deployments | inference; tools active |
| NVFP4 KV, 2–4-bit KV, FP4 attention math | Frontier / early adoption | NVIDIA caveat; TurboQuant study; arXiv 2609.04105 |
| Speculative decoding via MTP / EAGLE-3 | Standard | NIM docs; vLLM/SGLang posts |
| Parallel drafting (P-EAGLE/DFlash/DSpark), adaptive verification | Rapidly productizing (2026) | vLLM 2026 posts |
| P/D disaggregation + KV transfer | Standard at scale | Hao AI Lab retro; Dynamo/vLLM/SGLang |
| Wide EP + DP-attention + EPLB | Standard for DeepSeek-class MoE at scale | DeepSeek Day 6; vLLM/SGLang posts |
| Elastic EP, decode/prefill context parallelism | New in 2026, shipping | vLLM 2026 posts |
| Attention–FFN disaggregation | Frontier (gains 0–17% in FastAFD) | FastAFD; vLLM AFD plugin |
| KV offload to CPU/NVMe/distributed stores | Standard for agentic/multi-turn | vLLM/SGLang/LMCache posts |
| KV-aware routing | Standard | SGLang router, vLLM router |
| Training-free KV eviction (H2O, SnapKV, Quest) | Mostly research | few production references found |
| Model-native sparse/compressed/hybrid attention | Production models with engine day-0 support | DeepSeek V3.2/V4/V4.1, Qwen3.8 |
| Megakernels | Frontier → model-specific production | Hazy, Mirage, vLLM Kimi K3/FlyDSL |
| Structured output (XGrammar/llguidance/Outlines) | Standard | vLLM intro post |
| Multi-LoRA | Standard (incl. MoE LoRA) | vLLM 2026 post |
| CUDA graphs + torch.compile | Standard | vLLM torch.compile post |

- Medusa has effectively been superseded in practice by EAGLE-3 and native MTP. The 2026 production guidance and posts discuss MTP, EAGLE-3, DFlash and DSpark, not Medusa, and the Medusa repo has been inactive since 2024.
- Paper repos (DistServe, S-LoRA, Punica, SmoothQuant, GPTQ, Marlin) are archival learning artifacts. Learners should read the paper and then the engine implementation.

### Gaps
- Several per-paper venue attributions are from memory: Sarathi-Serve (OSDI 2024), Splitwise (ISCA 2024), Mooncake (FAST 2025), S-LoRA/Punica (MLSys 2024), Preble (ICLR 2025), VTC fairness (OSDI 2024), FA1/FA3 (NeurIPS), NSA (ACL 2025). arXiv records were verified but their comment fields did not always state the venue.
- For many 2026 vLLM posts I read only titles and meta descriptions, not the full text. Numbers labeled "via search summary" (vLLM GB200 TPGS, SGLang Qwen3.8 and GB300 figures, NIM spec-decode ordering) should be re-read at the source.
- The "954K tokens/s per GPU" DeepSeek-V4.1-Flash figure almost certainly counts input and cached tokens on an input-heavy agentic benchmark; I did not confirm the accounting.
- DeepInfer's release date and license were not stated in the LMSYS post.
- Medium (FlashAttention ELI5), kexue.fm, Red Hat Developers and the Snowflake engineering blog returned 403 to automated fetch, so their liveness is unverified.

---

## Q3. Which techniques are standard in production in 2026 vs. research frontier, and what were the key 2025–2026 developments?

### Takeaway
2025 was the year cluster-scale MoE serving went mainstream. DeepSeek's open infra release (Feb 2025) revealed EP32/EP144 + DP-attention with dual-batch overlap. NVIDIA Dynamo (Mar 2025) arrived, SGLang and vLLM followed with wide-EP + P/D on H100/H200 and then GB200, and by Nov 2025 the DistServe authors could say that almost every production framework "runs on disaggregation". 2026 has been about Blackwell/GB300 and FP4 (NVFP4 MoE experts, FP8 KV), FlashAttention-4, and DeepSeek-V4 and V4.1's compressed sparse attention (V4.1-Flash: 890 bytes/token of KV). Other 2026 threads: parallel-drafting speculative decoding, elastic EP, decode context parallelism, multi-tier KV for agentic workloads (including NVMe), Vera Rubin bring-up, and DeepSeek open-sourcing its own engine, DeepInfer. AFD, FP4 attention, megakernels and adaptive verification remain frontier.

### Cited Findings

**2025 milestones**
- Jan 27: vLLM V1 core-architecture rewrite — [vLLM V1 (2025-01-27)](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- Feb 2025 DeepSeek Open-Source Week: FlashMLA, DeepEP, DeepGEMM, EPLB, 3FS, and the Day-6 inference system overview — [open-infra-index](https://github.com/deepseek-ai/open-infra-index); [Day 6 doc](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- Mar 3: EAGLE-3 — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840). Mar 18: NVIDIA Dynamo announced at GTC — [NVIDIA](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- May 5: SGLang PD + large-scale EP — [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/). May 27: Hazy megakernel — [Hazy](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Jun 5: NVIDIA "Beyond the Buzz" disaggregation study — [arXiv 2506.05508](https://arxiv.org/abs/2506.05508). Jun 16: SGLang GB200 Part I — [LMSYS](https://www.lmsys.org/blog/2025-06-16-gb200-part-1/). Jun 24: NVFP4 introduced — [NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- Jul 7: Helix parallelism — [arXiv 2507.07120](https://arxiv.org/abs/2507.07120). Jul 25: SpecForge — [LMSYS](https://www.lmsys.org/blog/2025-07-25-spec-forge/)
- Aug 20: vLLM torch.compile explainer — [vLLM](https://vllm.ai/blog/2025-08-20-torch-compile). Aug 29: "Inside vLLM" — [Gordić](https://www.aleksagordic.com/blog/vllm)
- Sep 10: Thinking Machines nondeterminism post — [link](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); SGLang HiCache — [LMSYS](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/). Sep 22: SGLang deterministic inference — [LMSYS](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/). Sep 25: SGLang GB200 Part II (FP8 attention, NVFP4 MoE) — [LMSYS](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/). Sep 26: Modal FA4 reverse-engineering — [Modal](https://modal.com/blog/reverse-engineer-flash-attention-4). Sep 29: DeepSeek-V3.2-Exp sparse attention, day-0 in vLLM and SGLang — [vLLM](https://vllm.ai/blog/2025-09-29-deepseek-v3-2), [LMSYS](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/)
- Oct 8: LMCache paper — [arXiv 2510.09665](https://arxiv.org/abs/2510.09665). Oct 9: SemiAnalysis InferenceMAX with vLLM/NVIDIA Blackwell FP4 kernels — [vLLM](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax). Oct 10: Together ATLAS — [Together](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas). Oct 30: Kimi Linear — [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)
- Nov 3: "Disaggregated Inference: 18 Months Later" — [Hao AI Lab](https://hao-ai-lab.github.io/blogs/distserve-retro/). Nov 10: vLLM bitwise-consistent RL — [vLLM](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference). Nov 25: HF continuous batching — [HF](https://huggingface.co/blog/continuous_batching)
- Dec 2: DeepSeek-V3.2 paper — [arXiv 2512.02556](https://arxiv.org/abs/2512.02556). Dec 8: NVFP4 KV cache — [NVIDIA](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/). Dec 13: Speculators v0.3.0 — [vLLM](https://vllm.ai/blog/2025-12-13-speculators-v030). Dec 15: vLLM EPD — [vLLM](https://vllm.ai/blog/2025-12-15-vllm-epd). Dec 17: vLLM wide-EP at 2.2k tok/s/H200 — [vLLM](https://vllm.ai/blog/2025-12-17-large-scale-serving). Dec 31: "Speculative Decoding: Performance or Illusion?" — [arXiv 2601.11580](https://arxiv.org/abs/2601.11580)

**2026 milestones (Jan–Nov)**
- Jan 8: vLLM KV offloading connector — [vLLM](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- Feb 3: vLLM WideEP on GB200 — [vLLM](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1). Feb 10: Modular acquires BentoML, and the handbook moves to handbook.modular.com — [Modular](https://modular.com/blog/bentoml-joins-modular). Feb 13: DeepSeek-V3.2 on GB300 with NVFP4 — [vLLM](https://vllm.ai/blog/2026-02-13-gb300-deepseek). Feb 19: SGLang GB300 long-context (DeepSeek R1-NVFP4, 128K/8K) — [LMSYS](https://www.lmsys.org/blog/2026-02-19-gb300-longctx). Feb 26: MoE multi-LoRA — [vLLM](https://vllm.ai/blog/2026-02-26-multi-lora)
- Mar 5: FlashAttention-4 paper (MLSys 2026) — [arXiv 2603.05451](https://arxiv.org/abs/2603.05451). Mar 13: P-EAGLE — [vLLM](https://vllm.ai/blog/2026-03-13-p-eagle). Mar 24: Model Runner V2 — [vLLM](https://vllm.ai/blog/2026-03-24-mrv2)
- Apr 21: hybrid-SSM P/D disaggregation — [vLLM](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg). Apr 22: FP8 KV/attention quantization state — [vLLM](https://vllm.ai/blog/2026-04-22-fp8-kvcache). Apr 24–26: DeepSeek-V4 preview (CSA+HCA, mHC, Muon, 1M context) with vLLM day-0 and an NVIDIA Blackwell post — [arXiv 2606.19348](https://arxiv.org/abs/2606.19348), [vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4), [NVIDIA](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)
- May 6: vLLM x Mooncake Store — [vLLM](https://vllm.ai/blog/2026-05-06-mooncake-store). May 11: TurboQuant KV study — [vLLM](https://vllm.ai/blog/2026-05-11-turboquant). May 14: Elastic EP — [vLLM](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism). May 26: EAGLE 3.1 — [vLLM](https://vllm.ai/blog/2026-05-26-eagle-3-1). May 28: DFlash speculator and Speculators v0.5.0 — [vLLM](https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor). May 31: Leyline — [arXiv 2606.01065](https://arxiv.org/abs/2606.01065)
- Jun 3: DeepLearning.AI course "Fast & Efficient LLM Inference with vLLM" — [DeepLearning.AI](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm/), [vLLM announcement](https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course)
- Jul 23: vLLM AFD plugin — [vLLM](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin). Jul 28: parallel drafting (P-EAGLE/DFlash/DSpark) — [vLLM](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting); SGLang quantization-stack cleanup — [LMSYS](https://www.lmsys.org/blog/2026-07-31-cleaner-quantization-stack/)
- Aug 7: decode context parallelism — [vLLM](https://vllm.ai/blog/2026-08-07-decode-context-parallelism). Aug 12: Qwen3.8-2.4T-A95B hybrid attention, day-0 in SGLang — [LMSYS](https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/). Aug 14: DSpark adaptive verification — [vLLM](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification). Aug 23: speculative decoding on AMD guide — [vLLM](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- Sep 3: FP4 FlashAttention-4 — [arXiv 2609.04105](https://arxiv.org/abs/2609.04105). Sep 8: vLLM x AgentX — [vLLM](https://vllm.ai/blog/2026-09-08-vllm-agentx). Sep 10: tiered KV offloading — [vLLM](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading). Sep 17: DeepSeek-V4.1-Flash (890 B/token KV) — [arXiv 2609.19969](https://arxiv.org/abs/2609.19969). Sep 29: vLLM practical disaggregation guide — [vLLM](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide)
- Oct 9: vLLM on Vera Rubin NVL72, claiming "7.8x the per-GPU throughput" over GB200 NVL72 (vendor/engine claim) — [vLLM](https://vllm.ai/blog/2026-10-09-vera-rubin-preview). Oct 12: bi-directional KV transfer — [vLLM](https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload). Oct 16: agent-assisted CuTe DSL kernels for Kimi K3 — [vLLM](https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel). Oct 19: near-zero-downtime elastic EP — [vLLM](https://vllm.ai/blog/2026-10-19-elastic-ep-near-zero-downtime). Oct 21: Humming quantized kernels — [vLLM](https://vllm.ai/blog/2026-10-21-humming). Oct 23: DeepSeek-V4.1-Flash P/D + Wide-EP on GB300 NVL72 — [vLLM](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep); GLM-5.3 PCP/DCP — [vLLM](https://vllm.ai/blog/2026-10-23-glm53-part2-pcp-dcp-hisparse). Oct 24: DeepSeek's DeepInfer engine open-sourced (per SGLang's comparison post) — [LMSYS](https://www.lmsys.org/blog/2026-10-24-dsv41-deepinfer)
- Nov 4: FastAFD — [arXiv 2611.07149](https://arxiv.org/abs/2611.07149). Nov 10: ForeSpark — [arXiv 2611.13442](https://arxiv.org/abs/2611.13442). Nov 11: EPD update — [vLLM](https://vllm.ai/blog/2026-11-11-encoder-disaggregation-epd). Nov 18: FlyDSL MonoKernels — [vLLM](https://vllm.ai/blog/2026-11-18-flydsl-monokernels). Nov 23: NVMe KV tier — [vLLM](https://vllm.ai/blog/2026-11-23-skip-the-dram-tier)

**Where the developments are best explained**
- Disaggregation history and state: [Hao AI Lab retrospective](https://hao-ai-lab.github.io/blogs/distserve-retro/) and the [vLLM practical guide (2026-09-29)](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide)
- Wide EP: [DeepSeek Day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), then the [LMSYS large-scale EP post](https://lmsys.org/blog/2025-05-05-large-scale-ep/), then the [vLLM wide-EP series](https://vllm.ai/blog/2025-12-17-large-scale-serving)
- FP4: [NVIDIA NVFP4 intro](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and the [SGLang GB200 Part II](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/)
- DeepSeek V4 attention: [vLLM first-principles walkthrough](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- Speculative decoding 2026: [vLLM "Parallel All the Way Down"](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting) and the [AMD guide](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)

### Inferences
- *Standard in production (2026):* the full engine core (Q2 A–C, M), GQA/MLA, FP8 everywhere, FP4 MoE on Blackwell, chunked prefill, prefix caching plus CPU/NVMe/distributed KV tiers, MTP/EAGLE-3 speculative decoding, structured output, multi-LoRA, TP/EP/DP-attention, P/D disaggregation at scale, and KV-aware routing.
- *Frontier (2026):* AFD (FastAFD shows parity to 1.17x, so it is not a clear win); FP4 attention math and 4-bit KV; adaptive and parallel-drafting speculative decoding (productizing quickly); megakernels; Helix-style sharding for multi-million-token decode; agent-directed KV management (Leyline); energy-aware speculation; diffusion LLMs ([Mercury (arXiv 2506.17298)](https://arxiv.org/abs/2506.17298), [LLaDA (arXiv 2502.09992)](https://arxiv.org/abs/2502.09992)).
- The 2026 workload center of gravity has moved to agentic, long-context, input-heavy traffic. Evidence: AgentX benchmarks; DeepSeek-V4.1-Flash's stated motivation that "workloads [are] increasingly input-heavy"; and vLLM's session routing, bi-directional KV and NVMe tiers. This raises the value of prefix/KV reuse, KV compression and prefill efficiency relative to pure decode-throughput tricks.
- Model–system co-design is now explicit: KV footprint per token fell from about 320 KiB (Llama-3.1-70B GQA, BF16) to about 68.6 KiB (DeepSeek-V3 MLA, BF16) to 890 bytes (DeepSeek-V4.1-Flash). Learners should study model architecture papers as serving papers.

### Gaps
- No independent verification of vendor/engine performance claims (Vera Rubin 7.8x; SGLang-vs-DeepInfer deltas; 954K tok/s/GPU).
- I did not find a neutral, current cross-engine benchmark of speculative-decoding methods. The search found a garbled "SpecDecode-Bench" result that I could not use.
- These arXiv items appeared in search results but were not verified via the arXiv API: Windowed-MTP (2607.21535), "Adversarial Prompts for Acceptance Collapse" (2607.21804), Tsubame (2609.33652), d3LLM (2601.07568).

---

## Q4. Comprehensive handbooks, surveys and curated reading lists — and are they maintained in 2026?

### Takeaway
For structured learning, the best maintained free long-form resources in late 2026 are: the Modular (ex-BentoML) LLM Inference Handbook (active), Baseten's free "Inference Engineering" book (2026), "How to Scale Your Model" (repo active Oct 2026), Stas Bekman's ML Engineering open book (active) and Modal's GPU Glossary. For literature, xlite-dev/Awesome-LLM-Inference (~5.6k★, pushed 2026-11-24) and AmberLJC/LLMSys-PaperList (~2.3k★, pushed 2026-10-20) are the maintained hubs, with topic lists for speculative decoding, KV cache, compression and long context. The 2024 surveys (Zhou et al.; Yuan et al. "LLM Inference Unveiled") remain good taxonomies but predate wide EP, FP4 and disaggregation-as-default. The newest broad surveys are Park et al. on inference engines (v3 Nov 2025) and Pan & Li (Jun 2025).

### Cited Findings

**Handbooks, books and long-form guides**

| Resource | Maintainer | Date / activity | Format · Level · Lang | Key topics | Notes |
|---|---|---|---|---|---|
| [LLM Inference Handbook](https://handbook.modular.com/) | Modular (from BentoML; [repo](https://github.com/modular/llm-inference-handbook) ~0.5k★, pushed 2026-11-24) | maintained | web · B→I · EN | foundations, GPU memory, optimization, routing, kernels | BentoML joined Modular on 2026-02-10 ([Modular](https://modular.com/blog/bentoml-joins-modular)); the old URL is dead |
| [Inference Engineering](https://www.baseten.co/inference-engineering/) | Philip Kiely, Baseten | 2026 | free PDF, online, audiobook · B→I · EN | models, hardware, software, techniques, modalities, production | 256 pages; free download |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Google DeepMind; [repo](https://github.com/jax-ml/scaling-book) ~1.5k★, pushed 2026-10-19 | maintained | online book · I→A · EN | rooflines, sharding, inference chapters + problems | TPU-centric |
| [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (has an inference chapter, [inference/](https://github.com/stas00/ml-engineering/tree/master/inference)) | Stas Bekman; ~19.7k★, pushed 2026-11-24 | maintained | GitHub book · I→A · EN | hardware, networking, training, inference | Chapter page could not be fetched from this environment (GitHub HTML blocked); repo active |
| [GPU Glossary](https://modal.com/gpu-glossary) | Modal; [repo](https://github.com/modal-labs/gpu-glossary) ~0.8k★, pushed 2026-09-04 | maintained | web · B · EN | GPU hardware/software | Prerequisite literacy |
| [LLM Engineer's Almanac](https://modal.com/llm-almanac) | Modal | now redirects to "LLM Engine Advisor" | interactive · I · EN | engine/hardware benchmark advisor | Changed format; verify contents |
| [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | Hugging Face (nanotron) | 2025 | interactive book · I · EN | TP/PP/CP/EP (training-oriented) | Best parallelism visuals |
| [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | NVIDIA | 2023-11-17 | blog · B→I · EN | overview | Dated |
| [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | Lilian Weng | 2023-01-10 | blog · I · EN | compression-centric | Dated |
| [Machine Learning Systems textbook (cs249r)](https://github.com/harvard-edge/cs249r_book) | Harvard (V. J. Reddi); ~30.3k★, pushed 2026-11-23 | maintained | open textbook · B→I · EN | broad ML systems | Broader than LLM inference |

**Surveys (papers)**

| Survey | Authors | Dates (arXiv) | Level | Notes / companion list |
|---|---|---|---|---|
| [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294) | Zhou et al. | 2024-04-22; v3 2024-07-19 | I | Good taxonomy. Pre-2025 (no wide EP/FP4/DSA) |
| [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363) | Yuan et al. | 2024-02-26; v6 2024-05-01 | I | Roofline framing; tool [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) (~0.7k★) last pushed 2024-09-11, so stale |
| [Towards Efficient Generative LLM Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234) | Miao et al. | 2023-12-23; v2 2025-07-23 (ACM Computing Surveys) | I | Updated 2025 |
| [LLM Inference Serving: Survey of Recent Advances and Opportunities](https://arxiv.org/abs/2407.12391) | Li et al. | 2024-07-17 | I | Short systems survey |
| [A Survey of LLM Inference Systems](https://arxiv.org/abs/2506.21901) | Pan & Li | 2025-06-27 | I→A | Recent systems survey (25 pages) |
| [Taming the Titans: A Survey of Efficient LLM Inference Serving](https://arxiv.org/abs/2504.19720) | Zhen et al. | 2025-04-28 | I | Marked "work in progress" |
| [A Survey on Inference Engines for LLMs: Perspectives on Optimization and Efficiency](https://arxiv.org/abs/2505.01658) | Park et al. | 2025-05-03; v3 2025-11-26 | I→A | 106 pages; list [sihyeong/Awesome-LLM-Inference-Engine](https://github.com/sihyeong/Awesome-LLM-Inference-Engine) (~0.3k★, pushed 2026-11-24) |
| [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863) | Wan et al. (TMLR) | 2023-12-06; v4 2024-05-23 | I | List [AIoT-MLSys-Lab/Efficient-LLMs-Survey](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey) (~1.3k★, pushed 2026-10-30) |
| [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/abs/2412.19442) | Li et al. (TMLR 2025) | 2024-12-27; v3 2025-07-30 | I→A | List [TreeAI-Lab/Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) (~0.4k★, pushed 2026-07-16) |
| [Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851) | Xia et al. (ACL Findings 2024) | 2024-01-15; v3 2024-06-04 | I | List [hemingkx/SpeculativeDecodingPapers](https://github.com/hemingkx/SpeculativeDecodingPapers) (~1.3k★, pushed 2026-06-27) |
| [From Attention to Disaggregation: Tracing the Evolution of LLM Inference](https://arxiv.org/abs/2511.07422) | Kumar et al. | 2025-10-16 | B→I | Lesser-known authors; use as a narrative, not authoritative |
| MoE survey (TKDE'25) | Mu et al. (from memory) | n/a | I | List [withinmiaov/A-Survey-on-Mixture-of-Experts-in-LLMs](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts-in-LLMs) (~0.5k★, pushed 2026-08-18) |

**Curated GitHub reading lists (stars and last push as of 2026-11-24)**

| List | ~Stars | Last push | Status | Scope |
|---|---|---|---|---|
| [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | 5.6k | 2026-11-24 | Actively maintained | "LLM/VLM Inference Papers with Codes": Flash-Attention, paged/KV, parallelism, quantization, etc. |
| [AmberLJC/LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) | 2.3k | 2026-10-20 | Maintained | LLM systems papers (training + serving), plus talks/blogs |
| [HuangOwen/Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) | 1.9k | 2026-11-16 | Maintained | quantization/pruning/distillation papers and tools |
| [horseee/Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) | 2.1k | 2025-06-17 | Stale (no 2025H2–2026 updates) | efficient LLM papers |
| [hemingkx/SpeculativeDecodingPapers](https://github.com/hemingkx/SpeculativeDecodingPapers) | 1.3k | 2026-06-27 | Maintained (slowing) | speculative decoding |
| [Zefan-Cai/Awesome-LLM-KV-Cache](https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache) | 0.5k | 2026-06-17 | Maintained (slowing) | KV cache papers with code |
| [TreeAI-Lab/Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) | 0.4k | 2026-07-16 | Maintained | KV cache management (survey companion) |
| [Xnhyacinth/Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) | 2.2k | 2026-11-13 | Maintained | long-context papers/blogs |
| [codecaution/Awesome-Mixture-of-Experts-Papers](https://github.com/codecaution/Awesome-Mixture-of-Experts-Papers) | 0.7k | 2024-10-30 | Stale | MoE papers |
| [sihyeong/Awesome-LLM-Inference-Engine](https://github.com/sihyeong/Awesome-LLM-Inference-Engine) | 0.3k | 2026-11-24 | Maintained | engines (survey companion) |
| [ModelTC/awesome-lm-system](https://github.com/ModelTC/awesome-lm-system) | <0.1k | 2023-12-17 | Dead | skip |
| [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) | 2.4k | 2026-06-15 | Maintained | GPU programming news/material |
| Chinese-language: [Infrasys-AI/AISystem](https://github.com/Infrasys-AI/AISystem) (formerly chenzomi12/AISystem) | 18.2k | 2025-09-03 | Slowing | AI systems full stack (中文) |
| Chinese-language: [Infrasys-AI/AIInfra](https://github.com/Infrasys-AI/AIInfra) | 8.7k | 2025-12-22 | Slowing | AI infra incl. inference (中文) |
| Chinese-language: [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action) | 25.3k | 2026-10-14 | Maintained | LLM engineering incl. inference (中文) |
| Chinese-language: [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | 3.4k | 2026-11-02 | Maintained | CUDA optimization notes, many LLM-inference write-ups (中文) |

**Lecture series and courses with strong inference content** (another researcher covers courses in depth)
- [CMU 11-868 LLM Systems, Spring 2026 site](https://llmsystem.github.io/llmsystem2026spring/) (live; the 2025 site is also live); [UCSD CSE 234 (Hao Zhang), Winter 2025](https://hao-ai-lab.github.io/cse234-w25/); [CMU 15-442/642 ML Systems](https://mlsyscourse.org/); [Stanford CS336 (2025)](https://stanford-cs336.github.io/spring2025/) (has an inference lecture, from memory; a spring2026 URL returned 404); [MIT 6.5940 Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940)
- [GPU MODE (YouTube)](https://www.youtube.com/@GPUMODE), with [gpu-mode/lectures](https://github.com/gpu-mode/lectures) at ~7.1k★ (pushed 2026-09-09)
- [DeepLearning.AI, Efficiently Serving LLMs (2024)](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/); [DeepLearning.AI, Fast & Efficient LLM Inference with vLLM (2026-06-03)](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm/). The vLLM course covers LLM Compressor quantization, GuideLLM benchmarking and KV cache sizing — [vLLM announcement](https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course)

**Primary-source blogs to follow** (the de-facto "living handbook")
- [vLLM blog](https://vllm.ai/blog) (very high cadence in 2026); [LMSYS/SGLang blog](https://www.lmsys.org/blog/); [Hao AI Lab](https://hao-ai-lab.github.io/blogs/distserve-retro/); [FlashInfer blog](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html); [LMCache blog](https://blog.lmcache.ai/); [NVIDIA Technical Blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/); [DeepSeek open-infra-index](https://github.com/deepseek-ai/open-infra-index)

### Inferences
- For a self-learner targeting vLLM/SGLang roles, the most efficient path is to read the Modular handbook and the Baseten book for breadth. Then go deep via scaling-book problems, and track the vLLM and LMSYS blogs, which in 2026 have become the most current "textbook" for wide EP, P/D, FP4 and speculative decoding.
- Use xlite-dev/Awesome-LLM-Inference as the master index and the topic lists (speculative decoding, KV cache, compression, long-context) for depth. Avoid horseee/Awesome-Efficient-LLM and codecaution MoE as primary sources since they are stale.
- Surveys from 2024 should be read for taxonomy only. Their "systems" sections miss 2025–26 practice: disaggregation as default, wide EP, FP4, model-native sparse attention.

### Gaps
- I did not verify paywalled books such as Chip Huyen's *AI Engineering* (O'Reilly, 2025; chapter on inference optimization) or O'Reilly's *AI Systems Performance Engineering* (2025). Both are paywalled, and I did not check them this session.
- Survey quality and maintenance were judged from metadata and repo activity, not a full read. For list repos, "last push" can include trivial commits.

---

## Q5. Hands-on exercises that build intuition

### Takeaway
The highest-value exercises are: (1) napkin math on real model configs (KV bytes/token, concurrency per GPU, speed-of-light decode latency, roofline batch thresholds); (2) benchmarking TTFT/ITL/throughput/goodput under load while toggling one engine feature at a time; (3) building a mini engine (KV cache, then paged allocator, then continuous batching, then radix prefix cache) with nano-vllm, mini-sglang or tiny-llm as references; (4) writing online-softmax/FlashAttention and a speculative sampler from scratch; (5) running quantization, speculative decoding and P/D disaggregation in vLLM/SGLang and comparing measured results with theory.

### Cited Findings

**Verified model configs for KV-cache exercises** (BF16 KV; my arithmetic from the configs)
- Llama-3.1-8B: 32 layers, 32 heads, 8 KV heads, head_dim 128, so 2 x 32 x 8 x 128 x 2 B = 131,072 B = **128 KiB/token**, or **16 GiB per 128K-token sequence** — [config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/raw/main/config.json)
- Llama-3.1-70B: 80 layers, 64 heads, 8 KV heads, head_dim 128, so **320 KiB/token**, or **40 GiB per 128K sequence** — [config](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/raw/main/config.json)
- Qwen3-32B: 64 layers, 8 KV heads, head_dim 128, so **256 KiB/token**; native max_position_embeddings is 40,960 — [config](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)
- gpt-oss-120b: 36 layers, 8 KV heads, head_dim 64, so 72 KiB/token if every layer were full attention. Layers alternate `sliding_attention` (window 128) and `full_attention`, and it has 128 experts with 4 active per token — [config](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json)
- DeepSeek-V3 (MLA): 61 layers, kv_lora_rank 512 plus rope dim 64, so (512+64) x 61 x 2 B = **70,272 B ≈ 68.6 KiB/token**. It has 256 routed experts with 8 active per token — [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)
- DeepSeek-V4.1-Flash: **890 bytes/token** global KV footprint (CSA2 cross-layer reuse plus FP4 KV) — [arXiv 2609.19969](https://arxiv.org/abs/2609.19969)
- Roofline anchors: A100 ratio 208 ([kipply](https://kipp.ly/transformer-inference-arithmetic/)); TPU v5e 240 FLOPs/byte ([scaling book](https://jax-ml.github.io/scaling-book/inference/)); H100 SXM 3,350 GB/s ([zeux](https://zeux.io/2024/03/15/llm-inference-sol/)). A worked lower-bound problem: a 30B dense model on a TPU v5e 4x4 slice in int8, 8,192 context, 100 kB/token KV, batch 4 vs 256 — [scaling book](https://jax-ml.github.io/scaling-book/inference/)

**Hands-on codebases, courses and tools**

| Resource | Maintainer | ~Stars / last push | Format · Level · Lang | Prereqs | Use it for |
|---|---|---|---|---|---|
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | GeeeekExplorer | 16.5k / 2026-04-26 | Python repo · I · EN | PyTorch | Read a compact vLLM-style engine: paged KV, scheduler, prefix cache, CUDA graphs (features from memory of README) |
| [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) | SGLang team | 5.6k / 2026-05-17 | Python repo · I · EN | PyTorch | "A compact implementation of SGLang, designed to demystify" serving (radix cache, scheduler) |
| [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) | Alex Chi Z. (skyzh) | 5.8k / 2026-11-23 | course + repo (MLX, Apple Silicon) · B→I · EN | Python, systems | "learn LLM inference system on Apple Silicon for systems engineers": build a tiny serving system step by step |
| [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) + [blog](https://pytorch.org/blog/accelerating-generative-ai-2/) | PyTorch team | 6.3k / 2025-08-22 | repo + blog · I · EN | PyTorch | torch.compile, CUDA graphs, int8/int4, speculative decoding in "<1000 LOC" |
| [andrewkchan/yalm](https://github.com/andrewkchan/yalm) + [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) | Andrew Chan | 0.6k / 2025-09-13 | C++/CUDA repo + blog · I→A · EN | C++, CUDA | Build an engine "no libraries except for I/O". Includes bandwidth-bound tok/s math, e.g. Mistral-7B FP16 bytes/token vs CPU bandwidth (blog date not verified) |
| [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) | Peter Kim | 1.2k / 2024-12-30 | CUDA · I→A · EN | CUDA basics | FlashAttention forward "in ~100 lines of CUDA" |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) + [Coding the KV Cache](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) | Sebastian Raschka | 109k / 2026-11-24 (article 2025-06-17) | repo + article · B · EN | PyTorch | Implement a KV cache from scratch |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | Andrej Karpathy | 60.8k / 2026-09-07 | repo · B→I · EN | PyTorch | Full small-LLM stack incl. a simple inference engine with KV cache (engine details from memory) |
| [vllm-project/guidellm](https://github.com/vllm-project/guidellm) | vLLM project (Red Hat) | 1.8k / 2026-11-23 | CLI · B→I · EN | a running server | Load tests: TTFT/ITL/throughput sweeps |
| [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf) | NVIDIA Dynamo | 0.9k / 2026-11-24 | CLI · B→I · EN | a running server | Benchmarking generative AI serving |
| [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) | SemiAnalysis | 2.0k / 2026-11-17 | benchmark harness · I→A · EN | multi-GPU | Reproduce/compare agentic (AgentX) and fixed-sequence results |
| [hahnyuan/LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) | Yuan et al. | 0.7k / 2024-09-11 (stale) | tool · B→I · EN | none | Roofline analysis per layer (older models) |
| [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) | xlite-dev | 12.2k / 2026-11-24 | CUDA notes/exercises · B→A · EN/中文 | C++ | CUDA kernels incl. BF16/FP8/FP4 practice |
| [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | GPU MODE | 7.1k / 2026-09-09 | lecture notebooks · I→A · EN | CUDA/Triton | Kernel and inference talks with code |
| [DeepLearning.AI: Fast & Efficient LLM Inference with vLLM](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm/) | DeepLearning.AI + vLLM | 2026-06-03 | video course + labs · B→I · EN | Python | "Optimize, deploy, and benchmark an open-source LLM with vLLM" |
| [DeepLearning.AI: Efficiently Serving LLMs](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/) | DeepLearning.AI (with Predibase, from memory) | 2024 | short course · B · EN | Python | KV caching, batching, quantization, LoRA serving labs |
| [Modular handbook: Calculating GPU memory for serving LLMs](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms.md) | Modular | maintained | web · B · EN | none | Memory-sizing worked method |
| [vLLM disaggregated serving guide](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide) | vLLM | 2026-09-29 | tutorial blog · I · EN | 2+ GPUs | Run P/D end to end |

### Inferences
An exercise ladder, from beginner to advanced (my design, built on the resources above):
1. **KV-cache and capacity napkin math (B):** compute KV bytes/token for the configs above. Then compute how many 8K-token Llama-3.1-8B sequences fit on an 80 GB GPU: weights ≈ 16 GB in BF16, leaving about 60 GiB, and 60 GiB / 128 KiB per token ≈ 491k tokens, or about 60 sequences of 8K. Repeat with FP8 KV and with DeepSeek-V3 MLA. Extension: use the DeepSeek-V3 config's per-head dims to compute what MHA would need, and why MLA matters. Compare against the Modular handbook calculator and vLLM's reported KV capacity at startup.
2. **Speed-of-light decode (B→I):** lower-bound per-token latency ≈ (weight bytes + KV bytes read) / HBM bandwidth. For Llama-3.1-8B BF16 on an H100 (about 16 GB / 3.35 TB/s ≈ 4.8 ms), batch-1 decode is capped near 200 tok/s. Measure the real number and compute the bandwidth-utilization gap, as zeux and Hazy do. Then find the batch size where weight matmuls turn compute-bound. NVIDIA's dense BF16 TFLOPS for H100 should be taken from the datasheet; the ridge point is about 300 FLOPs/byte. Work the scaling book's inference problems.
3. **Benchmark like an SRE (I):** serve a model in vLLM or SGLang. Sweep request rate with GuideLLM, AIPerf or the engine's bench tool. Report p50/p99 TTFT, ITL/TPOT and throughput, and goodput at an SLO (e.g., TTFT < 1 s, TPOT < 50 ms). Then toggle one thing at a time: prefix caching (with a shared-system-prompt dataset), the chunked-prefill token budget, CUDA graphs (eager vs graphs), FP8 KV, and max batch size.
4. **Build a mini engine (I→A):** implement a KV cache (Raschka), then a block allocator with block tables (PagedAttention), then an iteration-level scheduler with chunked prefill, then a radix-tree prefix cache. Check your design against nano-vllm, mini-sglang and the tiny-llm course.
5. **Kernels (I→A):** derive online softmax and implement a FlashAttention forward pass in Triton or CUDA (UW CSE 599M notes, flash-attention-minimal). Benchmark against PyTorch SDPA/FlashAttention, then implement split-KV decode (FlashDecoding) and measure long-context batch-1 gains.
6. **Speculative decoding (I):** implement speculative sampling (Jay Mody) and verify empirically that the output distribution matches the target model. Measure acceptance vs temperature. Then run MTP or EAGLE-3 in vLLM/SGLang and plot speedup vs batch size and concurrency, reproducing the "Performance or Illusion?" finding that verification cost and batch size erode gains.
7. **Quantization (I):** use LLM Compressor to produce FP8, W4A16 (GPTQ/AWQ) and, if you have Blackwell, NVFP4 variants. Measure accuracy (a small eval suite) and throughput. Enable FP8 KV and compare long-context capacity.
8. **Disaggregation (I→A, needs 2+ GPUs):** follow vLLM's disaggregated-serving guide (NIXL). Compare colocated vs P/D on a mixed long-prompt/short-prompt workload, and explain the TTFT/ITL changes via DistServe's interference argument.
9. **MoE serving math (I→A):** using DeepSeek Day 6, compute experts per GPU for EP32 vs EP144 with 32 redundant experts. Explain dual-batch overlap. Study the EPLB repo's balancing algorithm and the DeepEP README; profile all-to-all if you have multi-GPU access.
10. **Overheads and determinism (A):** profile a decode step with Nsight Systems and count kernels and launch gaps (cf. Hazy's "around a hundred separate kernels"). Reproduce batch-size-dependent output drift (Thinking Machines) and test engine deterministic/batch-invariant modes.
11. **Structured output (I):** write a tiny regex-to-FSM logits mask, then compare XGrammar vs Outlines backends on JSONSchemaBench-style schemas for latency per token.

### Gaps
- I did not run any of the exercises; numeric results above are arithmetic from configs and cited hardware figures. The H100 dense BF16 TFLOPS value (needed for an exact ridge point) was not re-verified in-session.
- Some feature details of hands-on repos (nano-vllm's feature list, nanochat's engine, tiny-llm's week-by-week syllabus) were not re-read in-session. Check their READMEs.
- I did not confirm that the DeepLearning.AI vLLM course is free; DeepLearning.AI short courses usually are, but this was not checked.
