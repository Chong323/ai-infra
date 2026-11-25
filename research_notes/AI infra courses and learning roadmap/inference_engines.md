# Open-source LLM inference/serving engines (vLLM, SGLang, TensorRT-LLM, llama.cpp and others) and the best resources for learning their internals — state as of 2026-11-24

## 1. What are the major open-source inference engines in late 2026, how are they built, how do they compare in adoption and maturity, and what changed in 2025–2026?

### Takeaway
vLLM (about 97.5k GitHub stars, v0.34.0 released 2026-11-16) and SGLang (about 39k stars, v0.5.23, with a 1.0 "stability release" under way) are the two dominant open-source datacenter serving engines. Both are extremely active (about 2.3–2.4k commits in the last month), both spun out venture-backed companies in 2026 (Inferact and RadixArk) while staying open source, and both now have neutral governance: vLLM is a PyTorch Foundation-hosted project, and SGLang sits under the non-profit LMSYS with a new industry consortium. Around them, TensorRT-LLM became a PyTorch-native engine (its TensorRT backend was removed in 1.2). llama.cpp reached v1.0.0 with semantic versioning after ggml.ai joined Hugging Face. Ollama remains the most-starred local runner. Hugging Face TGI went into maintenance mode and was archived on 2026-03-21. DeepSpeed-MII, ExLlamaV2, Tokasaurus and ScaleLLM are dormant.

### Cited Findings

#### Data provenance
- Stars, forks, last push and latest release for every repo below were retrieved on 2026-11-24 from the ungh.cc mirror of the GitHub API. The gh CLI and GitHub API were not authorized in this environment. Example: [ungh.cc/repos/vllm-project/vllm](https://ungh.cc/repos/vllm-project/vllm)
- "Commits last month", "open issues" and "open good-first-issue" counts come from shields.io GitHub badges retrieved the same day. Example: [shields commit-activity vLLM](https://img.shields.io/github/commit-activity/m/vllm-project/vllm.json)

#### Engine comparison snapshot (2026-11-24)

| Engine | Repo (current path) | Stars | Latest release (date) | Last push | Commits last month | Open issues / open "good first issue" | Status | Source |
|---|---|---|---|---|---|---|---|---|
| vLLM | vllm-project/vllm | 97,488 | v0.34.0 (2026-11-16) | 2026-11-24 | 2.4k | 3k / 14 | Very active | [ungh](https://ungh.cc/repos/vllm-project/vllm), [shields](https://img.shields.io/github/commit-activity/m/vllm-project/vllm.json) |
| SGLang | sgl-project/sglang | 38,987 | v0.5.23 (2026-11-02) | 2026-11-23 | 2.3k | 1k / 46 | Very active, 1.0 in progress | [ungh](https://ungh.cc/repos/sgl-project/sglang), [shields](https://img.shields.io/github/commit-activity/m/sgl-project/sglang.json) |
| TensorRT-LLM | NVIDIA/TensorRT-LLM | 15,121 | v1.3.0 (2026-11-20); main is at 1.4.0rc5 | 2026-11-24 | 1.1k | 747 / 0 | Very active | [ungh](https://ungh.cc/repos/NVIDIA/TensorRT-LLM), [README](https://github.com/NVIDIA/TensorRT-LLM) |
| llama.cpp (ggml) | ggml-org/llama.cpp | 136,167 | v1.0.0 (2026-11-13) | 2026-11-23 | 833 | 706 / 14 | Very active | [ungh](https://ungh.cc/repos/ggml-org/llama.cpp) |
| Ollama | ollama/ollama | 187,952 | v0.40.18 (2026-11-24) | 2026-11-24 | 70 | 2.7k / 3 | Active | [ungh](https://ungh.cc/repos/ollama/ollama) |
| LMDeploy | InternLM/lmdeploy | 8,156 | v0.18.0 (2026-09-28) | 2026-11-23 | 52 | 550 / 0 | Active | [ungh](https://ungh.cc/repos/InternLM/lmdeploy) |
| MLC-LLM | mlc-ai/mlc-llm | 23,433 | no GitHub releases | 2026-11-21 | 14 | 306 / 0 | Low activity | [ungh](https://ungh.cc/repos/mlc-ai/mlc-llm) |
| WebLLM | mlc-ai/web-llm | 19,402 | v0.2.89 (2026-11-17) | 2026-11-20 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/mlc-ai/web-llm) |
| ExLlamaV3 | turboderp-org/exllamav3 | 1,861 | v1.6.7 (2026-11-13) | 2026-11-21 | 64 | 56 / 0 | Active | [ungh](https://ungh.cc/repos/turboderp-org/exllamav3) |
| ExLlamaV2 | turboderp-org/exllamav2 | 4,656 | v0.3.2 (2025-07-13) | 2026-03-04 | n/a | n/a | Superseded by V3 | [ungh](https://ungh.cc/repos/turboderp-org/exllamav2) |
| LightLLM | ModelTC/LightLLM | 4,371 | v1.2.0 (2026-08-10) | 2026-11-22 | 65 | 100 / 0 | Active | [ungh](https://ungh.cc/repos/ModelTC/lightllm) |
| HF TGI | huggingface/text-generation-inference | 10,876 | v3.3.7 (2025-12-19) | 2026-03-21 | 0 | 285 / 0 | **Archived 2026-03-21** | [ungh](https://ungh.cc/repos/huggingface/text-generation-inference), [GitHub](https://github.com/huggingface/text-generation-inference) |
| DeepSpeed-MII / FastGen | deepspeedai/DeepSpeed-MII | 2,116 | v0.3.3 (2025-03-25) | 2025-06-30 | 0 | 198 / 1 | **Dormant** | [ungh](https://ungh.cc/repos/deepspeedai/DeepSpeed-MII) |
| mistral.rs | EricLBuehler/mistral.rs | 7,856 | v0.9.5 (2026-10-19) | 2026-11-20 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/EricLBuehler/mistral.rs) |
| KTransformers | kvcache-ai/ktransformers | 19,697 | v0.7.2 (2026-10-15) | 2026-11-24 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/kvcache-ai/ktransformers) |
| xLLM (JD) | xLLM-AI/xllm (was jd-opensource/xllm) | 1,630 | v0.12.0 (2026-10-31) | 2026-11-23 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/jd-opensource/xllm) |
| Sonar (formerly Aphrodite Engine) | dphnAI/sonar (was aphrodite-engine/aphrodite-engine) | 1,913 | v0.24.1 (2026-09-11) | 2026-11-22 | n/a | n/a | Active, renamed | [ungh](https://ungh.cc/repos/aphrodite-engine/aphrodite-engine) |
| Xinference | xorbitsai/inference | 9,668 | v4.2.0 (2026-11-07) | 2026-11-24 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/xorbitsai/inference) |
| mlx-lm (Apple MLX) | ml-explore/mlx-lm | 7,740 | v0.32.2 (2026-10-16) | 2026-11-21 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/ml-explore/mlx-lm) |
| llamafile | mozilla-ai/llamafile (was Mozilla-Ocho) | 26,702 | 0.10.6 (2026-09-15) | 2026-11-09 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/Mozilla-Ocho/llamafile) |
| LocalAI | mudler/LocalAI | 50,512 | v5.0.1 (2026-11-20) | 2026-11-23 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/mudler/LocalAI) |
| KoboldCpp | LostRuins/koboldcpp | 12,381 | v1.124.2 (2026-10-31) | 2026-11-24 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/LostRuins/koboldcpp) |
| BitNet (1-bit LLM inference) | microsoft/BitNet | 42,458 | no releases | 2026-07-27 | n/a | n/a | Slower | [ungh](https://ungh.cc/repos/microsoft/BitNet) |
| PowerInfer | Tiiny-AI/PowerInfer (was SJTU-IPADS) | 9,900 | no releases | 2026-05-11 | n/a | n/a | Slower | [ungh](https://ungh.cc/repos/SJTU-IPADS/PowerInfer) |
| Triton Inference Server | triton-inference-server/server | 11,161 | v2.75.0 (2026-11-17) | 2026-11-24 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/triton-inference-server/server) |
| OpenVINO GenAI | openvinotoolkit/openvino.genai | 624 | 2026.4.1.0 (2026-10-01) | 2026-11-23 | n/a | n/a | Active | [ungh](https://ungh.cc/repos/openvinotoolkit/openvino.genai) |
| IPEX-LLM (Intel) | intel/ipex-llm | 8,842 | v2.2.0 (2025-04-07) | 2026-01-28 | n/a | n/a | Appears inactive | [ungh](https://ungh.cc/repos/intel/ipex-llm) |
| Tokasaurus (Stanford) | ScalingIntelligence/tokasaurus | 483 | none | 2025-11-25 | n/a | n/a | **Dormant** | [ungh](https://ungh.cc/repos/ScalingIntelligence/tokasaurus) |
| ScaleLLM | vectorch-ai/ScaleLLM | 497 | v0.2.6 (2025-09-13) | 2025-12-19 | n/a | n/a | **Dormant** | [ungh](https://ungh.cc/repos/vectorch-ai/ScaleLLM) |

Supporting libraries that the engines build on, plus out-of-scope orchestration that integrates with them:

| Project | Role | Stars | Latest release | Source |
|---|---|---|---|---|
| FlashInfer | Kernel library for LLM serving (used by SGLang and Mini-SGLang) | 7,000 | v0.7.5.post1 (2026-11-20) | [ungh](https://ungh.cc/repos/flashinfer-ai/flashinfer), [Mini-SGLang README](https://github.com/sgl-project/mini-sglang) |
| LMCache | KV-cache layer | 12,464 | v0.5.8 (2026-11-22) | [ungh](https://ungh.cc/repos/LMCache/LMCache) |
| Mooncake | KV-cache transfer and store, the serving platform for Kimi. vLLM ships Mooncake connectors | 7,125 | v0.3.14 (2026-11-03) | [ungh](https://ungh.cc/repos/kvcache-ai/Mooncake), [vLLM docs index](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/) |
| NVIDIA Dynamo (out of scope) | Datacenter-scale distributed serving framework | 8,554 | v1.6.2 (2026-11-05) | [ungh](https://ungh.cc/repos/ai-dynamo/dynamo) |
| llm-d (out of scope) | Kubernetes inference stack | 5,340 | v1.0.0 (2026-10-20) | [ungh](https://ungh.cc/repos/llm-d/llm-d) |
| HF Transformers | Model-definition library with built-in continuous batching | 169,281 | v5.24.0 (2026-11-21) | [ungh](https://ungh.cc/repos/huggingface/transformers), [CB architecture docs](https://huggingface.co/docs/transformers/continuous_batching_architecture) |

#### vLLM: architecture, distinguishing features, 2025–2026 changes
- **V1 engine (alpha announced 2025-01-27).**
  - The core loop moved into an isolated `EngineCore` process "that focuses exclusively on the scheduler and model executor", so tokenization and streaming overlap GPU work.
  - The scheduler "removes the traditional distinction between 'prefill' and 'decode' phases" and uses a fixed token budget.
  - Prefix caching is on by default with "near-zero performance degradation, even when the cache hit rate is 0%".
  - V1 adds a persistent batch ("caches the input tensors and only applies the diffs"), piecewise CUDA graphs plus torch.compile, FlashAttention 3, an encoder cache for multimodal inputs, and a symmetric tensor-parallel architecture.
  - It claims "up to 1.7x higher throughput" than V0. — [vLLM blog: V1](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- **V0 is gone.** The docs say "We have fully deprecated V0" (RFC #18571). V1 re-architected the scheduler, KV cache manager, worker, sampler and API server, and aims for "a simple, modular, and easy-to-hack codebase" with "near-zero CPU overhead". Some V0 features were dropped, such as `best_of` and GPU<->CPU KV swapping. — [vLLM V1 guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)
- **Process architecture (current docs).**
  - An API server process handles HTTP, tokenization and multimodal loading, and talks to the EngineCore process or processes over ZMQ sockets.
  - Separate GPU worker processes run the model, plus a conditional DP-coordinator process.
  - The doc also covers scheduling policies, preemption, async scheduling, LLMEngine/AsyncLLMEngine, Worker, Model Runner and the model class hierarchy. — [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- **Model Runner V2 (MRV2), announced 2026-03-24.**
  - It is a ground-up rewrite. The old `gpu_model_runner.py` had grown past 6,700 lines, while the largest MRV2 file is under 1,300 lines.
  - A `ModelState` abstraction isolates model-specific logic. The design is "async-first" and aims for zero CPU-GPU synchronization.
  - Input preparation moves to the GPU in Triton kernels, and the sampler is largely reimplemented in Triton. — [vLLM blog: MRV2](https://vllm.ai/blog/2026-03-24-mrv2)
  - The design doc admits that "In hindsight, many of V1's design choices were suboptimal". — [MRV2 design doc](https://docs.vllm.ai/en/latest/design/model_runner_v2/)
- **v0.34.0 (2026-11-16).**
  - "1,061 commits from 412 contributors (120 new)."
  - "Model Runner V1 is gone. Model Runner V2 is now the only GPU runner." Setting `VLLM_USE_V2_MODEL_RUNNER=0` now fails.
  - The Rust frontend was expanded (stateless Responses API, `n > 1`). An extensible KV cache, sized from free memory after warmup, is on by default.
  - `--block-size` now defaults to 64 (it was 16). ROCm images and wheels are ROCm 10.0 only. — [vLLM v0.34.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)
- **Rust frontend in the build workflow.** The contributor guide now includes `./tools/build_rust.sh` "to rebuild only the Rust frontend binary". — [vLLM contributing guide](https://docs.vllm.ai/en/latest/contributing/)
- **Design-doc themes that show architectural direction.** There are design docs for a hybrid KV cache manager, dual batch overlap, the plugin system, "vLLM IR: Functional Intermediate Representation", the fused MoE modular kernel, NIXL push-mode KV transfer and HiSparse local KV offload. — [vLLM docs search index (design/)](https://docs.vllm.ai/en/latest/search/search_index.json)
- **Governance.** vLLM became a PyTorch ecosystem project in December 2024. It became a PyTorch Foundation-hosted project, alongside DeepSpeed, when the Foundation became an umbrella foundation in May 2025. Ray followed as a hosted project on 2025-10-22. — [PyTorch blog: Foundation welcomes vLLM](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/); [PyTorch press release vLLM + DeepSpeed](https://pytorch.org/blog/press-release-pytorch-foundation-expands-welcomes-projects-vllm-deepspeed/); [Ray joins](https://pytorch.org/blog/pytorch-foundation-welcomes-ray-to-deliver-a-unified-open-source-ai-compute-stack/)
- **Commercial spin-out (Inferact).** Inferact raised a $150M seed at an $800M valuation on 2026-01-22, co-led by Andreessen Horowitz and Lightspeed. CEO Simon Mo is "one of vLLM's original creators". — [TechCrunch](https://techcrunch.com/2026/01/22/inference-startup-inferact-lands-150m-to-commercialize-vllm/)
  - Woosuk Kwon and Simon Mo are listed as Inferact co-founders, and Inferact hosted the first vLLM Conference. — [vLLM Conference 2026](https://vllm.ai/events/vllm-conference/2026)
  - Coverage reports that the company says it will keep vLLM open source (from a search summary; not verified on the article page). — [SiliconANGLE](https://siliconangle.com/2026/01/22/inferact-launches-150m-funding-commercialize-vllm/)
- **Self-reported scale (PyTorch Conference Europe 2026 "Project Update" deck).** The presenter is Tyler Michael Smith, Core Maintainer of vLLM and Chief Architect, Inference Engineering at Red Hat.
  - "70K+ GitHub stars, 800+ PRs/month", "500K++ GPUs deployed 24/7", "10K+ members in slack.vllm.ai", ">2000 Contributors from >50 major companies" and ">100 arches".
  - Hardware: CUDA, ROCm, Gaudi/XPU, TPU, Neuron, Ascend, CPU, Metal, MACA, RBLN, Spyre, MLU and Kunlun.
  - Wide-EP reached "26.2k tokens/s per GPU on P and 10.1K tokens/s per GPU on D" on H200/GB200.
  - The Q2 roadmap is organized by SIG: core (MRV2 toward default, a KV cache manager rethink), CI, large-scale serving, torch.compile, multimodality and omni. — [vLLM deck PDF](https://hosted-files.sched.co/pytorchconferenceeu2026/3e/2.%20Tyler%20Michael%20Smith.pdf)
- **Ecosystem repos (stars, latest release).**
  - vllm-omni: 7,914, v0.34.0. semantic-router: 6,523, v0.4.0. aibrix: 5,202, v0.8.0. llm-compressor: 3,988, 0.15.0.
  - vllm-ascend: 3,139, v0.30.0. production-stack: 2,733, vllm-stack-0.1.14. guidellm: 1,808, v0.8.1. recipes: 1,099. speculators: 941, v0.9.0.
  - The tpu-inference repo description now reads "Moving to vllm-project/vllm-jax". — [ungh vllm-omni](https://ungh.cc/repos/vllm-project/vllm-omni); [ungh tpu-inference](https://ungh.cc/repos/vllm-project/tpu-inference); [ungh recipes](https://ungh.cc/repos/vllm-project/recipes)
- **2025–2026 feature arc, from blog titles and dates.**
  - Hardware plugins: Ascend, 2025-05-12.
  - Large-scale serving: "DeepSeek @ 2.2k tok/s/H200 with Wide-EP", 2025-12-17; "WideEP ... on Blackwell (Part I)", 2026-02-03; Elastic EP, 2026-05-14.
  - KV cache: KV offloading connector, 2026-01-08; tiered KV cache offloading, 2026-09-10.
  - Disaggregation: "Taking vLLM Apart: A Practical Guide to Disaggregated Serving", 2026-09-29; encoder disaggregation, 2025-12-15 and 2026-11-11.
  - New projects and platforms: vLLM-Omni, 2025-11-30; vllm-metal for Apple Silicon, 2026-09-22; vLLM TorchTPU, 2026-10-20.
  - Engine plumbing: native RL APIs, 2026-05-28; migrating to the LibTorch Stable ABI, 2026-11-06. — [vLLM blog index](https://vllm.ai/blog)
- **New home on the web.** blog.vllm.ai now 301-redirects to vllm.ai/blog. The new vllm.ai website launched on 2025-12-27, with an Events page, contact addresses (including talentpool@vllm.ai, which forwards resumes to partner companies) and a "vLLM Daily" repository that summarizes daily changes. — [vLLM: Announcing vllm.ai website](https://vllm.ai/blog/2025-12-27-vllm-ai-website)

#### SGLang: architecture, distinguishing features, 2025–2026 changes
- **Architecture.** SGLang's high-level design is a frontend API server, a tokenizer server and a backend scheduler per GPU, with Radix Attention / Radix Cache for prefix KV reuse, chunked prefill, overlap scheduling and tensor parallelism. SGLang's codebase "has grown to nearly 300k lines of Python". — [LMSYS: Mini-SGLang](https://lmsys.org/blog/2025-12-17-minisgl/)
- **Earlier landmark design post.** "SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs" (2024-12-04). — [sgl-learning-materials README](https://github.com/sgl-project/sgl-learning-materials)
- **Large-scale EP plus PD disaggregation (2025-05-05).**
  - Ran on 12 nodes × 8 H100 (96 GPUs) with DeepEP, DeepGEMM, two-batch overlap and EPLB.
  - Reached 52.3k input and 22.3k output tokens/s per node for 2,000-token inputs.
  - Billed as "the first open-source implementation to nearly match the throughput reported in the official DeepSeek blog", at about "$0.20/1M output tokens". — [LMSYS: large-scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
  - The team called SGLang "the first fully open-source LLM serving engine to support large-scale Expert-Parallelism (EP) and Prefill-Decode disaggregation". — [sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)
- **Ecosystem status.** SGLang joined the PyTorch ecosystem on 2025-03-19. This is the independently governed tier, not Foundation-hosted. — [PyTorch blog: SGLang joins PyTorch](https://pytorch.org/blog/sglang-joins-pytorch/)
- **2025–2026 LMSYS blog timeline, from post slugs and dates.**
  - 2025: GB200 NVL72 parts 1 and 2 (06-16, 09-25); slime RL (07-09); SpecForge speculative-decoding training (07-25); HiCache hierarchical KV cache (09-10); deterministic inference (09-22); SGLang-JAX (10-29); SGLang Diffusion (11-07); Miles RL (11-19); Mini-SGLang (12-17).
  - 2026: EPD disaggregation (01-12); unified radix cache (08-11); advanced CUDA graph (08-17); fast recovery (08-21); low-latency mode (10-18); TorchTPU (10-20); Trainium beta (10-22); HiCache distributed serving (11-02); SGLang-Neuron v0.1.0 and the Consortium (11-12); Summit recap (11-17). — [LMSYS blog source tree via ungh](https://ungh.cc/repos/lm-sys/lm-sys.github.io/files/main)
- **Commercial spin-out (RadixArk).**
  - TechCrunch reported on 2026-01-21, citing sources, that SGLang was spinning out as RadixArk at a $400M valuation. — [TechCrunch](https://techcrunch.com/2026/01/21/sources-project-sglang-spins-out-as-radixark-with-400m-valuation-as-inference-market-explodes)
  - On 2026-05-05 RadixArk announced $100M led by Accel with Spark Capital, NVentures, AMD, Databricks and Hock Tan. It said "SGLang powers AI clusters that contain more than 400,000 graphics cards in aggregate" (a company claim) and that it builds Miles for RL. — [SiliconANGLE](https://siliconangle.com/2026/05/05/nvidia-amd-back-100m-round-ai-tooling-startup-radixark/)
  - Ying Sheng is RadixArk co-founder and CEO. — [LMSYS: SGLang Consortium](https://www.lmsys.org/blog/2026-11-12-sglang-consortium)
- **SGLang Consortium (2026-11-12).**
  - Founding members are NVIDIA, AMD, Intel, Google, Meta, Thinking Machines Lab, Cloudflare, Modal and RadixArk. Each member names a technical representative and joins two roadmap meetings a year.
  - "SGLang's maintainers keep responsibility for technical decisions and code acceptance."
  - SGLang runs on "hundreds of thousands of GPUs". Named production users are Meta, Microsoft, Google, Amazon, Thinking Machines Lab, Cloudflare, Reflection and Modal. — [LMSYS: SGLang Consortium](https://www.lmsys.org/blog/2026-11-12-sglang-consortium)
- **SGLang Summit (2026-11-12/13, San Francisco).**
  - More than 2,000 attendees and more than 90 technical talks. SGLang "grew to 332 organizations". Meta reported more than 2× GPU savings after migrating.
  - SGLang v1.0 was announced, with the note that "Agents now produce most merged PRs". Miles v0.2 and NVIDIA NIM Certified microservices powered by SGLang were also announced. — [LMSYS: Summit recap](https://lmsys.org/blog/2026-11-17-sglang-summit-recap/)
- **SGLang 1.0 tracker (issue #50045, opened 2026-11-12).**
  - It is "a stability release" with no target date set yet. SGLang "does not silently downgrade a configuration or override a value you set".
  - sgl-router 0.1.0 is released and replaces sgl-model-gateway. The Rust frontend server is experimental and off by default.
  - `--disable-cuda-graph` and `--enable-dp-attention` are removed in 1.0. — [SGLang issue #50045](https://github.com/sgl-project/sglang/issues/50045)
- **Docs moved.** The docs now live at docs.sglang.io, built on Mintlify. docs.sglang.ai returns 301 → docs.sglang.io, and SGLang is described as "hosted under the non-profit LMSYS". — [docs.sglang.io](https://docs.sglang.io/) (redirect checked with curl on 2026-11-24)

#### Other engines: architecture, distinguishing features, 2025–2026 changes
- **TensorRT-LLM (NVIDIA).**
  - 1.0: "the PyTorch-based architecture is now stable and the default experience", and the LLM API is stable.
  - 1.1: added a KV Cache Connector API for disaggregated serving, B300/GB300 support and GPT-OSS.
  - 1.2: "PyTorch is now the sole execution backend" (the TensorRT backend was removed) and two-model speculative decoding was dropped.
  - 1.3 deprecates the TRITON MoE backend "as of TensorRT-LLM 1.3 (2026-09)". 1.4 deprecates KV cache manager V1 and the C++ cache transceiver. — [TRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/latest/release-notes.html); [Migration guide: TensorRT backend removed](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)
  - The README describes "specialized kernels for common operations, an efficient runtime, and a pythonic framework". It links an architecture overview and a numbered tech-blog series, for example "NVFP4 MLA" (11/17) and "Dense DWDP ... on Vera Rubin NVL72" (11/11). — [TRT-LLM README](https://github.com/NVIDIA/TensorRT-LLM)
- **llama.cpp / ggml.**
  - The ggml.ai team, Georgi Gerganov and others, joined Hugging Face on 2026-02-20. The projects stay open source. — [Simon Willison](https://simonwillison.net/2026/Feb/20/ggmlai-joins-hugging-face/)
  - libllama and libggml reached v1.0.0 on 2026-11-13, and future API/ABI breaks will bump the major version under semver. llama.cpp "supports more than 150 architecture variants as well as all mainstream speculative decoding methods", and ggml backends can be built as runtime-loaded dynamic libraries. — [HF blog: llama.cpp v1](https://huggingface.co/blog/ggml-org/llama-cpp-v1)
  - v1.0.0 "removes the deprecated public API and promotes the NextN/speculative API to `llama.h`". It also hardens server security, adds `/v1/audio/speech`, redesigns the WebUI and folds `clip.h` into `mtmd`. Nightly builds keep `bNNNN` tags. — [llama.cpp v1.0.0 release](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0)
- **Ollama.**
  - Ollama 0.30 (2026-06-05) added "GGUF model compatibility through llama.cpp", which "augments Ollama's MLX engine on Apple silicon". Vulkan is on by default. — [Ollama blog](https://ollama.com/blog/improved-performance-and-model-support-with-gguf)
  - Ollama's May 2025 "new engine for multimodal models" was discussed on Hacker News. This is a secondary source. — [HN thread mirror](https://alt-hn.vercel.app/item/44001087)
  - The README now pitches "Access the latest releases locally or in the cloud". — [Ollama README](https://github.com/ollama/ollama)
- **LMDeploy (Shanghai AI Lab / InternLM).**
  - It has two engines: the C++ TurboMind engine and a "PyTorch inference engine, developed entirely in Python".
  - 2025 additions: DeepSeek PD disaggregation via DLSlime and Mooncake (2025/06), and TurboMind MXFP4 on GPUs from V100 onward, self-reported at "1.5x the performance of vLLM on H800" for gpt-oss (2025/09).
  - The TurboMind paper was accepted to EuroSys 2027. — [LMDeploy README](https://github.com/InternLM/lmdeploy)
- **MLC-LLM.** It is an "ML compiler and high-performance deployment engine" built on TVM, running "MLCEngine" with an OpenAI-compatible API across REST, Python, JavaScript, iOS and Android. — [MLC-LLM README](https://github.com/mlc-ai/mlc-llm). It has no GitHub releases and only 14 commits in the last month, while its sibling WebLLM is active. — [ungh](https://ungh.cc/repos/mlc-ai/mlc-llm); [shields](https://img.shields.io/github/commit-activity/m/mlc-ai/mlc-llm.json)
- **ExLlamaV3.** It uses "EXL3, based on QTIP" with 2–8 bit cache quantization. It offers tensor- and expert-parallel inference for consumer setups, CPU offloading for MoE, continuous dynamic batching, speculative decoding and multimodal support. TabbyAPI is "the official and recommended backend server". — [ExLlamaV3 README](https://github.com/turboderp-org/exllamav3)
- **LightLLM (ModelTC).**
  - A "Python-based LLM inference and serving framework".
  - Releases: v1.0.0 (2025/02, billed as "fastest DeepSeek-R1 serving performance on single H200 machine"), v1.1.0 (2025/09), and prefix KV transfer between DP ranks (2025/11).
  - Research: a Past-Future scheduler (ASPLOS'25) and Pre³ constrained decoding (ACL 2025 outstanding paper). — [LightLLM README](https://github.com/ModelTC/LightLLM)
- **Hugging Face TGI.**
  - The README banner says "text-generation-inference is now in maintenance mode". It will accept only "minor bug fixes, documentation improvements and lightweight maintenance tasks" and recommends vLLM and SGLang, plus llama.cpp and MLX locally. — [TGI README](https://raw.githubusercontent.com/huggingface/text-generation-inference/main/README.md)
  - The GitHub banner reads "This repository was archived by the owner on Mar 21, 2026. It is now read-only." — [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- **DeepSpeed-MII / FastGen.** The README's "Latest News" ends at 2024/01 (FastGen). Its technologies are blocked KV caching, continuous batching and Dynamic SplitFuse. — [DeepSpeed-MII README](https://github.com/deepspeedai/DeepSpeed-MII). The last release was v0.3.3 on 2025-03-25 and the last push was 2025-06-30. — [ungh](https://ungh.cc/repos/deepspeedai/DeepSpeed-MII)
- **Notable newer or other entrants.**
  - mistral.rs (Rust): paged attention is the default on supported Metal models (v0.9.5), and it serves OpenAI- and Anthropic-compatible APIs. — [mistral.rs README](https://github.com/EricLBuehler/mistral.rs)
  - KTransformers: "efficient inference and fine-tuning ... through CPU-GPU heterogeneous computing" for MoE models. — [KTransformers README](https://github.com/kvcache-ai/ktransformers)
  - xLLM (JD): "officially donated to the OpenAtom Foundation" on 2026-07-06. — [xLLM README](https://github.com/xLLM-AI/xllm)
  - Sonar: the former Aphrodite Engine, "based on vLLM", which serves the Dolphin Inference Network and PygmalionAI. — [Sonar README](https://github.com/dphnAI/sonar)

### Inferences
- **Two leaders, converging designs.** vLLM and SGLang are the two clear leaders for GPU datacenter serving, and their feature sets are converging. Both now have wide-EP, PD and encoder disaggregation, hierarchical or offloaded KV caches, RL weight-update APIs and Rust frontends. vLLM leads on community size, hardware-plugin breadth and the number of ecosystem sub-projects. SGLang was the first open-source engine to show large-scale EP with PD disaggregation, and it is heavily used for frontier MoE models and RL rollouts.
- **Stars vs. fleet share.** Stars measure developer reach, not GPU fleet share. The local-first runners (Ollama 188k, llama.cpp 136k) far outnumber datacenter engines in stars. The GPU-deployment claims (vLLM 500K+, SGLang 400K+ GPUs) are self-reported, from different dates and methods, so they should not be compared head to head.
- **Version numbers understate maturity.** vLLM is still 0.x despite its production use, SGLang is preparing 1.0, TensorRT-LLM is 1.x and llama.cpp just declared 1.0.
- **Activity tiers.** Commits in the last month cluster into three tiers:
  - very high (vLLM, SGLang, TensorRT-LLM, llama.cpp);
  - moderate (LMDeploy, LightLLM, ExLlamaV3, Ollama, at roughly 50–70);
  - near-zero (MLC-LLM, TGI, DeepSpeed-MII).
- **Advice for a learner.** Learners should not invest in TGI, DeepSpeed-MII or ExLlamaV2 internals. TensorRT-LLM's pre-1.0 "build a TensorRT engine" tutorials are obsolete.
- **Vendor convergence on PyTorch.** TensorRT-LLM is now PyTorch-native, and vLLM is migrating to the LibTorch stable ABI. Knowing PyTorch internals (torch.compile, CUDA graphs, custom ops) is therefore a transferable core skill across engines.
- **Contributor counts.** The shields.io contributor badges returned about 445–450 for every large repo, which looks capped. Contributor totals should come from project statements instead, such as vLLM's ">2000 contributors" or the 412 contributors to v0.34.0 alone.

### Gaps
- There are no independent, third-party adoption statistics (GPU fleet share, request volume). All GPU and company counts are self-reported by the projects or their companies.
- I could not find the exact release in which vLLM deleted the V0 code. The docs only say V0 is "fully deprecated" and point to RFC #18571.
- Third-party sources date TGI's maintenance-mode switch to 2025-12-11, but the README banner has no date, so that date is unverified.
- Ollama's engine architecture (Go "Ollama engine" vs llama.cpp runner vs MLX) is only partially documented in primary sources. A WebProNews claim about a new engine in "0.17" was not verified.
- MLC-LLM's maintenance status is ambiguous. It shows a recent push but low commit volume and no releases, and I found no official statement.
- The Inferact "will keep vLLM open source" commitment was seen only in search summaries, not confirmed on a fetched page.
- Intel IPEX-LLM's archival status could not be verified, because github.com pages are blocked for curl and I did not fetch it with WebFetch.

## 2. What official learning resources exist for vLLM and SGLang?

### Takeaway
vLLM has the richest official learning stack:
- docs with 32 design documents and 20 contributor pages;
- biweekly recorded office hours (59 sessions so far);
- an engineering blog with many deep dives;
- a free DeepLearning.AI course (June 2026);
- recipes, plus a first vLLM Conference (Aug 2026).

SGLang's official material consists of:
- the docs.sglang.io Mintlify docs (strong Advanced Features and HiCache design pages, a Developer Guide and 177 cookbook pages);
- the LMSYS blog (123 posts in 2025–2026);
- the official Mini-SGLang teaching codebase;
- the sgl-learning-materials repo, which is only lightly maintained (last push 2026-01-05, dated links);
- talks from the first SGLang Summit, whose recordings were still "coming soon" at the time of writing.

### Cited Findings

#### vLLM official resources
(The Level and Prerequisites columns are this researcher's assessment. Other fields are from the cited source.)

| Name | URL | Maintainer | Latest version / update | Format | Level | Prerequisites | Language | Key topics | Why it stands out | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| vLLM documentation | https://docs.vllm.ai/en/latest/ | vLLM project | Tracks main (v0.34.x); 352 pages in the latest search index | Docs site | Beginner → advanced | Python; basic GPU/LLM concepts | English | Quickstart, offline/online serving, features (prefix caching, spec decoding with 15+ methods, structured outputs, LoRA, quantization), configuration, parallelism (DP/EP/CP), K8s and framework integrations, benchmarking CLI | Canonical, very complete | [docs search index](https://docs.vllm.ai/en/latest/search/search_index.json) |
| vLLM design docs (32 pages) | https://docs.vllm.ai/en/latest/design/arch_overview/ | vLLM project | Updated with main | Design docs | Intermediate → advanced | Python; PyTorch; transformer inference basics | English | Architecture overview (process model, scheduler, preemption, async scheduling, worker/model runner), Paged Attention, Automatic Prefix Caching, Hybrid KV Cache Manager, CUDA Graphs, torch.compile integration and fusion passes, Python multiprocessing, Plugin System, Model Runner V2, Metrics, Fused MoE modular kernel, Dual Batch Overlap, vLLM IR, attention backends | The closest thing to an official internals textbook. MRV2 doc explains why V1 choices were changed | [docs search index](https://docs.vllm.ai/en/latest/search/search_index.json); [MRV2 doc](https://docs.vllm.ai/en/latest/design/model_runner_v2/) |
| vLLM blog | https://vllm.ai/blog | vLLM team and partners | Posts through 2026-11-24 | Blog | Intermediate → advanced | Varies | English | V1 (2025-01-27), torch.compile intro (2025-08-20), "Inside vLLM" (2025-09-05), Triton attention backend deep dive (2026-03-04), MRV2 (2026-03-24), KV offloading connector (2026-01-08), disaggregated serving guide (2026-09-29), CI/benchmarking/release process (2026-07-16), plugin system (2025-11-20), debugging with CUDA core dumps (2025-08-11) and tracing hanging kernels (2025-12-03) | Engineering-level posts from core maintainers. blog.vllm.ai redirects here | [vLLM blog index](https://vllm.ai/blog) |
| vLLM Office Hours | https://vllm.ai/office-hours ; recordings: https://www.youtube.com/playlist?list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3 | Red Hat with vLLM committers | #59 on 2026-10-22 (59 sessions per the page counter) | Biweekly livestream (Thursdays 2:00 PM ET), recorded with slides | Beginner → advanced | None | English | About 15 min "What's new in vLLM" (from `mgoin`), 30 min guest talk, 15 min Q&A. Key sessions: #22 Intro to vLLM V1 (2025-03-27); #26 torch.compile (2025-05-29); #30 Hybrid Memory Allocator (2025-08-14); #35 How to Build and Contribute to vLLM (2025-10-23); #39 batch invariance (2026-01-08); #42 CPU Offloading Connector (2026-01-29); #55 Mooncake + vLLM/llm-d (2026-08-06) | Long-running, free, recorded; the best way to follow the project's evolution | [vLLM office hours page](https://vllm.ai/office-hours) |
| "Fast & Efficient LLM Inference with vLLM" (DeepLearning.AI short course) | https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/ | DeepLearning.AI + Red Hat; instructor Cedric Clyburn | Released early June 2026 (vLLM blog 2026-06-03) | Video course, about 1.5 h, 9 video lessons + 3 code labs | Intermediate (per course page) | Python; basic LLM concepts (tokens, inference, GPU usage) | English | KV cache and GPU memory hierarchy; quantization with LLM Compressor; deploying with vLLM's OpenAI-compatible API; continuous batching, PagedAttention and prefix caching; load testing with GuideLLM; quality checks with lm-eval | The only official structured course. Free, but quizzes and graded notebooks need the paid Pro tier | [DLAI search result](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/); [vLLM blog post](https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course); [Red Hat dev blog](https://developers.redhat.com/blog/2026/06/03/learn-optimize-deploy-and-benchmark-llms-vllm-new-free-course) |
| vllm-project/recipes | https://github.com/vllm-project/recipes | vLLM project | Last push 2026-11-24; about 1.1k stars | Repo of how-tos | Beginner → intermediate | Basic vLLM usage | English | Model-specific "common recipes to run vLLM" | Copy-paste deployment configs for current models | [ungh](https://ungh.cc/repos/vllm-project/recipes) |
| vLLM Conference 2026 (at Ray Summit) | https://vllm.ai/events/vllm-conference/2026 | Hosted by Inferact | 2026-08-24/26, San Francisco | Conference, about 17 talks | Intermediate → advanced | n/a | English | "State of vLLM 2026" (Woosuk Kwon and Zachary Xi) covering the roadmap and Model Runner V2 migration; "How a Transformers Model Loads in vLLM" (Harry Mellor, HF); agentic workloads with PD disaggregation and KV offloading | First dedicated vLLM conference. Recordings and slides are not linked on the page | [vLLM Conference 2026](https://vllm.ai/events/vllm-conference/2026) |
| PyTorch Conference sessions | https://pytorch.org/blog/vllm-sessions-at-pytorch-conference-north-america-2026/ | PyTorch Foundation | NA 2026: San Jose, Oct 20–21 | Talks | Intermediate → advanced | n/a | English | KV cache management, disaggregated serving, hardware portability, MoE inference | Several vLLM sessions in one place | [pytorch.org](https://pytorch.org/blog/vllm-sessions-at-pytorch-conference-north-america-2026/) |
| Meetups and events | https://vllm.ai/events (docs: https://docs.vllm.ai/en/latest/community/meetups/) | vLLM community | Ongoing (e.g., Korea Sep 2025 and Apr 2026; Vienna meetup streamed as office hours #45 on 2026-03-12) | In-person/streamed | All | n/a | Mostly English | Project updates and guest talks | Regional community and slides | [vLLM blog index](https://vllm.ai/blog); [office hours page](https://vllm.ai/office-hours); [docs meetups page](https://docs.vllm.ai/en/latest/community/meetups/) |
| Contributor docs (20 pages) | https://docs.vllm.ai/en/latest/contributing/ | vLLM project | Updated with main | Docs | Intermediate | Git, Python, (C++/CUDA optional) | English | Dev setup, profiling vLLM, incremental C++/CUDA compilation, adding a model (basic model, registration, unit tests, multimodal, speech-to-text), CI failures, nightly wheels, deprecation policy, release process, labels, editing agent instructions | Onboarding path to real code | [docs search index](https://docs.vllm.ai/en/latest/search/search_index.json) |

- vLLM's deck points to roadmap.vllm.ai, weekly SIG meetings listed at vllm.ai/events, and "slack.vllm.ai — 10K+ members, find your SIG". — [vLLM deck PDF](https://hosted-files.sched.co/pytorchconferenceeu2026/3e/2.%20Tyler%20Michael%20Smith.pdf)

#### SGLang official resources

| Name | URL | Maintainer | Latest version / update | Format | Level | Prerequisites | Language | Key topics | Why it stands out | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| SGLang documentation | https://docs.sglang.io/ (docs.sglang.ai 301-redirects here) | SGLang team / LMSYS | Live; machine-readable index at /llms.txt (387 entries) | Docs site (Mintlify) | Beginner → advanced | Python; GPU basics | English | Get Started; Basic Usage (OpenAI-, Anthropic- and Ollama-compatible APIs, offline engine, native APIs); Advanced Features: radix cache and eviction policies, attention backends, speculative decoding, structured outputs, tool/reasoning parsers, quantization, DP/DP-attention and sgl-router, prefill/decode context parallelism, expert parallelism, PD and EPD disaggregation, HiCache plus "HiCache System Design and Optimization", unified memory design, CUDA graph pages, deterministic inference, SGLang for RL; Supported Models; Developer Guide; References; Hardware Platforms | Feature-design pages double as internals documentation | [docs.sglang.io](https://docs.sglang.io/); [llms.txt](https://docs.sglang.io/llms.txt) |
| SGLang Cookbook | https://docs.sglang.io/cookbook/intro.md | SGLang team | 177 cookbook pages as of 2026-11-24 (e.g., DeepSeek-V4.1, GLM-5.3, Gemma 4) | Recipes | Beginner → intermediate | Basic SGLang usage | English | Verified launch commands and tuning per model and hardware (H100/H200/B200/B300/GB300/MI3xx/TPU) | Equivalent of vLLM recipes, integrated into docs | [llms.txt](https://docs.sglang.io/llms.txt) |
| LMSYS blog | https://lmsys.org/blog/ | LMSYS / SGLang team | 123 posts dated 2025–2026; latest 2026-11-23 | Blog | Intermediate → advanced | Varies | English | Large-scale EP/PD (2025-05-05), GB200 NVL72 (2025-06-16, 2025-09-25), HiCache (2025-09-10), deterministic inference (2025-09-22), Mini-SGLang (2025-12-17), EPD (2026-01-12), unified radix cache (2026-08-11), advanced CUDA graph (2026-08-17), fast recovery (2026-08-21), low-latency mode (2026-10-18), agent-assisted SGLang development (2026-07-02) | Primary source for SGLang's design decisions | [LMSYS blog tree via ungh](https://ungh.cc/repos/lm-sys/lm-sys.github.io/files/main) |
| sgl-project/sgl-learning-materials | https://github.com/sgl-project/sgl-learning-materials | SGLang team | Last push 2026-01-05; 908 stars. Newest slides dated 2025-10-22; newest "Announcement" May 2025; videos end Jan 2025 | Curated slides, videos, blog links | Intermediate | n/a | English | Slides: PyTorch Conference 2025 SGLang (2025-10-22), AMD AI Dev Day 2025, AMD SGLang Meetup (2025-08-22: roadmap, large-scale EP, AITER/MoRI), GTC 2025 CUDA tech briefing, cache-aware router, DeepSeek MLA, RLHF with SGLang. Videos: GPU MODE "SGLang Performance Optimization" (2024-11-10), first SGLang online meetup (2024-10-16), biweekly developer syncs (2024-09 → 2025-01) | Best single index of older SGLang talks. **Partly outdated**: its docs link points to the old sgl-project.github.io site | [README](https://github.com/sgl-project/sgl-learning-materials); [ungh](https://ungh.cc/repos/sgl-project/sgl-learning-materials) |
| LMSYS YouTube channel | https://www.youtube.com/@lmsys-org | LMSYS | n/a | Video | All | n/a | English | Meetups and developer syncs | Official video home | [sgl-learning-materials README](https://github.com/sgl-project/sgl-learning-materials) |
| Mini-SGLang (official teaching engine) | https://github.com/sgl-project/mini-sglang | SGLang team (blog by Ziyi Xu) | Last push 2026-05-17; 5,636 stars | Code | Intermediate | Python, PyTorch, CUDA GPU on Linux | English | See Q4 | Official, readable 5k-line distillation of SGLang | [LMSYS blog](https://lmsys.org/blog/2025-12-17-minisgl/) |
| SGLang Summit 2026 talks | https://sglang.io/summit | LMSYS / SGLang | 2026-11-12/13; recordings and slides "available soon" | Conference talks plus 6 hands-on workshops | Intermediate → advanced | n/a | English | More than 90 talks. Workshops covered Miles RL, AMD agentic RL, a "Local AI Coding Lab" (debugging and repairing optimizations in Mini-SGLang with a local coding agent), Nebius, Crusoe K8s deployment, and Modal "first-principles" serving design | Largest SGLang-specific talk corpus once posted | [Summit recap](https://lmsys.org/blog/2026-11-17-sglang-summit-recap/) |
| Roadmap | https://roadmap.sglang.io → GitHub issue #22949 | SGLang team | Live | GitHub issue | Intermediate | n/a | English | Quarterly roadmap | Shows where help is wanted | redirect checked with curl 2026-11-24; [docs.sglang.io](https://docs.sglang.io/) |

- SGLang's Developer Guide pages cover the Contribution Guide, Docker-based development, JIT kernel development, a quantization contribution guide, out-of-tree serve-backend plugins, layer boundaries, benchmark and profiling, the bench-serving guide, evaluating new models and MSProbe debugging. — [llms.txt](https://docs.sglang.io/llms.txt)

### Inferences
- For a learner, the most efficient official path for vLLM is:
  1. the docs Quickstart;
  2. office hours #22 (V1) and #35 (contributing);
  3. the Architecture Overview, Paged Attention, Prefix Caching, CUDA Graphs and torch.compile design docs, then the MRV2 doc;
  4. the blog deep dives.

  The DeepLearning.AI course is a good structured on-ramp for using vLLM, but it is short and usage-oriented rather than about internals.
- SGLang's docs have no single "architecture overview" page comparable to vLLM's (none appeared in the docs index). Its internals are spread across feature-design pages and LMSYS posts, which is why the community walkthroughs in Q3 and Mini-SGLang matter more for SGLang.
- Any SGLang learning material that links to sgl-project.github.io or docs.sglang.ai predates the 2026 docs move. The flags and APIs it shows should be checked against docs.sglang.io, especially after the 1.0 flag removals.

### Gaps
- I could not confirm whether the vLLM Conference 2026 (Ray Summit) and PyTorch Conference 2026 vLLM talks are posted as recordings, or under which URLs.
- SGLang Summit 2026 recordings were not yet published at sglang.io/summit (recap dated 2026-11-17).
- I did not enumerate a current SGLang YouTube playlist of 2025–2026 developer syncs or meetups. sgl-learning-materials stops at Jan 2025.
- I did not open the vllm.ai/events page content (it rendered empty to the fetch tool), so the current list of meetups and SIG meetings is unverified.

## 3. What are the best community deep dives and code walkthroughs of vLLM and SGLang internals?

### Takeaway
The best community deep dives are Aleksa Gordić's "Inside vLLM" (Sep 2025, also on the official vLLM blog, pinned to an Aug 2025 commit) and the Awesome-ML-SYS-Tutorial repo (English and Chinese SGLang and RL-infra code walkthroughs, active through Oct 2026). Newer code-level vLLM walkthroughs worth reading are Infomaniak's "VLLM Ultimate guide part 1" (Apr 2026, v0.18.0) and the "vLLM V1 Understanding" post (Nov 2025). All pre-2026 vLLM walkthroughs predate Model Runner V2, which became the only GPU runner in v0.34.0. Their model-runner sections are therefore outdated, while their scheduler and KV cache concepts still apply.

### Cited Findings

| Name | URL | Author / maintainer | Latest update | Format | Level | Prerequisites | Language | Key topics | Why it stands out | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm (also on aleksagordic.com) | Aleksa Gordić (reviewed by Nick Hill) | 2025-09-05; analysis pinned to vLLM commit 42172ad (2025-08-09) | Long-form blog, "first in a series" | Intermediate → advanced | Python; transformer and attention basics | English | LLM engine and engine core; scheduler; paged attention; continuous batching; prefix caching; speculative decoding; disaggregated P/D; scaling from offline to async, multi-GPU and multi-node; serving layer; benchmarks and auto-tuning (TTFT/ITL/TPOT, roofline) | Widely described as the most in-depth explanation of vLLM V1. Predates MRV2 | [search result for the post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm); [vLLM blog index](https://vllm.ai/blog) |
| zhaochenyang20/Awesome-ML-SYS-Tutorial | https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial | zhaochenyang20 (notes started Aug 2024; thanks Quanquan Gu, Ying Sheng and Lianmin Zheng as advisors) | Last push 2026-10-21; 7,695 stars | Markdown notes, many mirrored on Zhihu | Intermediate → advanced | Python, PyTorch, distributed basics | English and Chinese (README-cn); many linked articles are Chinese-only on Zhihu | SGLang: "SGLang Code Walk Through" (life of a request), "From KV Cache to Zero Overhead Scheduling", KV cache code walkthrough, "When SGLang OOMs, What Exactly Runs Out of Memory?" (mem-fraction-static), zero-overhead batch scheduler, DP attention, constrained decoding, quantization design, online weight update, multimodal request lifecycle, SGLang-Diffusion walkthrough, SGLang-Omni design and TTS serving. Fundamentals: CUDA graph, NCCL/topology, torch.distributed, TP from scratch. RL infra: slime, verl, AReal, OpenRLHF | The deepest public SGLang code walkthroughs. Several entries are marked "Pending Review" or "Not finished" | [README](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial); [ungh](https://ungh.cc/repos/zhaochenyang20/Awesome-ML-SYS-Tutorial) |
| "VLLM Ultimate guide part 1" | https://engineering.infomaniak.com/vllm-ultimate-guide-part-1/ | BLANC Swan (Infomaniak Engineering) | 2026-04-16; analyzes vLLM v0.18.0 | Blog (about 16 min) | Intermediate → advanced | Python; vLLM usage | English | A chat-completion request traced through 14 hops from the FastAPI route to the returned token; ZMQ boundary between API server and EngineCore; busy loop; scheduler token budgeting, admission and preemption; KV-cache memory budgeting and block computation; `GPUModelRunner.execute_model()`; where `--gpu-memory-utilization` and `--max-num-batched-tokens` take effect | The most recent code-level request trace with file and line links. Parts 2 and later were not yet published | [Infomaniak post](https://engineering.infomaniak.com/vllm-ultimate-guide-part-1/) |
| "vLLM V1 Understanding" (Gradients & Tides blog) | https://jianyuh.github.io/vllm/inference/2025/11/30/vLLM.html | Author not stated on page | 2025-11-30; commit 42172ad (2025-08-09) | Blog (about 8 min) | Advanced | vLLM basics | English | Scheduler `allocate_slots` and recompute preemption; free block queue and 16-token blocks; block-hash prefix caching; TP/PP/DP and frontend load balancing; PD disaggregation and the KV connector flow; guided and speculative decoding; roofline modeling and `vllm bench` | Dense structured overview of V1 internals | [post](https://jianyuh.github.io/vllm/inference/2025/11/30/vLLM.html) |
| "Continuous batching from first principles" | https://huggingface.co/blog/continuous_batching | Hugging Face (PR authors ror, ArthurZ, mcpotato) | 2025-11-25 | Blog | Beginner → intermediate | Attention and KV cache basics | English | Derives continuous batching from attention and KV caching, optimizing for throughput; follow-ups on KV cache management and async batching | Best foundational explainer before reading engine code | [HF blog PR](https://github.com/huggingface/blog/pull/3176); [Transformers CB architecture docs](https://huggingface.co/docs/transformers/continuous_batching_architecture) |
| GPU MODE lectures | https://github.com/gpu-mode/lectures | GPU MODE community | Last push 2026-09-09; 7,120 stars | Recorded lectures plus slides | Intermediate → advanced | CUDA/PyTorch basics | English | Lecture 22 "Hacker's Guide to Speculative Decoding in VLLM"; Lecture 35 "SGLang Performance Optimization" (Yineng Zhang, 2024-11-10; video https://www.youtube.com/watch?v=XQylGyG7yp8) | Talks by engine developers; the series also covers foundational GPU topics | [lectures README](https://raw.githubusercontent.com/gpu-mode/lectures/HEAD/README.md); [sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials) |
| Official deep dives (vLLM blog) | https://vllm.ai/blog | vLLM maintainers | 2025–2026 | Blog | Intermediate → advanced | vLLM basics | English | "Introduction to torch.compile and How It Works with vLLM" (2025-08-20); "vLLM Triton Attention Backend Deep Dive" (2026-03-04); "Inside vLLM's New KV Offloading Connector" (2026-01-08); "Model Runner V2" (2026-03-24); "Building Clean, Maintainable vLLM Modifications Using the Plugin System" (2025-11-20); "Keeping vLLM Production Quality: A Look Inside CI, Benchmarking, and the Release Process" (2026-07-16); "CUDA Core Dump: An Effective Tool to Debug Memory Access Issues" (2025-08-11); "Tracing Hanging and Complicated GPU Kernels Down To The Source Code" (2025-12-03); "Structured Decoding in vLLM: a gentle introduction" (2025-01-14); "Distributed Inference with vLLM" (2025-02-17) | Component-level internals from the people who wrote them | [vLLM blog index](https://vllm.ai/blog) |
| Official deep dives (LMSYS blog) | https://lmsys.org/blog/ | SGLang team | 2025–2026 | Blog | Intermediate → advanced | SGLang basics | English | Zero-overhead batch scheduler (v0.4, 2024-12-04); large-scale EP/PD (2025-05-05); HiCache (2025-09-10); deterministic inference (2025-09-22); unified radix cache (2026-08-11); advanced CUDA graph (2026-08-17); agent-assisted SGLang development (2026-07-02) | Design rationale for SGLang subsystems | [LMSYS blog tree via ungh](https://ungh.cc/repos/lm-sys/lm-sys.github.io/files/main); [sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials) |

- The "Inside vLLM" post is "the first in a series", uses an "inverse-pyramid approach" and pins its analysis to commit 42172ad from 2025-08-09. — [search result for vLLM blog mirror](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- MRV2's design doc says V1's GPU model runner had "several fundamental design mistakes and accumulated significant technical debt". As of v0.34.0, "Model Runner V1 is gone". — [MRV2 design doc](https://docs.vllm.ai/en/latest/design/model_runner_v2/); [v0.34.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)
- The search also surfaced a "vLLM Internals" page (jvoltci.github.io/mosaic) and an AI-generated-looking "ollama architecture" article (jonathanding.github.io). I could not assess either for quality, so neither is recommended. — [search results](https://jvoltci.github.io/mosaic/applied/inference-internals/vllm-internals)

### Inferences
- A good reading order for internals:
  1. HF continuous-batching post (concepts).
  2. "Inside vLLM" (whole-system map).
  3. vLLM Architecture Overview and the KV-cache and prefix-caching design docs.
  4. Infomaniak part 1 (code-level request trace on v0.18).
  5. MRV2 design doc (current model runner).
  6. Awesome-ML-SYS-Tutorial's SGLang code walkthrough and scheduler notes, alongside Mini-SGLang, for the SGLang side.
- Learners should pin any walkthrough to its stated commit or version (42172ad, v0.18.0) and diff it against current main, especially for the model runner (MRV2), the default block size (now 64) and new Rust frontend paths.
- Chinese-language learners have far more SGLang/vLLM walkthroughs available, via Zhihu links in Awesome-ML-SYS-Tutorial. English-only learners should rely on the English versions in the repo, which are marked for most core SGLang notes.

### Gaps
- I could not retrieve Aleksa Gordić's blog index, so I could not confirm whether parts 2 and later of the "Inside vLLM" series exist.
- I did not survey Chinese-language deep dives (Zhihu, WeChat) beyond those linked from Awesome-ML-SYS-Tutorial. Another researcher covers Chinese curricula.
- I did not enumerate YouTube code-walkthrough videos beyond office hours and GPU MODE. Quality ratings for less-known blogs are not available.

## 4. Which minimal/educational re-implementations of a serving engine exist, how large are they, what do they implement, and how good are they for learning?

### Takeaway
The three standouts serve different needs:
- **nano-vllm**: about 1,200 lines of Python, offline engine with prefix caching, TP, torch.compile and CUDA graphs; matches vLLM throughput on a small model; 16.5k stars. It is the fastest way to see a vLLM-style engine end to end.
- **Mini-SGLang**: about 5,000 lines, official, with an OpenAI-compatible server, radix cache, chunked prefill, overlap scheduling, TP and FlashAttention/FlashInfer kernels. It is the closest small replica of a production engine's online architecture.
- **tiny-llm**: a four-week, test-driven course on Apple Silicon/MLX whose Week 3 builds a "Mini vLLM" with continuous batching, chunked prefill and paged KV/attention. It is the most pedagogical option.

gpt-fast (under 1,000 lines, last updated Aug 2025) remains a good single-request latency and torch.compile reference, but it is not a serving engine.

### Cited Findings

| Name | URL | Author / maintainer | Size | Latest update / stars | Format | Level | Prerequisites | Language | Features implemented | Strengths for learning / caveats | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nano-vLLM | https://github.com/GeeeekExplorer/nano-vllm | GeeeekExplorer | "~1,200 lines of Python" | Last push 2026-04-26; 16,541 stars; 2,887 forks; no releases | Code | Intermediate | Python, PyTorch, an NVIDIA GPU (benchmarked on an RTX 4070 Laptop 8GB) | English | "Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc."; API mirrors vLLM's `LLM.generate` | Benchmark on Qwen3-0.6B: 1,434 tok/s vs vLLM 1,362 tok/s (256 requests). The README documents only offline inference and no OpenAI-compatible server. Many forks and mirrors exist (e.g., wicky-vllm on PyPI) | [README](https://raw.githubusercontent.com/GeeeekExplorer/nano-vllm/HEAD/README.md); [ungh](https://ungh.cc/repos/GeeeekExplorer/nano-vllm) |
| Mini-SGLang | https://github.com/sgl-project/mini-sglang | SGLang team (announcement by Ziyi Xu, 2025-12-17) | "~5,000 lines of Python"; "fully type-annotated" | Last push 2026-05-17; 5,636 stars; 998 forks | Code | Intermediate → advanced | Linux only (x86_64/aarch64), CUDA toolkit matching the driver (JIT-compiled kernels), Python 3.10+ | English | Radix cache, chunked prefill, overlap scheduling, tensor parallelism, FlashAttention-3 prefill and FlashInfer decode, JIT CUDA kernels via TVM FFI; OpenAI-compatible server (`python -m minisgl`), interactive `--shell`, NVTX annotations | Shares SGLang's high-level process design (API server, tokenizer server, per-GPU scheduler). Claims to outperform Nano-vLLM offline thanks to overlap scheduling, and "nearly identical performance to SGLang" online (Qwen3-32B, TP4, radix cache disabled for the comparison). Also designed as a research prototyping base. No Windows/macOS | [README](https://raw.githubusercontent.com/sgl-project/mini-sglang/HEAD/README.md); [LMSYS blog](https://lmsys.org/blog/2025-12-17-minisgl/) |
| tiny-llm ("LLM serving in a week", now a 4-week course) | https://github.com/skyzh/tiny-llm ; book: https://skyzh.github.io/tiny-llm/ | skyzh ("Chi" in the README audit notes) | Small, "small enough to read end to end" | Last push 2026-11-24; 5,807 stars | Course book plus exercises, tests and reference solution (`tiny_llm_ref`) | Beginner → intermediate (systems engineers) | Apple Silicon Mac; Python (pdm; Python 3.14 pinned); comfort with C++/Metal for kernel chapters | English | Week 1: Qwen3 from `mlx.core` ops (attention, RoPE, GQA, RMSNorm, MLP, sampling, decoding loop). Week 2: KV cache, packed W4 weights, quantized matvec, SIMD prefill, fused kernels, tiled attention (marked 🚧). Week 3 "Build a Mini vLLM": continuous batching, chunked prefill, paged KV cache, direct paged attention, paged FlashAttention, optional MoE and speculative decoding. Week 4: build a coding agent (Days 1–9 published) | Most pedagogical: exercises with tests and MLX as correctness oracle, on cheap local hardware ("instead of depending on an expensive CUDA GPU setup"). Caveat: Metal/MLX rather than CUDA, and Week 2 is still in progress | [README](https://raw.githubusercontent.com/skyzh/tiny-llm/HEAD/README.md); [ungh](https://ungh.cc/repos/skyzh/tiny-llm) |
| gpt-fast | https://github.com/meta-pytorch/gpt-fast (moved from pytorch-labs) | PyTorch team (Meta) | "<1000 lines of python" | Last push 2025-08-22; about 6.3k stars | Code plus walkthrough blog | Intermediate | PyTorch (README says to install PyTorch nightly), NVIDIA or AMD GPU | English | Very low latency at batch size 1; int8/int4 quantization; speculative decoding; tensor parallelism; Mixtral MoE | Shows what native PyTorch plus torch.compile can do. Explicitly "NOT intended to be a framework". Not a serving engine (no batching scheduler). **Low activity since Aug 2025** | [README](https://raw.githubusercontent.com/meta-pytorch/gpt-fast/HEAD/README.md); [walkthrough blog](https://pytorch.org/blog/accelerating-generative-ai-2/); [ungh](https://ungh.cc/repos/meta-pytorch/gpt-fast) |
| paged-attention-minimal | https://github.com/tspeterkim/paged-attention-minimal | tspeterkim | Minimal | Last push 2024-08-26; 151 stars | Code | Intermediate | PyTorch | English | "a minimal cache manager for PagedAttention, on top of llama3" | Focused on just the block manager. **Unmaintained since 2024** | [ungh](https://ungh.cc/repos/tspeterkim/paged-attention-minimal) |
| ingine | https://github.com/ashishsubedi/ingine | ashishsubedi | n/a | n/a | Code | Beginner → intermediate | Apple Silicon | English | "Pedagogical implementation of continuous batching" with PagedAttention and concurrent request serving; about 1.29× throughput vs sequential paged decoding on Qwen2.5-1.5B | Small and recent-looking, but unvetted. Stars and activity not verified | [search result](https://github.com/ashishsubedi/ingine) |
| liteinfer, smol-vllm (PyPI) | https://pypi.org/project/liteinfer/ ; https://pypi.org/project/smol-vllm/ | ValeGian (liteinfer); unknown | n/a | n/a | Packages | Intermediate | n/a | English | liteinfer: paged KV cache, continuous batching via AsyncLLM, prefix caching on the roadmap. smol-vllm: paged KV, continuous batching, preemption ("educational") | **Unvetted**: quality, size and maintenance not verified | [liteinfer](https://pypi.org/project/liteinfer/); [smol-vllm](https://pypi.org/project/smol-vllm/) |
| llama2.c | https://github.com/karpathy/llama2.c | Andrej Karpathy | "one file of pure C" | Last push 2024-08-06; 20,308 stars | Code | Beginner | C | English | Single-request Llama-2 inference | Foundation for the forward pass and KV cache. Not a serving engine. **Unmaintained since 2024** | [ungh](https://ungh.cc/repos/karpathy/llama2.c) |
| calm | https://github.com/zeux/calm | zeux | Small | Last push 2025-05-29; 643 stars | Code | Advanced | CUDA/Metal | English | "CUDA/Metal accelerated language model inference" | Kernel-level single-stream inference. Low activity | [ungh](https://ungh.cc/repos/zeux/calm) |
| yalm | https://github.com/andrewkchan/yalm | andrewkchan | Small | Last push 2025-09-13; 609 stars | Code | Advanced | C++/CUDA | English | "LLM inference in C++/CUDA, no libraries except for I/O" | Kernel-level single-stream inference. Low activity | [ungh](https://ungh.cc/repos/andrewkchan/yalm) |

- Mini-SGLang exists because SGLang "has grown to nearly 300k lines of Python". It aims to keep "the advanced features that define state-of-the-art performance" in "only 5k lines". — [LMSYS: Mini-SGLang](https://lmsys.org/blog/2025-12-17-minisgl/)
- tiny-llm describes itself as "an LLM-serving counterpart to CMU's Needle project". It notes that Qwen3-4B "is large enough to expose real weight-bandwidth, attention, and cache costs, but small enough to iterate on locally". — [tiny-llm README](https://raw.githubusercontent.com/skyzh/tiny-llm/HEAD/README.md)
- The SGLang Summit 2026 ran a "Local AI Coding Lab" workshop on "debugging and repairing optimizations in Mini-SGLang using a local coding agent". This shows Mini-SGLang being used as an official teaching vehicle. — [Summit recap](https://lmsys.org/blog/2026-11-17-sglang-summit-recap/)
- A web search for nano-vllm alternatives mostly returned forks and mirrors of nano-vllm (coderonion, HIT-Felix, wicky-vllm, nano-vllm-fork) rather than independent engines. — [search results incl. HIT-Felix fork](https://github.com/HIT-Felix/Nano-vllm); [wicky-vllm](https://pypi.org/project/wicky-vllm/0.1.1/)

### Inferences
- **Suggested progression.**
  1. tiny-llm Weeks 1 and 3 if the learner has a Mac, or nano-vllm if they have an NVIDIA GPU. Both are small enough to read in a weekend.
  2. Mini-SGLang next, which adds the online-serving process architecture, radix cache and overlap scheduling.
  3. Then the corresponding files in vLLM and SGLang, guided by the walkthroughs in Q3.
- **Hands-on extensions that mirror real engine work.**
  - Port a feature from the real engine (e.g., chunked prefill or prefix caching) into nano-vllm.
  - Add an OpenAI-compatible server to nano-vllm.
  - Reproduce the nano-vllm vs vLLM and Mini-SGLang vs SGLang benchmarks.
  - Ablate overlap scheduling with `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`.
- **Engine vs kernel references.** nano-vllm and Mini-SGLang are "real" engines with production techniques. llama2.c, calm, yalm and gpt-fast teach the forward pass, kernels and single-stream latency, which are foundational but do not cover scheduling and memory management under concurrency.

### Gaps
- Line counts are the projects' own README claims (about 1,200 and about 5,000), not measured. Exact feature completeness, such as whether nano-vllm's scheduler handles preemption or chunked prefill, was not verified from source.
- I found no verified independent minimal SGLang-like or vLLM-like engines with substantial adoption beyond nano-vllm and Mini-SGLang. The smaller projects (ingine, liteinfer, smol-vllm) are unvetted.
- I found no maintained minimal re-implementation of TensorRT-LLM or llama.cpp-style serving for learning.

## 5. How does a newcomer start using and contributing to vLLM/SGLang?

### Takeaway
Both projects have a low-friction start:
- **Install and serve.** vLLM: `uv pip install vllm --torch-backend=auto` then `vllm serve <model>`. SGLang: `sglang serve --model-path <model>`.
- **Contributor guides.** vLLM's covers a Python-only dev install via `VLLM_USE_PRECOMPILED=1`, DCO sign-off and a "job board". SGLang's covers fork-and-PR, pre-commit, unit tests that mirror `python/sglang/srt/`, and docs as a stated way "to learn the SGLang codebase".
- **Good first issues.** About 14 are open on vLLM and 46 on SGLang.
- **Community.** Large Slack workspaces (vLLM 10K+ members, organized into SIGs), vLLM's forum and biweekly office hours, and SGLang's roadmap issue.

Both repos increasingly use coding agents (SGLang: "Agents now produce most merged PRs"). This raises the bar for evidence-backed, well-scoped human contributions.

### Cited Findings
- **vLLM quickstart.** `uv pip install vllm --torch-backend=auto`, then for example `vllm serve deepseek-ai/DeepSeek-V3.1 -tp 8`. — [vLLM deck PDF](https://hosted-files.sched.co/pytorchconferenceeu2026/3e/2.%20Tyler%20Michael%20Smith.pdf); docs quickstart at [docs.vllm.ai quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- **vLLM contributing guide.**
  - **Job Board.** Lists "Good first issues", "Selected onboarding tasks", "New model requests" and "Models with multi-modal capabilities".
  - **Dev setup.** Recommends `uv venv --python 3.12` and, for Python-only work, `VLLM_USE_PRECOMPILED=1 uv pip install -e .` plus `vllm download-kernels`. C++/CUDA work requires a full build, with an "Incremental Compilation Workflow" page for faster iteration. vLLM supports Python 3.11–3.14, but CI uses 3.12.
  - **Commits.** Must carry a DCO `Signed-off-by:` line (`git commit -s`).
  - **Beyond code.** Answering questions, reviewing PRs and documentation count as contributions. — [vLLM contributing guide](https://docs.vllm.ai/en/latest/contributing/)
- **Model-contribution path.** vLLM has dedicated pages for adding a model ("Basic Model", "Registering a Model", "Unit Testing", "Multi-Modal Support", "Speech-to-Text") and for "Profiling vLLM", "Labels", "Deprecation Policy", "Releasing vLLM" and "Editing Agent Instructions". — [vLLM docs search index (contributing/)](https://docs.vllm.ai/en/latest/search/search_index.json); e.g. [basic model page](https://docs.vllm.ai/en/latest/contributing/model/basic/)
- **Office hours on contributing.** Session #35 "How to Build and Contribute to vLLM" (2025-10-23) is recorded. Office hours run every other Thursday at 2:00 PM ET (calendar at red.ht/office-hours). — [vLLM office hours](https://vllm.ai/office-hours)
- **Community channels.**
  - slack.vllm.ai redirects to an invite (HTTP 302 → inviter.co/vllm-slack), and discuss.vllm.ai returns HTTP 200 (checked with curl, 2026-11-24). — [vLLM website post (footer links)](https://vllm.ai/blog/2025-12-27-vllm-ai-website)
  - The Slack has "10K+ members"; newcomers should "find your SIG". Weekly SIG meetings are listed at vllm.ai/events, and the roadmap is at roadmap.vllm.ai. The Q2 2026 SIGs were core, CI, quantization, large-scale serving, model performance, torch.compile, multimodality and omni. — [vLLM deck PDF](https://hosted-files.sched.co/pytorchconferenceeu2026/3e/2.%20Tyler%20Michael%20Smith.pdf)
  - talentpool@vllm.ai forwards resumes to partner companies. — [vLLM website post](https://vllm.ai/blog/2025-12-27-vllm-ai-website)
- **Open good-first-issues (2026-11-24).** vLLM had 14, SGLang 46, llama.cpp 14 and Ollama 3. TensorRT-LLM, LMDeploy, MLC-LLM and LightLLM had 0 under that label name. — [shields vLLM](https://img.shields.io/github/issues/vllm-project/vllm/good%20first%20issue.json); [shields SGLang](https://img.shields.io/github/issues/sgl-project/sglang/good%20first%20issue.json)
- **SGLang quickstart and docs.** The docs start at Get Started → Quickstart ("Start a model server and verify it with one request") and Installation (pip/uv, source, Docker, Kubernetes, cloud). — [SGLang llms.txt](https://docs.sglang.io/llms.txt). The CLI form is `sglang serve --model-path Qwen/Qwen3-8B`. — [SGLang contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide.md)
- **SGLang contribution guide.**
  - **Fork first.** "New contributors do not have the write permission", so fork, branch and open a PR.
  - **Formatting.** Run `pre-commit run --all-files`.
  - **Tests.** Unit tests live under `test/registered/unit/` and mirror `python/sglang/srt/`, e.g. `srt/mem_cache/radix_cache.py → unit/mem_cache/test_radix_cache_unit.py`. CI suites run via `test/run_suite.py`.
  - **Docs.** "Documentation is a good way for new contributors to learn the SGLang codebase".
  - **Accuracy check.** Run a GSM8K sanity check with `sgl-eval run gsm8k`, warned to have 1–5% variance. Speed is measured via the Benchmark and Profiling guide. — [SGLang contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide.md)
- **SGLang community links.** slack.sglang.io redirects to a Slack invite (302), and roadmap.sglang.io redirects to GitHub issue #22949 (checked with curl, 2026-11-24). The learning-materials README still advertises slack.sglang.ai. — [docs.sglang.io](https://docs.sglang.io/); [sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)
- **Agent-assisted development (SGLang).**
  - SGLang's repo ships `.claude/skills` for CUDA crash debugging, kernel integration, tests, CI, profiling and production triage.
  - Agents "can create more PRs, and they can also create more plausible mistakes". The goal is to "hand repetitive execution to agents, and leave judgment, design, and review to people".
  - Recommended practice for performance work: define a narrow task with fixed workload and hardware, fix the benchmark before profiling, and check for silent backend fallbacks. — [LMSYS: agent-assisted SGLang development](https://lmsys.org/blog/2026-07-02-agent-assisted-sglang-development/)
  - At the Summit the team said "Agents now produce most merged PRs". — [Summit recap](https://lmsys.org/blog/2026-11-17-sglang-summit-recap/)
- **Benchmarking tools for hands-on projects.**
  - vLLM: the `vllm bench` CLI (serve, throughput, latency, sweep) and a performance dashboard. — [vLLM docs search index (cli/, benchmarking/)](https://docs.vllm.ai/en/latest/search/search_index.json)
  - GuideLLM (v0.8.1) for realistic load testing, which is also taught in the DLAI course. — [ungh guidellm](https://ungh.cc/repos/vllm-project/guidellm); [vLLM course post](https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course)
  - SGLang: genai-bench (v0.0.5) and the Bench Serving Guide. — [ungh genai-bench](https://ungh.cc/repos/sgl-project/genai-bench); [SGLang llms.txt](https://docs.sglang.io/llms.txt)
- **Contribution volume.** vLLM's v0.34.0 had 120 first-time contributors out of 412, which shows a steady newcomer intake. — [vLLM v0.34.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)

### Inferences
- **A practical newcomer sequence.**
  1. Rent one GPU (or use a Mac with tiny-llm). Run the vLLM and SGLang quickstarts against the same small model (e.g., Qwen3-0.6B or 8B) and query both through the OpenAI-compatible API.
  2. Benchmark both with `vllm bench serve` or GuideLLM, varying `--max-num-batched-tokens`, the block size and chunked prefill. Explain the results using the design docs.
  3. Read nano-vllm, then Mini-SGLang, then the walkthroughs in Q3.
  4. Join Slack, pick a SIG, attend office hours and watch #35.
  5. Start with a docs fix, a good-first-issue, a unit test that mirrors an `srt/` module (SGLang), or a "new model request" (vLLM), each backed by accuracy and benchmark evidence.
- Because both projects now absorb many agent-generated PRs, a human newcomer stands out through careful scoping, reproducible benchmarks, accuracy checks and good reviews, not raw PR count.
- For a self-learner targeting AI-infra jobs, visible contributions (merged PRs, benchmark write-ups, office-hours talks) plus vLLM's talent-pool email are concrete career levers. Both projects' companies (Inferact, RadixArk) and consortium/partner companies hire from these communities. The hiring claim is an inference, not directly sourced.

### Gaps
- I did not verify the exact current content of vLLM's "Selected onboarding tasks" list or of SGLang's good-first-issue backlog (which issues, what difficulty).
- I did not quantify how quickly newcomer PRs get reviewed or merged in either project; no data was found.
- I did not find a public, structured "mentorship" or "contributor onboarding program" for either project beyond docs, office hours, Slack and SIGs. LMSYS blog posts about a PhD fellowship exist (2025-11-23, 2026-06-08), but I did not open them.
