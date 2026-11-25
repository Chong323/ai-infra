# Chinese-language AI infra courses, community resources, learning roadmaps, curated lists, communities and role skills (state as of 2026-11-24)

Data provenance (applies to every GitHub figure below): stars, forks, "created" and "pushed" dates were pulled on 2026-11-24 from the ungh.cc GitHub-API cache (e.g. https://ungh.cc/repos/Infrasys-AI/AISystem). "Last commit" means the last commit on the default branch, taken from shields.io last-commit badges (e.g. https://img.shields.io/github/last-commit/chenzomi12/AISystem.json). Shields leaves the year off dates in the current year, so "Oct" means Oct 2026. "Pushed" counts pushes to any branch, so it can be later than the last commit. This environment blocked github.com and api.github.com, so READMEs were read from raw.githubusercontent.com and the figures come from these mirrors, which can lag by a few hours. Star counts are approximate.

Status labels used: **Active** = default-branch commit in Sep–Nov 2026. **Maintained** = commit in Jan–Aug 2026. **Slowing** = last commit in 2025. **Stale** = last commit in 2024 or earlier. "Level" and "prerequisites" are my own assessment unless a source is cited.

## Q1. What are the most important Chinese-language open-source AI infra / AI systems courses and books, what do they cover, and how current are they?

### Takeaway
The leading Chinese course-style resources split into two groups.

The first covers the classic AI-systems stack:
- ZOMI's **AISystem**: chips, compilers, inference engines and frameworks.
- **MSRA AI-System**.
- **MLC (Tianqi Chen)**.
- **d2l-zh**.

These are good foundations, but they predate LLM serving and are slowing or stale.

The second covers the LLM era:
- ZOMI's **AIInfra**: clusters, communication, cloud-native, distributed training and distributed inference, with notebooks.
- **openmlsys V2**: rewritten in 2026, it adds GPU/CUDA/Triton, model serving, RL systems and GPU-cluster chapters.
- **Datawhale self-llm**: the most active Chinese hands-on deployment cookbook. It covers vLLM, Ollama and llama.cpp, plus Ascend, AMD, MUSA and MetaX hardware.

No Chinese course teaches vLLM or SGLang internals in depth. The AIInfra course and the openmlsys serving chapter teach the concepts: KV cache, PagedAttention, continuous batching, PD disaggregation and speculative decoding.

### Cited Findings

#### 1. AISystem (ZOMI 酱《AI 系统》, "AI System & AI Infra")
- **URL and maintainer:** https://github.com/Infrasys-AI/AISystem. The old URL chenzomi12/AISystem now resolves to the Infrasys-AI org. It is maintained by ZOMI (ZOMI 酱). — [ungh](https://ungh.cc/repos/chenzomi12/AISystem); [README](https://github.com/Infrasys-AI/AISystem)
- **Repo stats:** about 18.1k stars and 2.5k forks. Created 2022-09-22, last commit Sep 2025 (pushed 2025-09-03). Status: **Slowing**. — [ungh](https://ungh.cc/repos/Infrasys-AI/AISystem); [shields](https://img.shields.io/github/last-commit/chenzomi12/AISystem.json)
- **Format:** text course at https://infrasys-ai.github.io/aisystem-docs/ (HTTP 200 on 2026-11-24, title "课程内容大纲 — AI System"), slides, and a video series on Bilibili (space.bilibili.com/517221395) and YouTube. Language: Chinese. — [README](https://github.com/Infrasys-AI/AISystem); [docs site](https://infrasys-ai.github.io/aisystem-docs/)
- **Structure:** five parts: 一. AI 系统概述; 二. AI 芯片体系结构; 三. AI 编译原理; 四. AI 推理系统; 五. AI 框架核心技术. The README says the text course is "正在一节节补充更新" (still being filled in section by section). — [README](https://github.com/Infrasys-AI/AISystem)
- **Inference part:** its folders are 01Inference, 02Mobilenet, 03Slim, 04Converter, 05Optimize and 06Kernel. That is the classic edge/mobile inference-engine pipeline (lightweight models, model slimming, conversion, graph optimization, kernels), not LLM serving. — [file tree via ungh](https://ungh.cc/repos/Infrasys-AI/AISystem/files/main)
- **Level (assessed):** beginner to intermediate. Prerequisites: basic deep learning and computer-architecture literacy.
- **Why it stands out:** the most-starred Chinese AI-systems course, covering the whole stack from chips to frameworks in video form.

#### 2. AIInfra (ZOMI 酱《大模型系统 / AI 基础设施》)
- **URL:** https://github.com/Infrasys-AI/AIInfra. The old URL chenzomi12/AIInfra resolves here. — [ungh](https://ungh.cc/repos/chenzomi12/AIInfra)
- **Repo stats:** about 8.6k stars and 1.1k forks. Created 2024-07-17, last commit Dec 2025 (pushed 2025-12-22). Status: **Slowing**. — [ungh](https://ungh.cc/repos/Infrasys-AI/AIInfra); [shields](https://img.shields.io/github/last-commit/Infrasys-AI/AIInfra.json)
- **Format:**
  - Slides (PPT/PDF), Markdown articles and Jupyter notebooks.
  - Videos on Bilibili ("ZOMI 酱") and YouTube ("ZOMI6222").
  - A web version at https://infrasys-ai.github.io/aiinfra-docs/ (HTTP 200).
  - The repo is about 10 GB, so the README recommends downloading from Releases rather than cloning.

  — [GitHub page](https://github.com/Infrasys-AI/AIInfra); [docs site](https://infrasys-ai.github.io/aiinfra-docs/)
- **Eight modules:**
  - 00 overview (scaling laws, training and inference stacks).
  - 01 AI compute clusters.
  - 02 communication and storage (MPI/NCCL/HCCL, checkpointing).
  - 03 containers and cloud-native (Docker, Kubernetes, scheduling).
  - 04 distributed training (Megatron-LM, DeepSpeed/ZeRO, pipeline parallelism, FP8, FlashAttention, RL post-training with PPO/DPO/GRPO).
  - 05 distributed inference.
  - 06 algorithms and data (MHA/MQA/GQA/MLA, MoE, multimodal).
  - 07 applications.

  Several sub-sections are still marked ⭕ (in progress). — [GitHub page](https://github.com/Infrasys-AI/AIInfra)
- **The 05Infer module:**
  - Inference-engine overview and performance.
  - KV cache, with code.
  - PagedAttention, with code.
  - LLM in a Flash, StreamingLLM, KTransformers and FlashMLA-FP8.
  - Continuous batching (with code), FCFS scheduling and PD dispatch (with code).
  - Long-sequence inference: RingAttention, Ulysses, PD 分离, FastGen, FlexPrefill, StarAttention.
  - Sampling and speculative decoding, with code.
  - Quantization and distillation.

  No file in the repo is named after vLLM, SGLang, LMDeploy or TensorRT, and the README does not name vLLM or SGLang. — [file tree via ungh](https://ungh.cc/repos/Infrasys-AI/AIInfra/files/main); [GitHub page](https://github.com/Infrasys-AI/AIInfra)
- **Level (assessed):** intermediate. Prerequisites: deep learning, Transformer and Python basics.
- **Why it stands out:** the only large Chinese course that runs end to end through LLM-era clusters, communication, cloud-native, training and inference, with runnable notebooks.

#### 3. 人工智能系统 (Microsoft Research Asia, microsoft/AI-System)
- **URL:** https://github.com/microsoft/AI-System
- **Repo stats:** about 4.4k stars and 541 forks. Created 2020-09, last commit Oct 2024. Status: **Stale**. — [ungh](https://ungh.cc/repos/microsoft/AI-System); [shields](https://img.shields.io/github/last-commit/microsoft/AI-System.json)
- **Format and coverage:** lecture slides (Lectures), labs (Labs) and a textbook section (人工智能系统教材), in Chinese. The syllabus's inference lecture (No. 8) covers efficiency, latency, throughput and deployment, using TensorRT, TensorFlow Lite and ONNX as the example systems. — [README](https://github.com/microsoft/AI-System)
- **Level (assessed):** intermediate, at university-course depth. Prerequisites: OS/architecture and deep-learning basics.
- **Verdict:** a solid foundation in framework, compiler and scheduling theory, but pre-LLM and unmaintained.

#### 4. 机器学习系统：设计和实现 (openmlsys)
- **URL:** https://github.com/openmlsys/openmlsys. The old openmlsys-zh URL resolves here.
- **Repo stats:** about 4.9k stars and 475 forks. Last commit Oct 2026 (pushed 2026-10-26). The repo description reads "(V2 is launching soon)". Status: **Active**. — [ungh](https://ungh.cc/repos/openmlsys/openmlsys-zh); [shields](https://img.shields.io/github/last-commit/openmlsys/openmlsys-zh.json)
- **Site:** https://openmlsys.github.io/ returned HTTP 200 with the title "Machine Learning Systems: Design and Implementation, 2nd Edition". — [site](https://openmlsys.github.io/)
- **Changelog:**
  - 2022-01: project started.
  - 2022-05: extended chapters added (federated learning, RL, explainable AI).
  - 2023-05: adapted to MindSpore 2.0.
  - 2026-03: bilingual Chinese/English build restructured and the English version launched.

  — [README](https://github.com/openmlsys/openmlsys)
- **V2 chapters:**
  1. 导论
  2. 编程接口与计算图
  3. AI 加速器与编程 (GPU architecture and the CUDA/Triton/CUTLASS programming models)
  4. AI 编译器与运行时
  5. 数据处理
  6. 训练系统
  7. 模型服务 (推理优化、在线服务与模型管理)
  8. 强化学习系统
  9. 大规模 GPU 集群管理

  — [README](https://github.com/openmlsys/openmlsys)
- **Companion repos:** openmlsys-cuda has 134 stars and its last commit was Aug 2023 (**Stale**). openmlsys-en has 40 stars and its last commit was Jan 2025. — [ungh](https://ungh.cc/repos/openmlsys/openmlsys-cuda); [ungh](https://ungh.cc/repos/openmlsys/openmlsys-en)
- **Level (assessed):** intermediate, textbook style. Prerequisites: Python and deep-learning basics.
- **Why it stands out:** a textbook-quality Chinese book that is being refreshed for the LLM/RL era in 2026.

#### 5. MLC 机器学习编译 Chinese course (陈天奇 / Tianqi Chen)
- **Notes repo:** https://github.com/mlc-ai/mlc-zh. About 641 stars, created 2022-06, last commit Apr 2026 (status **Maintained**, but the content dates from 2022). The English version is mlc-ai/mlc-en (465 stars). — [ungh](https://ungh.cc/repos/mlc-ai/mlc-zh); [ungh](https://ungh.cc/repos/mlc-ai/mlc-en)
- **Course page:** https://mlc.ai/summer22-zh/ is live (HTTP 200). https://mlc.ai/zh/ and https://mlc.ai/zh/index.html returned **404** on 2026-11-24, so links to the Chinese notes site may be broken. — [course page](https://mlc.ai/summer22-zh/)
- **What the course page says:**
  - Lecturer: 陈天奇. TA: 冯思远 (SJTU).
  - Prerequisites: familiarity with Python and NumPy, some background in a deep-learning framework (PyTorch, TensorFlow or JAX), and systems programming experience (C/CUDA) as a plus.
  - Topics: representing ML programs, automatic optimization, and compilation and runtime for end-to-end deployment.
  - The Chinese forum is GitHub Discussions on mlc-zh.

  — [course page](https://mlc.ai/summer22-zh/)
- **Verdict:** a dated summer-2022 compiler course from the TVM era. Valuable for understanding compilers, not for LLM serving.

#### 6. 动手学深度学习 (d2l-zh)
- **URL and stats:** https://github.com/d2l-ai/d2l-zh. About 82.7k stars. Last commit on the default branch was Aug 2023 (pushed Jul 2024). Status: **Stale**, but the material is evergreen. — [ungh](https://ungh.cc/repos/d2l-ai/d2l-zh); [shields](https://img.shields.io/github/last-commit/d2l-ai/d2l-zh.json)
- **Site:** https://zh.d2l.ai/ is live (2.0.0 docs). The repo describes itself as "面向中文读者、能运行、可讨论。中英文版被70多个国家的500多所大学用于教学". — [site](https://zh.d2l.ai/); [ungh](https://ungh.cc/repos/d2l-ai/d2l-zh)
- **Level:** beginner. It is the standard Chinese prerequisite for deep learning.

#### 7. Datawhale projects (datawhalechina)
- **llm-deploy (大模型/LLM推理和部署理论与实践)**
  - URL: https://github.com/datawhalechina/llm-deploy. About 435 stars, created 2024-01, last commit Jul 2025. Status: **Slowing**. — [ungh](https://ungh.cc/repos/datawhalechina/llm-deploy)
  - Audience: "算法工程师" and "对推理部署感兴趣的同学".
  - Ten chapters, each with an owner: 量化, 蒸馏, 剪枝, 低秩分解, 表示, 运行, 框架, 并行, 并发, 内存. Some chapters have Tencent Meeting video recordings.

  — [README](https://github.com/datawhalechina/llm-deploy)
- **self-llm《开源大模型食用指南》**
  - URL: https://github.com/datawhalechina/self-llm. About 32.8k stars, last commit Nov 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/datawhalechina/self-llm)
  - Content: tutorials for fine-tuning (full-parameter and LoRA) and deploying many open LLMs and MLLMs on Linux.
  - Hardware zones:
    - AMD GPU: a ROCm setup plus LM Studio, vLLM, Ollama and llama.cpp deployment for each model.
    - Ascend NPU: deployment with vLLM-ascend and sglang-ascend.
    - Moore Threads MUSA, MetaX (沐曦) and Apple M.

  — [README](https://github.com/datawhalechina/self-llm)
- **happy-llm《从零开始构建大模型》**
  - URL: https://github.com/datawhalechina/happy-llm. About 35.1k stars, last commit Aug 2026. Status: **Maintained**. The site datawhalechina.github.io/happy-llm/ is live. — [ungh](https://ungh.cc/repos/datawhalechina/happy-llm)
  - Content: PDF and PPT downloads. Chapter 8 covers GRPO, OPD, Search-R1 and ReTool. Extra chapters include "S1: Thinking Budget with vLLM" (2025-08-03) and "大模型生成 Token 的方式有哪些？" (2025-10-17).

  — [README](https://github.com/datawhalechina/happy-llm)
- **so-large-lm《大模型基础》**
  - URL: https://github.com/datawhalechina/so-large-lm. About 7.8k stars, last commit Jun 2026. Status: **Maintained**. — [ungh](https://ungh.cc/repos/datawhalechina/so-large-lm)
  - Content: a learning path and a three-part syllabus (基础与架构; 数据与训练; 安全、伦理与前沿), with links to Bilibili videos.

  — [README](https://github.com/datawhalechina/so-large-lm)
- **tiny-universe《大模型白盒子构建指南》**
  - URL: https://github.com/datawhalechina/tiny-universe. About 5.1k stars, last commit Feb 2026. Status: **Maintained**. — [ungh](https://ungh.cc/repos/datawhalechina/tiny-universe)
  - Content: hand-built TinyLLM, TinyTransformer, TinyRAG, TinyAgent, TinyEval, TinyDiffusion, TinyGraphRAG (released 2025-08-12) and Qwen-Blog, with Tencent Meeting video recordings.

  — [README](https://github.com/datawhalechina/tiny-universe)
- **Related Datawhale repos:**
  - handy-ollama ("CPU玩转大模型部署"): about 2.6k stars, last commit Jan 2026.
  - llms-from-scratch-cn: about 4.5k stars, last commit Sep 2026.
  - thorough-pytorch (PyTorch 入门): about 3.9k stars, last commit Oct 2025.

  — [ungh](https://ungh.cc/repos/datawhalechina/handy-ollama); [ungh](https://ungh.cc/repos/datawhalechina/llms-from-scratch-cn); [ungh](https://ungh.cc/repos/datawhalechina/thorough-pytorch)

#### 8. 书生大模型实战营 (InternLM/Tutorial)
- **URL and stats:** https://github.com/InternLM/Tutorial. About 2.0k stars. Last commit on the default branch was Nov 2024, although other branches were pushed in Apr 2026. — [ungh](https://ungh.cc/repos/InternLM/Tutorial); [shields](https://img.shields.io/github/last-commit/InternLM/Tutorial.json)
- **Structure (third camp):**
  - L0: Linux, Python and Git, each with a Bilibili video.
  - L1: demo, prompting, LlamaIndex RAG, XTuner fine-tuning and OpenCompass evaluation.
  - L2: includes "LMDeploy 量化部署进阶实践" (with a Bilibili video), InternVL deployment, and Android on-device deployment as an easter egg.
  - Completing tasks earns compute credits (算力点).

  — [README](https://github.com/InternLM/Tutorial)
- **Level:** beginner. It is free, gamified and comes with compute credits.

#### 9. Other Chinese course-like repos noted
- **modelscope/modelscope-classroom:** about 1.5k stars, last commit Apr 2026. The README was not inspected. — [ungh](https://ungh.cc/repos/modelscope/modelscope-classroom)
- **mli/paper-reading (李沐 "深度学习经典、新论文逐段精读"):** about 34.1k stars, last commit Mar 2025 (**Slowing**). — [ungh](https://ungh.cc/repos/mli/paper-reading)

### Inferences
- **Recommended use for a deployment-focused learner:**
  - d2l-zh or happy-llm for model foundations.
  - AIInfra module 05 plus openmlsys V2 chapter 7 for serving concepts.
  - self-llm for hands-on deployment.
  - AISystem and MSRA AI-System only for background on chips, compilers and frameworks.
- **Currency gap:** every big Chinese course is either pre-LLM (MSRA, MLC, AISystem's inference part) or slowing (AISystem since Sep 2025, AIInfra since Dec 2025). openmlsys V2 and self-llm are the only course-scale projects with commits in Oct–Nov 2026.
- **The 2026 hardware angle is distinctly Chinese:** self-llm and AIInfra cover Ascend, MUSA, MetaX and AMD. That is useful for learners who will deploy on domestic accelerators, and the English resources rarely cover it.
- **Scope of llm-deploy:** it is better described as a compression-and-deployment theory primer than as a vLLM/SGLang course, and its activity is low.

### Gaps
- Bilibili and YouTube channel statistics (followers, last upload date, number of videos in ZOMI's series) were not verified, because Bilibili pages were not fetched.
- No release date was found for openmlsys V2 ("launching soon"). The authorship and affiliations of openmlsys were not re-verified here.
- Publication details of the MSRA textbook were not verified.
- It is unresolved whether the Chinese MLC notes moved from mlc.ai/zh (404) to another URL.
- The depth of llm-deploy's chapters was not inspected beyond the README.

## Q2. Which Chinese community resources are most valued for learning CUDA and LLM inference/serving, and which are actively maintained in 2026?

### Takeaway
The most valuable actively maintained Chinese-community resources (Oct–Nov 2026) are:
- **LeetCUDA** (xlite-dev/DefTruth): 200+ graded CUDA, Triton and CuTe kernels, a PDF book updated 2026-11-20, and an index of Chinese blog posts.
- **Awesome-LLM-Inference**.
- **zhaochenyang20/Awesome-ML-SYS-Tutorial**: bilingual SGLang and RLHF-system notes from an SGLang core contributor.
- **BBuf/how-to-optim-algorithm-in-cuda**: now moving towards English entry points.
- **liguodongiot/llm-action**.
- **ForceInjection/AI-fundamentals**: updated daily, and covers vLLM, SGLang, llm-d and Mooncake.

Mini-engines are the favourite hands-on route. They include the Chinese-authored KuiperLLama and rapid_llm, and the widely studied nano-vllm and mini-sglang.

Zhihu is the main place for long-form posts, but Zhihu pages could not be fetched (HTTP 403), so author lists come from curated GitHub indexes. Several classic Chinese CUDA tutorials are stale.

China-based SGLang meetups were frequent in 2026 (Beijing, Shanghai, Hangzhou, Shenzhen).

### Cited Findings

#### CUDA and GPU kernels
- **xlite-dev/LeetCUDA (formerly DefTruth/CUDA-Learn-Notes)**
  - URL: https://github.com/xlite-dev/LeetCUDA
  - Stats: about 12.2k stars, last commit 2026-11-24. Status: **Active**. — [ungh](https://ungh.cc/repos/xlite-dev/LeetCUDA); [shields](https://img.shields.io/github/last-commit/xlite-dev/LeetCUDA.json)
  - Self-description: "An Open-source book with Modern CUDA learning Notes for Beginners - BF16/FP8/FP4, HGEMM, CuTe, Flash-Attention". — [ungh](https://ungh.cc/repos/xlite-dev/LeetCUDA)
  - Content: 200+ CUDA kernels graded Easy, Medium, Hard, Hard+ and Hard++; Triton kernels; CUTLASS/CuTe kernels; HGEMM and FA2-MMA benchmarks; Blackwell setup notes (RTX 5090, CUDA ≥ 13.2); a PDF release (leetcuda-20261120.pdf, v4.2.2); and a "100+ 高性能计算与分布式-技术博客" index of Chinese Zhihu posts. — [README](https://github.com/xlite-dev/LeetCUDA)
  - Language: bilingual (English code and headings, Chinese blog index).
  - Level (assessed): beginner to advanced. Prerequisites: C++ and basic CUDA.
- **BBuf/how-to-optim-algorithm-in-cuda**
  - URL: https://github.com/BBuf/how-to-optim-algorithm-in-cuda
  - Stats: about 3.4k stars, created 2018, last commit Nov 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/BBuf/how-to-optim-algorithm-in-cuda)
  - Contents:
    - Handwritten kernels: reduce, softmax, elementwise, GEMV and linear attention.
    - CUDA-MODE lecture notes.
    - CUTLASS and CuTe DSL notes (GEMM, TMA, WGMMA, swizzling).
    - Triton, PTX ISA and PyTorch internals.
    - Notes on LLM serving and training optimization.
  - Status note in the README: "Actively curated around CUDA kernels, LLM inference optimization, and AI infrastructure. Older Chinese-language notes are being consolidated or replaced with English entry points."
  - Related repos: BBuf/tvm_mlir_learn and BBuf/how-to-learn-deep-learning-framework. The latter has about 502 stars and its last commit was Aug 2026.

  — [README](https://github.com/BBuf/how-to-optim-algorithm-in-cuda); [ungh](https://ungh.cc/repos/BBuf/how-to-learn-deep-learning-framework)
- **PaddleJitLab/CUDATutorial ("A self-learning tutorial for CUDA High Performance Programming"):** about 1.2k stars, last commit Jan 2026. Status: **Maintained**. — [ungh](https://ungh.cc/repos/PaddleJitLab/CUDATutorial)
- **Stale or slowing classics** (still often recommended, but treat as dated):
  - Tony-Tan/CUDA_Freshman: about 2.8k stars, last commit Jan 2024.
  - Liu-xiandong/How_to_optimize_in_GPU (GEMM/GEMV/reduce optimization series): about 1.4k stars, last commit Jul 2023.
  - ifromeast/cuda_learning: about 411 stars, last commit Mar 2025.
  - openmlsys-cuda: last commit Aug 2023.

  — [ungh](https://ungh.cc/repos/Tony-Tan/CUDA_Freshman); [ungh](https://ungh.cc/repos/Liu-xiandong/How_to_optimize_in_GPU); [ungh](https://ungh.cc/repos/ifromeast/cuda_learning); [ungh](https://ungh.cc/repos/openmlsys/openmlsys-cuda)

#### LLM inference and serving notes
- **zhaochenyang20/Awesome-ML-SYS-Tutorial**
  - URL: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial
  - Stats: about 7.7k stars, created 2024-11, last commit 2026-11-24. Status: **Active**. — [ungh](https://ungh.cc/repos/zhaochenyang20/Awesome-ML-SYS-Tutorial)
  - Language: bilingual, with separate English and Chinese READMEs.
  - Sections:
    - SGLang learning notes, including a KV-cache code walkthrough ("When SGLang OOMs, What Exactly Runs Out of Memory?") and the SGLang Omni design.
    - RLHF system notes on slime, AReal, verl and OpenRLHF.
    - Omni-model inference.
    - Scheduling and routing.
    - ML System Fundamentals: Transformers, CUDA and GPU, distributed training and communication, quantization.
    - A developer guide.

  — [README](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)
  - The maintainer, Chayenne Zhao, is listed as an SGLang author who actively helps people in the SGLang community. — [Awesome-MLSys-Blogger](https://github.com/MLSys-Learner-Resources/Awesome-MLSys-Blogger)
- **liguodongiot/llm-action (author 吃果冻不吐果冻皮)**
  - URL: https://github.com/liguodongiot/llm-action. About 25.3k stars, last commit Oct 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/liguodongiot/llm-action)
  - Sections:
    - LLM training: hands-on fine-tuning, distributed parallelism, network communication, alignment.
    - LLM inference: 推理引擎, 推理服务, 推理优化技术.
    - LLM compression: quantization, sparsification, distillation, low-rank decomposition.
    - LLM evaluation, including 推理性能压测 (inference load testing).
  - Many entries link to the author's Zhihu articles, some from the 2023 era (Alpaca, ChatGLM, BELLE). The author also has Zhihu and WeChat-account badges.

  — [README](https://github.com/liguodongiot/llm-action)
- **harleyszhang/llm_note**
  - URL: https://github.com/harleyszhang/llm_note. About 897 stars, last commit Aug 2026. Status: **Maintained**. — [ungh](https://ungh.cc/repos/harleyszhang/llm_note)
  - Content: Transformer structure, LLM quantization, inference optimization (performance analysis, algorithmic and parallel speedups), Triton and CUDA notes, and vLLM code analysis ("vllm推理流程剖析", "vllm优化之cuda_graph详解"). It also includes "2025 最新的高性能计算/推理框架岗位的大厂面试题汇总".
  - **Paywalled component:** the README advertises a paid "自制大模型推理框架" course ("扫码联系课程购买"). It is co-created with the author of 《自制深度学习推理框架》 (Bilibili space.bilibili.com/1822828582) and claims FlashAttention V1/V2/V3, GQA and PagedAttention implementations.

  — [README](https://github.com/harleyszhang/llm_note)
- **harleyszhang/rapid_llm (renamed from lite_llama):** about 198 stars, last commit Oct 2026. It describes itself as "an LLM inference framework with continuous batching, tensor and data parallelism, quantized weight...". — [ungh](https://ungh.cc/repos/harleyszhang/lite_llama); [README](https://github.com/harleyszhang/rapid_llm)
- **zjhellofss/KuiperInfer and KuiperLLama**
  - KuiperInfer: about 3.5k stars, last commit Jun 2025. It is pitched as "校招、秋招、春招、实习好项目！带你从零实现一个高性能的深度学习推理库", supporting Llama 2, UNet, YOLOv5 and ResNet.
  - KuiperLLama: about 583 stars, last commit Oct 2025. It is pitched as "从零动手实现支持LLama2/3和Qwen2.5的大模型推理框架".
  - Both are C++ and both are **Slowing**.

  — [ungh](https://ungh.cc/repos/zjhellofss/KuiperInfer); [ungh](https://ungh.cc/repos/zjhellofss/KuiperLLama)
- **ForceInjection/AI-fundamentals (renamed from AI-fundermentals)**
  - URL: https://github.com/ForceInjection/AI-fundamentals. About 3.5k stars, last commit 2026-11-24. Status: **Active**. — [ungh](https://ungh.cc/repos/ForceInjection/AI-fundermentals)
  - Chinese sections:
    - Hardware and interconnect.
    - AI cluster operations and high-performance communication.
    - Cloud-native AI infrastructure: Kubernetes, GPU virtualization, distributed storage.
    - GPU/CUDA, DPU and Huawei NPU programming.
    - LLM application development.
    - LLM theory.
    - An inference-system section with articles such as:
      - "vLLM + LWS：Kubernetes 上的多机多卡推理方案"
      - an NVIDIA ICMS KV-cache-storage architecture analysis
      - Mooncake's KV-cache-centric scheduling
      - vLLM/llm-d internals
      - "vLLM 与 SGLang 的 KV 复用支持现状"

  — [README](https://github.com/ForceInjection/AI-fundamentals)
- **Mini-engines widely used for learning** (English code, heavily studied in the Chinese community):
  - GeeeekExplorer/nano-vllm: about 16.6k stars, last commit Apr 2026.
  - sgl-project/mini-sglang ("A compact implementation of SGLang, designed to demystify the complexities of modern LLM serving systems"): about 5.6k stars, last commit May 2026.

  — [ungh](https://ungh.cc/repos/GeeeekExplorer/nano-vllm); [ungh](https://ungh.cc/repos/sgl-project/mini-sglang)
  - A Chinese nano-vLLM explainer was cross-posted to Zhihu and WeChat on 2025-12-31 (seen in search results; not read in full). — [cnblogs](https://www.cnblogs.com/cswuyg/p/19471225)
- **vLLM 中文站:** https://vllm.hyper.ai/ is live (HTTP 200), titled "vLLM 是一个快速且易于使用的库…| vLLM 中文站". It is a Chinese translation of the docs on a HyperAI domain. The maintainer and how far the translation lags were not verified. — [site](https://vllm.hyper.ai/)
- **DeepSeek open infra:**
  - deepseek-ai/open-infra-index: about 8.1k stars, last commit May 2025.
  - The "DeepSeek-V3 / R1 推理系统概览" Zhihu post is indexed as a 2025.03 entry in Awesome-LLM-Inference.

  — [ungh](https://ungh.cc/repos/deepseek-ai/open-infra-index); [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
- **Interview-prep repos (Chinese):**
  - wdndev/llm_interview_note: about 15.4k stars, but the default branch's last commit was Aug 2024 (**Stale** content).
  - luhengshiwo/LLMForEverybody: about 7.6k stars, last commit Aug 2026, "LLMs春/秋招大模型面试前必看".

  — [ungh](https://ungh.cc/repos/wdndev/llm_interview_note); [ungh](https://ungh.cc/repos/luhengshiwo/LLMForEverybody)

#### Zhihu authors and columns
Source: curated GitHub indexes, because Zhihu returned HTTP 403 to fetches. — [Zhihu fetch attempt](https://zhuanlan.zhihu.com/p/608318764)
- **MLSys-Learner-Resources/Awesome-MLSys-Blogger.** About 347 stars, created and last updated 2025-01-05, so it is a **Stale** snapshot. — [ungh](https://ungh.cc/repos/mlsys-learner-resources/Awesome-MLSys-Blogger)

  Its 系统方向 list includes:
  - 方佳瑞 (Colossal-AI, xDiT, YunChang).
  - 手抓饼熊.
  - Fazzie, author of the well-known "MLSys入坑指南" (zhuanlan.zhihu.com/p/608318764).
  - Chayenne Zhao (SGLang).
  - Hsword (SpecInfer/SpotServe author; Purdue AP).
  - Pentium PRO ("MLsys各方向综述").
  - ZZYSonny (Triton/SASS).
  - 刀刀宁, whose column is "刀刀宁聊大模型推理".
  - DefTruth (Awesome-LLM-Inference, CUDA-Learn-Notes).
  - 陈乐群 (Punica/Atom).
  - 陈天奇.
  - BBuf, who "会更新很多MLSys相关笔记（GPU Mode、SGLang）, GiantPandaCV运营者".
  - Yineng Zhang (SGLang).
  - zzk again (OneFlow/SiliconFlow, GiantPandaCV).

  — [Awesome-MLSys-Blogger README](https://github.com/MLSys-Learner-Resources/Awesome-MLSys-Blogger)
- **Authors in LeetCUDA's blog index:** DefTruth (about 86 entries), reed (16), 紫气东来 (12), 竹熙佳处 (6), StarCap (5), JoeNomad (5), 进击的Killua (4), 懒蚂蚁呀不嘿 (4), BBuf, 朱小霖, 有了琦琦的棍子 and zzk again. — [LeetCUDA README](https://github.com/xlite-dev/LeetCUDA)
- **猛猿《图解大模型计算加速系列》:** includes PagedAttention, "vLLM 源码解析1" (overall architecture) and "vLLM 源码解析2" (the scheduler: SequenceGroup, add_request, step and preemption), plus FlashAttention V1/V2. This was verified only through a repost, not the original Zhihu page. — [repost](https://blog.youkuaiyun.com/fengbeely/article/details/140117942)

#### Bilibili and video
- **ZOMI 酱** (AISystem/AIInfra videos): space.bilibili.com/517221395, linked from the AISystem README. — [README](https://github.com/Infrasys-AI/AISystem)
- **Author of 《自制深度学习推理框架》** (KuiperInfer lineage): space.bilibili.com/1822828582. — [llm_note README](https://github.com/harleyszhang/llm_note)
- **InternLM 实战营 lesson videos:** for example, LMDeploy 量化部署进阶实践 (BV1df421q7cR). — [README](https://github.com/InternLM/Tutorial)
- **so-large-lm videos:** for example, "进击的AI：大模型技术全景" (BV14x4y1x7bP). — [README](https://github.com/datawhalechina/so-large-lm)
- **HuaizhengZhang's "AI System School" video tutorial:** published on YouTube, Bilibili (BV1ZwYUerEtL) and Xiaohongshu. — [README](https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero)

#### Meetup materials in China
- **SGLang events page.** https://www.sglang.io/events (fetched 2026-11-24) lists many China events. — [SGLang Events](https://www.sglang.io/events)

  | Date | City | Event |
  | --- | --- | --- |
  | 2026-01-31 | Beijing | SGLang x OpenAnolis Meetup (recap available) |
  | 2026-03-07 | not shown | Meetup with SGLang, Qwen, Alibaba Cloud Tair KVCache, NVIDIA and Mooncake |
  | 2026-05-23 | Hangzhou | Ant tech day: Miles framework talk |
  | 2026-09-09 | Shanghai | Green Computing Forum |
  | 2026-09-17 | Shanghai | NVIDIA Open-Source AI Ecosystem Meetup: efficient DeepSeek-V4 inference with million-token contexts |
  | 2026-09-21 | Beijing | AICC 2026 main-stage talk plus an afternoon track on kernel optimization, agentic inference and production deployment |
  | 2026-09-22 | Beijing | Baidu AI Cloud x SGLang meetup on agentic inference |
  | 2026-09-22–24 | Hangzhou | Agentic OS showcase |
  | 2026-10-11 | Shanghai | SGLang x MetaX meetup |
  | 2026-10-17 | Shenzhen | KVCDN Workshop at GOSIM Shenzhen 2026 (slides available) |
  | 2026-11-20 | Beijing | Session on model serving, kernels and performance |

- **Alibaba developer-community coverage** of a "沐曦x龙蜥xSGLang 技术 MeetUp" in Beijing, with a recap article. — [preview](https://developer.aliyun.com/article/1739608); [recap](https://developer.aliyun.com/article/1740909)
  - **Conflict:** a search snippet dated this meetup 2026-06-06, but the events page lists an SGLang x OpenAnolis (龙蜥) Beijing meetup on 2026-01-31 and an SGLang x MetaX (沐曦) meetup in Shanghai on 2026-10-11. The date is unverified.
- **sgl-project/sgl-learning-materials:** about 908 stars, last commit Jan 2026. The slides span the first LMSYS online meetup (2024-10-16) through the AMD SGLang Meetup (2025-08-22). Its Slack link is slack.sglang.ai, while the main SGLang README now uses slack.sglang.io. — [ungh](https://ungh.cc/repos/sgl-project/sgl-learning-materials); [README](https://github.com/sgl-project/sgl-learning-materials); [SGLang README](https://github.com/sgl-project/sglang)
- **vLLM:** a PyTorch blog post, "vLLM Beijing Meetup: Advancing Large-scale LLM Deployment", is dated 2025-08-07. vLLM's docs meetups page now simply points to https://vllm.ai/events. That page is rendered by JavaScript, so its China entries could not be extracted. — [PyTorch blog](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/); [vLLM meetups.md](https://github.com/vllm-project/vllm/blob/main/docs/community/meetups.md)

#### WeChat public accounts
- **吃果冻不吐果冻皮** has a WeChat account and a Zhihu account, both linked from the llm-action README. — [README](https://github.com/liguodongiot/llm-action)
- **BBuf** runs GiantPandaCV (per the blogger index). A CSDN repost credits its source as the WeChat account "GiantPandaLLM". The repost credit was seen only via search, not read in full. — [Awesome-MLSys-Blogger](https://github.com/MLSys-Learner-Resources/Awesome-MLSys-Blogger); [CSDN repost](https://adg.csdn.net/6a2f9f10662f9a54cb7f4e3b.html)
- **Datawhale** promotes its WeChat and learning community. The tiny-universe README has a "关注我们" section and describes Datawhale as "一个专注于AI领域的学习圈子". — [README](https://github.com/datawhalechina/tiny-universe)

### Inferences
- **Best Chinese set for the LLM-serving learner** (all Active in Oct–Nov 2026):
  - LeetCUDA (kernels).
  - Awesome-ML-SYS-Tutorial (SGLang internals and RL systems).
  - AI-fundamentals (K8s-level deployment plus vLLM/SGLang/llm-d/Mooncake).
  - llm-action (a broad index).
  - Optionally, BBuf's repo for CUTLASS/CuTe and PTX.
- **Language shift:** some leading Chinese authors (BBuf, xlite-dev, Chayenne Zhao) now publish English or bilingual material first. For a learner who reads both languages, the community has converged on the same engine-centric content.
- **Paywall boundary:** most material is free. The notable paid offering found is the "自制大模型推理框架" course promoted in llm_note. Treat repos that market themselves as "校招/秋招好项目" mainly as portfolio projects.
- **Evidence that SGLang has strong China ties:** at least 10 China-based SGLang events appear on the official events page between Jan and Nov 2026, and SGLang people are prominent among the Zhihu bloggers.

### Gaps
- Original Zhihu articles and columns could not be fetched (HTTP 403), so their content, dates and paywall status (Zhihu 盐选) are unverified.
- Bilibili channels other than those linked from READMEs could not be verified. That includes the "[EP01][精剪版] vllm源码讲解" series seen only via a CSDN note. — [CSDN note](https://devpress.csdn.net/amd/6a195189662f9a54cb784366.html)
- The vLLM China meetups in 2026 could not be listed, because vllm.ai/events is JavaScript-rendered.
- WeChat groups have no publicly verifiable join information, and WeChat articles are not indexable.
- The maintainer of vLLM 中文站 and how current its translation is were not verified.

## Q3. What community-built AI infra / ML systems learning roadmaps exist, what learning order do they recommend, and where do they agree?

### Takeaway
Two polished English roadmaps from 2026 define the current consensus:
- roadmap.sh's **"Inference Engineering"** roadmap.
- Baseten's free book **Inference Engineering** by Philip Kiely, finished in January 2026.

They follow almost the same order:
1. Metrics and model mechanics.
2. Bottleneck analysis (roofline).
3. GPU architecture and CUDA.
4. Frameworks and inference engines (vLLM, SGLang, TensorRT-LLM).
5. Benchmarking.
6. Quantization, speculative decoding and caching.
7. Parallelism and disaggregation.
8. Production (containers, autoscaling, routing, multi-cloud, observability).

Chinese role-based roadmaps from 2026 (AtomGit/CSDN) agree on fundamentals first, then vLLM internals, then a mini-engine capstone. They disagree on where CUDA belongs, and they usually omit SGLang. Their vLLM source-reading lists point to V0 files that no longer exist on vLLM main.

### Cited Findings

#### English and global roadmaps
- **roadmap.sh "Inference Engineering"** (https://roadmap.sh/inference-engineering)
  - Described as a "Step-by-step guide to becoming an Inference Engineer in 2026" and credited to roadmap.sh/@nilbuild. — [roadmap.sh](https://roadmap.sh/inference-engineering)
  - The roadmap's data (175 nodes) is served at its API endpoint, and a PDF is available. — [API JSON](https://roadmap.sh/api/v1-official-roadmap/inference-engineering); [PDF](https://roadmap.sh/pdfs/roadmaps/inference-engineering.pdf)
  - The source repo, nilbuild/developer-roadmap, has about 372.7k stars and was last committed 2026-11-24. — [ungh](https://ungh.cc/repos/kamranahmedse/developer-roadmap)
  - Top-level topics in diagram order (sorted by vertical position, so the order is approximate). — [API JSON](https://roadmap.sh/api/v1-official-roadmap/inference-engineering)
    1. Introduction: inference vs. training; open vs. closed models; shared vs. dedicated inference; the inference stack; TTFT, tokens/s, inter-token latency, latency vs. throughput, latency percentiles.
    2. Neural Networks.
    3. Large Language Models: attention, KV cache, prefill and decode phases, MoE, online vs. offline and asynchronous inference.
    4. Image & Video Models.
    5. Bottleneck Analysis: ops:byte ratio, arithmetic intensity, roofline model.
    6. Attention Optimization: FlashAttention, PagedAttention, attention variants.
    7. GPU Architecture: compute, memory, Hopper, Blackwell, Grace and Vera, instances and interconnects, sizing and procurement.
    8. GPU Models.
    9. Local & Edge Inference.
    10. CUDA: kernel selection and fusion, writing custom kernels.
    11. Frameworks & Libraries: PyTorch, safetensors, TensorRT, Transformers and Diffusers, ONNX Runtime.
    12. Model Formats & Runtimes.
    13. Inference Engines: vLLM, SGLang, TensorRT-LLM.
    14. Benchmarking & Profiling: AIPerf, Locust, SGLang genai-bench, NVIDIA Dynamo.
    15. Quantization: QAT, PTQ, measuring quality impact, ModelOpt.
    16. Speculative Decoding: draft-target, EAGLE, n-gram; structured and constrained decoding.
    17. Caching: KV-cache storage, cache-aware routing.
    18. Model Parallelism: tensor, pipeline and expert parallelism, long context, multi-node.
    19. Disaggregation (NVIDIA Dynamo).
    20. Vision Language Models, Embedding Models, Speech (VAD, ASR, TTS) and Image & Video Generation.
    21. Containerization (NIMs).
    22. Autoscaling: cold starts, routing and load balancing, LoRA-aware routing, request queueing, scale to zero.
    23. Multi-Cloud: GPU procurement, capacity management, geo-aware load balancing, security and compliance.
    24. Testing & Deployment: zero-downtime deploys, cost estimation, observability.
- **Inference Engineering (book)** by Philip Kiely (Baseten Books, © 2026)
  - Free to read online in full as Markdown; print and ebook editions are also offered. — [llms.txt](https://www.baseten.co/inference-engineering/llms.txt); [book page](https://www.baseten.co/inference-engineering/)
  - Table of contents:
    - Ch. 1 Prerequisites: scale and specialization, model selection, measuring latency and throughput.
    - Ch. 2 Models: LLM and image-generation inference mechanics, calculating inference bottlenecks, optimizing attention.
    - Ch. 3 Hardware: GPU architecture and generations, instances, other accelerators, local inference.
    - Ch. 4 Software: CUDA, DL frameworks, inference engines, NVIDIA Dynamo, benchmarking and load testing.
    - Ch. 5 Techniques: quantization, speculative decoding, caching, model parallelism, disaggregation.
    - Ch. 6 Modalities: VLMs, embeddings, ASR, TTS, image and video generation.
    - Ch. 7 Production: containerization, autoscaling, multi-cloud capacity, testing and deployment, client code.
    - Appendix B, Recommended Reading.

    — [llms.txt](https://www.baseten.co/inference-engineering/llms.txt)
  - A search summary reported that the book was finished in January 2026 and is based on interviews with Baseten engineers and on conference talks. — [book page](https://www.baseten.co/inference-engineering/book/)
- **GPU MODE resource-stream.** Section order: Lectures/Reading Group → "1st Contact with CUDA" → "2nd Contact" → Hazy Research → Papers/Case Studies → Books → CUDA Courses → "CUDA Grandmasters" (Tri Dao, Tim Dettmers, Sasha Rush) → Practice → PyTorch Performance Optimization → PyTorch Internals & Debugging → Profiling → Python GPU Computing → Advanced Topics/Compilers. — [README](https://github.com/gpu-mode/resource-stream)
- **HuaizhengZhang "AI System School"** (repo renamed from Awesome-System-for-Machine-Learning to AI-Infra-from-Zero-to-Hero)
  - Organized as "Path to System for AI", with a whitepaper.
  - Categories: ML/DL infra (data processing, training system, inference system, ML infrastructure); LLM infra (LLM training, LLM serving); domain-specific infra; conferences, surveys, books, videos, courses and blogs.
  - It links a Zhihu question on what knowledge structure ML-systems research requires (question 315611053).

  — [README](https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero)
- **mlabonne/llm-course** ("Course to get into LLMs with roadmaps and Colab notebooks"): about 85.2k stars, last commit Feb 2026. Its content was not inspected here. — [ungh](https://ungh.cc/repos/mlabonne/llm-course)
- **Open textbooks used as roadmaps:**
  - Harvard CS249r "Machine Learning Systems" (now "Foundations, Scaling, Agentic AI, and Physical AI (Vols I–IV)"): about 30.3k stars, last commit 2026-11-23, site mlsysbook.ai live.
  - stas00/ml-engineering ("Machine Learning Engineering Open Book"): about 19.7k stars, last commit 2026-11-24.

  — [ungh](https://ungh.cc/repos/harvard-edge/cs249r_book); [ungh](https://ungh.cc/repos/stas00/ml-engineering); [mlsysbook.ai](https://mlsysbook.ai/)

#### Chinese roadmaps
- **"推理框架负责人 — 学习路线" by rosemary512 (AtomGit/CSDN, 2026-05-22).** A 6-month, about 520-hour plan. — [gitcode](https://gitcode.csdn.net/6a100a9710ee7a33f274631f.html)
  - Phase 1 (months 1–2), CUDA basics: SM, warps, memory hierarchy, shared memory and bank conflicts, coalescing, async copy, streams. Project: a hand-written GEMM reaching at least 50% of cuBLAS, using Simon Boehm's matmul post, the CUDA C++ Programming Guide and CUTLASS.
  - Phase 2 (months 3–4), inference-engine internals:
    - vLLM deep dive: PagedAttention, KV-cache management, scheduler, prefix caching.
    - Quantization: GPTQ, AWQ, FP8, INT4.
    - FlashAttention.
    - Project: a pure-CUDA PagedAttention benchmarked against vLLM's.
  - Phase 3 (months 5–6), advanced optimization: speculative decoding (Medusa, EAGLE) and PD disaggregation.
  - Capstone: a mini inference engine of about 3,000 lines of C++/CUDA, with a safetensors/GGUF loader, PagedAttention and FlashAttention, block KV cache with LRU eviction, continuous batching, INT8/FP8 and an OpenAI-compatible API server. The target is about 75% of vLLM's throughput on Llama-3-8B on an A100.
  - **It does not mention SGLang.**
- **"AI Infra 后端开发工程师 — 学习路线" by rosemary512 (2026-05-22).** A 3-month, about 18 hours/week plan. — [gitcode](https://gitcode.csdn.net/6a10095f10ee7a33f274629f.html)
  - Month 1: Go, gRPC/Protobuf, Docker and Kubernetes (minikube).
  - Month 2: Redis Streams, Kafka/NATS, the vLLM API server and scheduler architecture, SSE/WebSocket streaming.
  - Month 3: build an OpenAI-compatible Go inference gateway with SSE, model routing (load balancing, GPU-memory-aware), Redis sessions, and wrk/vegeta load tests with pprof.
  - It mentions neither SGLang nor llm-d.
- **"AI 推理工程师 · 零基础完整学习路线" (CSDN user 2501_92503814, 2026-06-01).** A 10–12 month plan. — [CSDN](https://aicoding.csdn.net/6a69c37410ee7a33f2940785.html)
  1. Programming basics (Python, C/C++, Linux).
  2. Deep-learning basics (d2l, Karpathy, 李宏毅).
  3. CUDA, in parallel with stage 2 (PMPP, the CUDA guide, Nsight, roofline).
  4. Inference frameworks (ONNX, TensorRT, ONNX Runtime, Triton Inference Server, FastAPI, Docker).
  5. LLM inference optimization (GPTQ/AWQ/SmoothQuant/GGUF, vLLM PagedAttention and continuous batching, TensorRT-LLM, tensor/pipeline/expert parallelism, custom kernels).
  6. Projects: a quantized 7B model on vLLM or TensorRT-LLM with FastAPI; a CUDA kernel packaged as a PyTorch extension; an end-to-end service with Prometheus and Grafana.
  7. Job preparation.

  It also does not mention SGLang.
- **csdiy.wiki "CS自学指南" (PKUFlyingPig/cs-self-learning).** About 76.8k stars, last commit Sep 2026. Its 机器学习系统 section lists CMU 15-442/642 Machine Learning Systems, 智能计算系统 (AICS), CMU 10-414/714 Deep Learning Systems, MIT 6.5940 TinyML and Efficient Deep Learning Computing, Machine Learning Compilation and UCSD CSE234. It also has 学习路线图 pages. — [ungh](https://ungh.cc/repos/PKUFlyingPig/cs-self-learning); [mkdocs.yml](https://github.com/PKUFlyingPig/cs-self-learning/blob/master/mkdocs.yml); [site](https://csdiy.wiki/)
- **Implicit orders inside Chinese courses:**
  - AIInfra: overview → clusters → communication and storage → cloud-native → training → inference → algorithms → applications. — [GitHub page](https://github.com/Infrasys-AI/AIInfra)
  - openmlsys V2: programming model → accelerators/CUDA → compilers and runtime → data → training → serving → RL systems → GPU clusters. — [README](https://github.com/openmlsys/openmlsys)
  - ForceInjection/AI-fundamentals: hardware → cluster operations → cloud-native → CUDA → apps → ML/LLM theory → inference systems. — [README](https://github.com/ForceInjection/AI-fundamentals)
- **Classic Chinese roadmap articles** named in curated lists but not readable here (Zhihu 403): Fazzie's "MLSys入坑指南" and Pentium PRO's "MLsys各方向综述". — [Awesome-MLSys-Blogger](https://github.com/MLSys-Learner-Resources/Awesome-MLSys-Blogger)

#### Outdated advice detected
- Both Chinese 2026 roadmaps tell learners to read vLLM V0 files:
  - The 推理框架负责人 roadmap lists vllm/worker/model_runner.py, vllm/core/block_manager.py, vllm/core/scheduler.py, vllm/attention/backends/flash_attn.py and vllm/model_executor/layers/fused_moe/.
  - The CSDN roadmap lists vllm/core/scheduler.py.
  - On 2026-11-24, vllm/core/scheduler.py, vllm/core/block_manager.py and vllm/worker/model_runner.py all return **404** on vLLM main, while vllm/v1/core/sched/scheduler.py and vllm/v1/core/kv_cache_manager.py return 200.
  - SGLang's python/sglang/srt/managers/scheduler.py and srt/mem_cache/radix_cache.py return 200.

  — [gitcode roadmap](https://gitcode.csdn.net/6a100a9710ee7a33f274631f.html); [CSDN roadmap](https://aicoding.csdn.net/6a69c37410ee7a33f2940785.html); [vLLM V1 scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py); [SGLang scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- Low-quality roadmap sources seen in search: viral LinkedIn posts ("The LLM Inference Roadmap I Wish More Engineers Followed") and an explicitly AI-generated roadmap page ("LLM Inference Engineering Roadmap | Generated by AI"). Neither was used for the consensus. — [LinkedIn](https://in.linkedin.com/in/m-n-g); [lzwjava](https://lzwjava.github.io/llm-inference-engineering-roadmap-en)

### Inferences
**Consensus learning order** (synthesized from roadmap.sh, the Baseten book, the two rosemary512 roadmaps, the CSDN roadmap and GPU MODE):

0. **Programming foundations:** Python, C++, Linux and Git. Learners heading for the backend/platform track also need Go, gRPC, Docker and Kubernetes.
1. **Model mechanics:** Transformer and attention, KV cache, prefill vs. decode, MoE. Sources: d2l-zh, happy-llm, AIInfra 06.
2. **Performance mental model:** TTFT, ITL/TPOT, throughput, percentiles, arithmetic intensity, roofline, and why decode is memory-bound.
3. **GPU architecture and CUDA basics, plus Triton:** LeetCUDA, PaddleJitLab, GPU MODE.
4. **Use engines and benchmark them:** vLLM and SGLang, with genai-bench or AIPerf. Sources: self-llm, vLLM 中文站, engine docs.
5. **Engine internals:** PagedAttention or radix cache, continuous batching, the scheduler, prefix caching. Read the source (V1 paths) and the mini-engines (nano-vllm, mini-sglang, KuiperLLama).
6. **Techniques:** quantization (GPTQ, AWQ, FP8), FlashAttention, speculative decoding (EAGLE), structured outputs.
7. **Distributed inference:** tensor, pipeline and expert parallelism, multi-node, PD disaggregation, KV-cache storage and transfer, cache-aware routing, NCCL/RDMA.
8. **Production:** containers, autoscaling and cold starts, routing and load balancing, multi-cloud capacity, observability, cost.

The capstone is a mini-engine, a gateway, or a benchmarked deployment written up as blog posts.

**Where the roadmaps agree:**
- Fundamentals and metrics come before engine internals.
- KV cache and PagedAttention are the pivotal concepts.
- Learning should be benchmark-driven.
- vLLM source reading is the default deep dive.
- Each roadmap ends with a portfolio project.

**Where they disagree:**
1. **Where CUDA goes:** first and deep in the Chinese kernel-oriented roadmap, mid-sequence in roadmap.sh and Baseten, and absent from the backend roadmap.
2. **SGLang:** covered by roadmap.sh and Baseten-era material, absent from the Chinese roadmaps.
3. **CV-era tooling:** TensorRT, ONNX and Triton Inference Server get heavy weight in the CSDN roadmap.
4. **Modalities:** roadmap.sh and Baseten add VLM, speech, image and video; the Chinese roadmaps are LLM-only.

**Caveats:**
- roadmap.sh's section sequence closely mirrors the Baseten book's chapter order. Whether one was derived from the other is unconfirmed, so treat them as one viewpoint rather than two independent votes.
- The vLLM V1 migration has outdated many Chinese source-reading guides, including 2026 ones. Learners should map V0 paths to vllm/v1/....

### Gaps
- The text of the classic Zhihu roadmaps (MLSys入坑指南 and others) could not be read.
- The third rosemary512 roadmap ("C++ 程序员转型 AI Infra 学习路线") was seen only in search results and not fetched. — [gitcode](https://gitcode.csdn.net/6a216c47662f9a54cb7994d6.html)
- mlabonne/llm-course's deployment and inference sections were not inspected.
- No authoritative, peer-reviewed curriculum exists. Every roadmap here is a practitioner's or a vendor's opinion; Baseten is an inference vendor.

## Q4. Which curated awesome-lists and paper lists are maintained and most useful as of 2025–2026?

### Takeaway
The lists that are clearly maintained in late 2026 are:
- **AmberLJC/LLMSys-PaperList**: Oct 2026, with MLSys'26 papers and a 2024→2026 trends section.
- **xlite-dev/Awesome-LLM-Inference**: commits in Nov 2026, with entries dated 2026.11.
- **byungsoo-oh/ml-systems-papers**: Oct 2026.
- **tensorchord/Awesome-LLMOps**: Nov 2026.
- GPU MODE's lectures repo (Sep 2026) and resource-stream (Mar 2026).

HuaizhengZhang's list (renamed "AI-Infra-from-Zero-to-Hero"), Awesome-Efficient-LLM and Hannibal046/Awesome-LLM have slowed since mid-2025. ModelTC/awesome-lm-system is dead.

### Cited Findings
- **AmberLJC/LLMSys-PaperList** ("Large Language Model (LLM) Systems Paper List")
  - URL: https://github.com/AmberLJC/LLMSys-PaperList. About 2.3k stars, last commit Oct 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/AmberLJC/LLMSys-PaperList)
  - Sections: "Trends at a Glance (2024 → 2026)", training and serving systems (including multimodal), LLM for Systems, industrial technical reports, ML conferences (e.g. NeurIPS 2025), frameworks, ML systems, surveys, benchmarks/traces, and an "MLSys Courses" section. It already lists MLSys'26 papers.
  - Language: English. Level: intermediate to advanced.

  — [README](https://github.com/AmberLJC/LLMSys-PaperList)
- **xlite-dev/Awesome-LLM-Inference** (maintainer DefTruth)
  - URL: https://github.com/xlite-dev/Awesome-LLM-Inference. About 5.6k stars, last commit 2026-11-24. Status: **Active**. — [ungh](https://ungh.cc/repos/xlite-dev/Awesome-LLM-Inference); [shields](https://img.shields.io/github/last-commit/xlite-dev/Awesome-LLM-Inference.json)
  - Papers with code, organized by topic: DeepSeek/MLA; multi-GPU and multi-node parallelism; prefill/decode disaggregation; continuous and in-flight batching; weight and activation quantization; IO/FLOPs-aware sparse attention; KV-cache scheduling, quantization and dropping; long-context; parallel decoding; MoE inference; and more.
  - Entries run up to 2026.11 (for example, a measurement of SGLang FP8-KV/HiCache flags on agent traces). The list also links Chinese Zhihu blogs.
  - Language: English with Chinese links.

  — [README](https://github.com/xlite-dev/Awesome-LLM-Inference)
- **HuaizhengZhang/AI-Infra-from-Zero-to-Hero** (formerly Awesome-System-for-Machine-Learning, now "AI System School")
  - About 4.5k stars, last commit Jul 2025. Status: **Slowing**. — [ungh](https://ungh.cc/repos/HuaizhengZhang/Awesome-System-for-Machine-Learning)
  - Categories: data processing, training, inference, ML infra, LLM training, LLM serving, domain-specific systems, conferences, books, videos, courses, blogs. The README announces a new "Lets Go AI" website. Language: English, with some Chinese links.

  — [README](https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero)
- **gpu-mode/resource-stream** ("GPU programming related news and material links"): about 2.4k stars, last commit Mar 2026 (pushed Jun 2026). Status: **Maintained**. — [ungh](https://ungh.cc/repos/gpu-mode/resource-stream)
- **gpu-mode/lectures** ("Material for gpu-mode lectures"): about 7.1k stars, last commit Sep 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/gpu-mode/lectures)
- **byungsoo-oh/ml-systems-papers** ("Curated collection of papers in machine learning systems"): about 662 stars, last commit Oct 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/byungsoo-oh/ml-systems-papers)
- **tensorchord/Awesome-LLMOps** ("best LLMOps tools for developers"): about 6.0k stars, last commit Nov 2026. Status: **Active**. — [ungh](https://ungh.cc/repos/tensorchord/Awesome-LLMOps)
- **chenhongyu2048/LLM-inference-optimization-paper:** about 279 stars, last commit Feb 2026. Status: **Maintained**. — [ungh](https://ungh.cc/repos/chenhongyu2048/LLM-inference-optimization-paper)
- **Chinese-specific indexes:**
  - LeetCUDA's blog index (**Active**). — [README](https://github.com/xlite-dev/LeetCUDA)
  - Awesome-MLSys-Blogger (snapshot from Jan 2025). — [ungh](https://ungh.cc/repos/mlsys-learner-resources/Awesome-MLSys-Blogger)
  - csdiy.wiki's ML-systems course list (**Active**, Sep 2026). — [ungh](https://ungh.cc/repos/PKUFlyingPig/cs-self-learning)
- **Slowing or stale lists:**
  - horseee/Awesome-Efficient-LLM: about 2.0k stars, last commit Jun 2025.
  - Hannibal046/Awesome-LLM: about 27.5k stars, last commit Jul 2025.
  - coderonion/awesome-llm-and-aigc: about 818 stars, last commit Aug 2025.
  - ModelTC/awesome-lm-system: about 57 stars, last commit Dec 2023.

  — [ungh](https://ungh.cc/repos/horseee/Awesome-Efficient-LLM); [ungh](https://ungh.cc/repos/Hannibal046/Awesome-LLM); [ungh](https://ungh.cc/repos/codingonion/awesome-llm-and-aigc); [ungh](https://ungh.cc/repos/ModelTC/awesome-lm-system)

### Inferences
- **For an LLM-serving learner in late 2026, three lists suffice:** Awesome-LLM-Inference (topic-organized, most current, links Chinese posts), LLMSys-PaperList (a broad systems view with conference tracking) and GPU MODE resource-stream (kernel learning). Use HuaizhengZhang's list only for historical, pre-2025 breadth.
- **Star counts do not track freshness.** The most-starred general list (Awesome-LLM, 27.5k) has been static since Jul 2025, while smaller lists (ml-systems-papers) are current.

### Gaps
- The number of entries and the update cadence of each list were not measured; only last-commit dates were.
- The GPU MODE Discord's own pinned resources and the lecture schedule (inside Discord) were not inspected.

## Q5. What active communities support learners, and what skills do AI infra / LLM inference engineer job postings and practitioner career guides commonly require (2025–2026)?

### Takeaway
**Communities.** The core hubs are:
- GPU MODE Discord, with lectures and kernel material.
- vLLM's Slack (slack.vllm.ai), forum (discuss.vllm.ai) and events page.
- SGLang's Slack (slack.sglang.io), events page with frequent China meetups, and SGLang Summit 2026.
- Chinese channels: Zhihu authors, Bilibili (ZOMI), the Datawhale and InternLM learning camps, and WeChat accounts.

**Skills.** Live postings (Nov 2026) from Anthropic, Inferact (vLLM-focused), RadixArk (SGLang creators), Perplexity, Together AI, xAI, Databricks, Baseten, Modal, Cohere and Meituan (LongCat) converge on a common core:
- Python and PyTorch, with internals for engine roles.
- LLM-serving mechanics: KV-cache management, prefix caching, continuous batching, scheduling, PD disaggregation, speculative decoding, quantization, TP/PP/EP.
- Hands-on experience with vLLM, SGLang or TensorRT-LLM.
- Profiling and roofline-style performance reasoning.

The role then branches into either kernels (CUDA, Triton, CUTLASS/CuTe DSL, TileLang) or systems (C++/Rust/Go, distributed systems, Kubernetes, autoscaling, routing, observability, NCCL/RDMA).

Open-source contributions and technical blog posts are stated differentiators. Chinese JDs put extra weight on SGLang/vLLM source-level familiarity and C++/CUDA/AscendC.

### Cited Findings

#### Communities
- **GPU MODE:** Discord at https://discord.gg/gpumode. Upcoming lectures are listed in the Discord server's Events. Lecture material is in gpu-mode/lectures (about 7.1k stars, last commit Sep 2026). — [resource-stream README](https://github.com/gpu-mode/resource-stream); [ungh](https://ungh.cc/repos/gpu-mode/lectures)
- **vLLM:** Slack at https://slack.vllm.ai "for coordinating contributions and development"; forum at https://discuss.vllm.ai (live, titled "vLLM Forums"); events at https://vllm.ai/events; blog at blog.vllm.ai. The repo has about 97.5k stars and was last committed 2026-11-24. — [vLLM README](https://github.com/vllm-project/vllm); [forum](https://discuss.vllm.ai/); [meetups.md](https://github.com/vllm-project/vllm/blob/main/docs/community/meetups.md); [ungh](https://ungh.cc/repos/vllm-project/vllm)
- **SGLang:**
  - Slack at https://slack.sglang.io and events (meetups, developer meetings, workshops, office hours) at https://www.sglang.io/events.
  - "SGLang Summit 2026" is promoted at sglang.io/summit. A Luma page shows "SGLang & LMSYS Summit 2026".
  - Docs at docs.sglang.io, a Cookbook at cookbook.sglang.io and a public roadmap at roadmap.sglang.io.
  - The repo has about 39.0k stars and was last committed 2026-11-24.

  — [SGLang README](https://github.com/sgl-project/sglang); [Luma](https://luma.com/z0s2dufq); [ungh](https://ungh.cc/repos/sgl-project/sglang)
- **Chinese learning communities:**
  - Datawhale, whose learning community and WeChat are promoted in its repos. — [tiny-universe README](https://github.com/datawhalechina/tiny-universe)
  - The InternLM 实战营 camps, with tasks, certificates and compute credits. — [README](https://github.com/InternLM/Tutorial)
  - SGLang China meetups (see Q2). — [SGLang Events](https://www.sglang.io/events)
  - Zhihu MLSys bloggers. — [Awesome-MLSys-Blogger](https://github.com/MLSys-Learner-Resources/Awesome-MLSys-Blogger)
- **MLSys conference:** MLSys 2026 papers already appear in LLMSys-PaperList (for example, "Efficient Long-Context LM Training by Core Attention Disaggregation", an MLSys'26 oral). — [README](https://github.com/AmberLJC/LLMSys-PaperList)

#### Job postings
Pulled from live Greenhouse and Ashby boards on 2026-11-24; posting dates in parentheses.

- **Inferact, Member of Technical Staff – Inference** (2026-06/09; SF, Singapore, Remote). — [Ashby](https://jobs.ashbyhq.com/inferact/43c0ca54-fcf5-41fa-83a1-38800c75ccc0)
  - Strong Python with experience in PyTorch internals.
  - Experience with LLM inference systems (vLLM, TensorRT-LLM, SGLang, TGI).
  - Deep understanding of KV-cache memory management, prefix caching and hybrid model serving.
  - Bonus: "Implemented core features in vLLM…", vLLM integrations (verl, OpenRLHF, Unsloth, LlamaFactory), and "Written widely-shared technical blogs or side projects on vLLM or LLM inference".
- **Inferact, Inference Engineering Co-op** (2026-11-19). — [Ashby](https://jobs.ashbyhq.com/inferact/2b6032f9-12a5-4083-b5e6-4bec5376cbaa)
  - Tracks:
    - Inference runtime.
    - Performance and scale: tensor, expert and context parallelism; prefill/decode separation; KV-cache transport.
    - Kernel engineering: CUDA, Triton, TileLang, CUTLASS/CuTe.
    - AMD: ROCm, HIP, CK, AITER.
    - TPU: JAX, XLA, Pallas.
    - Cloud orchestration: Kubernetes operators, topology-aware GPU scheduling, token-aware routing.
  - Languages: Python, C++, Rust or Go.
- **RadixArk** (its postings describe a company whose engineers "created SGLang (30K+ GitHub stars…")
  - Core Engine (2026-02-17): GPU architecture; latency- and throughput-critical systems; distributed systems and networking; Python/Rust/C++/Go; profiling; cross-layer debugging. Strong plus: SGLang, vLLM or TensorRT-LLM; CUDA or Triton; batching, KV-cache and scheduling. — [Ashby](https://jobs.ashbyhq.com/radixark/134eea08-4726-46e1-a0e7-7c1dfc9545cd)
  - Multi-Hardware (2026-08-03): CUDA, ROCm/HIP, Pallas/XLA or Triton; NCCL/RCCL/MPI; C++ and Python; compilers (XLA, MLIR, TVM, Triton); quantization and mixed precision. — [Ashby](https://jobs.ashbyhq.com/radixark/0c73a420-3ee0-4b63-9bd5-0a9266d8a727)
- **Anthropic, Performance Engineer, Inference Systems** (2026-05-20). — [Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5224564008)
  - Profiling, roofline analysis and root-cause investigation; Python.
  - Accelerator performance concepts (memory bandwidth, kernel overheads, quantization, collective communication). The posting says "Reasoning about these matters more than having written kernels yourself."
  - Autoscaling, load balancing, routing and tail latency; observability.
- **Anthropic, Engineering Manager, Inference Infrastructure** (2026-09-01): "Experience with LLM inference serving — KV caching, continuous batching, request scheduling, prefill/decode disaggregation", plus cluster schedulers, autoscalers and Kubernetes internals. — [Greenhouse](https://job-boards.greenhouse.io/anthropic/jobs/5411560008)
- **Perplexity, AI Inference Engineer** (2026-04-13). — [Ashby](https://jobs.ashbyhq.com/Perplexity/8a976851-9bef-4b07-8d36-567fa9540aef)
  - Support for new models (weight loading, request scheduling, KV-cache management).
  - Porting CUDA kernels to CuTe DSL for GB200 and Vera Rubin.
  - A Rust-native serving runtime.
  - Profiling "from network ingress through continuous batching and GPU kernel interleaving".
  - Deep CUDA, Triton or CUTLASS experience; PyTorch internals and torch.compile.
- **Together AI, Forward Deployed Engineer – Inference** (2026-05-07 and 2026-08-04, including a Mandarin-speaking role in Singapore). — [Greenhouse](https://job-boards.greenhouse.io/togetherai/jobs/5131941007); [Greenhouse](https://job-boards.greenhouse.io/togetherai/jobs/5199993007)
  - "Expert-level, hands-on experience with inference engines (e.g., vLLM, TensorRT-LLM, SGLang)".
  - KV-cache tuning, speculative decoding, tensor and pipeline parallelism, quantization; Python.
- **Together AI, ML Engineer – Inference** (older, 2024-06-06): Python and PyTorch; preferred knowledge of TGI, vLLM or TensorRT-LLM, speculative decoding, and CUDA/Triton. — [Greenhouse](https://job-boards.greenhouse.io/togetherai/jobs/4385540007)
- **xAI, Software Engineer, Training/Inference (C++)** (posted 2024-10-04 and still listed; a senior variant was posted 2026-11-05). — [Greenhouse](https://job-boards.greenhouse.io/xai/jobs/4533894007)
  - Distributed serving: load balancing, autoscaling, batch scheduling, global KV cache.
  - "1+ years… low-level system programming… C/C++ or Rust".
  - GPU inference engines (vLLM, SGLang, Triton, TensorRT-LLM).
  - Batching, caching and parallelism; GPU kernels and code generation.
- **xAI, Sr. Model Engineer, RL Inference** (2026-07-06): quantization and numerics; experience developing SGLang or vLLM. — [Greenhouse](https://job-boards.greenhouse.io/xai/jobs/5180223007)
- **Databricks**
  - Staff SWE, GenAI inference (2025-10-08): inference internals (attention, MLPs, quantization); CUDA, cuBLAS, cuDNN and NCCL; distributed systems (RPC, batching, sharding); instrumentation and profiling. — [Databricks](https://databricks.com/company/careers/open-positions/job?gh_jid=8202698002)
  - Foundation Model Inference (2026-07-24): OSS contributions to vLLM, SGLang and similar projects. — [Databricks](https://databricks.com/company/careers/open-positions/job?gh_jid=8649279002)
- **Baseten**
  - Model Performance Systems (2026-01-07): GSM8K/MMLU and custom performance suites (long context, KV-cache reuse, disaggregated serving); PyTorch Profiler, Nsight Systems and py-spy; GPU memory subsystems and InfiniBand; Python plus the NVIDIA stack. — [Ashby](https://jobs.ashbyhq.com/baseten/75d7beac-0298-40fa-b206-2e0c0c08a64f)
  - Voice AI Inference Runtime (2026-04-23): p95/p99 latency; dynamic batching; Docker/Kubernetes; vLLM, TensorRT or ONNX. — [Ashby](https://jobs.ashbyhq.com/baseten/6e396eb7-acb3-436a-89ec-05e755c477f2)
- **Modal**
  - MTS Research, Inference (2026-07-05): speculative decoding, disaggregated prefill/decode, FP8/INT4, KV-cache management, autoscaling. The posting asks that you be "fluent in the LLM serving stack, from kernels and quantization up to schedulers and autoscaling". — [Ashby](https://jobs.ashbyhq.com/modal/73c97bbc-8e27-4c5d-b38b-90b3afdb0d93)
  - Inference Runtime (2026-10-05): Linux internals (virtual memory, cgroups, namespaces), container runtimes, Rust/Go, GPU drivers, RDMA. — [Ashby](https://jobs.ashbyhq.com/modal/6195f552-ca6f-4540-b7f6-e7e6054e47ad)
- **Cohere, Staff SWE, Inference Infrastructure** (2026-01-12): Kubernetes with GPU workloads, accelerator characteristics, distributed systems, Go or C++. — [Ashby](https://jobs.ashbyhq.com/cohere/41f23dad-9da2-451a-bd1e-a1800437cb64)
- **OpenAI, Developer Productivity – Inference Runtime:** Python and debugging of distributed systems, plus: "excited to learn about large-scale inference systems, even if you have not worked directly on inference before". — [Ashby](https://jobs.ashbyhq.com/openai/9d48e2e6-41a9-4a90-8b3b-6cc960e95c2f)
- **Meituan LongCat 2027 campus hiring** (reposted from Meituan's official site by an aggregator)
  - 基座大模型推理引擎工程师:
    - Transformer internals (Attention, MoE, Memory).
    - "熟悉主流推理框架（SGLang/vLLM）源码，对PD分离、模型量化、投机推理、调度重叠、前缀缓存等关键技术有实战落地经验".
    - C++/CUDA/AscendC with FlashAttention and quantized-GEMM tuning.
    - RDMA, plus Mooncake, LMCache or Dynamo distributed KV-cache systems.
    - Profiling skills.

    — [niuqizp](https://campus.niuqizp.com/job-vUk5zZ5ta.html)
  - Intern 大模型推理AI工程师: C++/CUDA, data structures and algorithms, Transformer, PyTorch; preferred sglang, vllm, FasterTransformer, DeepSpeed FastGen or TensorRT-LLM. — [niuqizp](https://campus.niuqizp.com/job-vwl5zazMt.html)

#### Practitioner guides
- **The Baseten Inference Engineering book** (free) covers prerequisites through production, with a Recommended Reading appendix (architecture, developer tools, frontier open models, GPU infrastructure, inference optimization research, evaluation). — [llms.txt](https://www.baseten.co/inference-engineering/llms.txt)
- **Chinese 2026 role roadmaps** by rosemary512 and on CSDN rank portfolio items:
  - The 推理框架负责人 roadmap: (1) a mini inference engine, (2) a hand-written GEMM with profiling, (3) a vLLM source-analysis blog series, (4) a PD-disaggregation scheduler prototype.
  - The backend roadmap: an OpenAI-compatible Go gateway as the core project.

  — [gitcode](https://gitcode.csdn.net/6a100a9710ee7a33f274631f.html); [gitcode](https://gitcode.csdn.net/6a10095f10ee7a33f274629f.html)
- **A Pragmatic Engineer article, "What is inference engineering"**, also exists. It was not read and may be partly paywalled. — [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/what-is-inference-engineering?triedRedirect=true)

### Inferences
**Skill matrix for the learner's roadmap** (inferred from the postings above):
- **Must-have core:**
  - Python and PyTorch.
  - LLM inference mechanics: KV cache, prefix caching, continuous batching, scheduling, PD disaggregation, speculative decoding, quantization, TP/PP/EP.
  - Deploying and tuning vLLM or SGLang.
  - Benchmarking and profiling (Nsight Systems, PyTorch Profiler), with roofline reasoning.
- **Branch A, engine/kernel engineer:** PyTorch internals, CUDA, Triton, CUTLASS/CuTe DSL or TileLang, C++, and FlashAttention/GEMM tuning.
- **Branch B, serving platform engineer:** Go, Rust or C++; Kubernetes and operators; autoscaling, routing and load balancing; observability; NCCL/RDMA/InfiniBand; distributed KV-cache systems (Mooncake, LMCache, Dynamo).
- **Branch C, forward-deployed / performance tuning:** engine configuration across hardware and models, plus performance reporting.

**Differentiators** named explicitly in postings:
- Merged vLLM or SGLang pull requests.
- Widely shared technical blog posts.
- Multi-hardware experience: ROCm, TPU/JAX/Pallas, and Ascend in China.

**Entry path.** Platform and performance roles say accelerator reasoning matters more than writing kernels (Anthropic), and some adjacent infrastructure roles accept people with no inference background (OpenAI). A strong backend or distributed-systems engineer can therefore enter through serving infrastructure before going deep on kernels.

**China vs. the US.** Chinese postings emphasize source-level SGLang/vLLM knowledge and C++/CUDA (and AscendC). US postings put more weight on production distributed systems and observability. That matches the Chinese community's source-reading and mini-engine learning culture.

### Gaps
- Primary-source JDs from ByteDance, Alibaba, Moonshot, DeepSeek and Zhipu could not be retrieved, because their career sites are JavaScript-heavy. The Zoom Hangzhou "LLM 推理算法工程师" page returned 404. Chinese evidence is therefore limited to Meituan via an aggregator.
- No statistically representative sample of postings was built. The postings above are a purposive sample of well-known inference employers.
- The member counts and activity of the GPU MODE Discord and the vLLM/SGLang Slacks were not measured.
- WeChat group entry points for vLLM and SGLang China communities were not found. One search snippet mentioned a DingTalk group for a 2026 Shanghai SGLang event, but it was not verified.
- Paid courses seen only in search results were not inspected, including KodeKloud's "AI Infrastructure: llm-d, vLLM and GPUs" notes and interview-prep sites such as techinterview.org. — [KodeKloud](https://notes.kodekloud.com/docs/AI-Infrastructure-LLM-D-vLLM-and-GPUs/Meet-llm-d/Your-Next-Move/page); [techinterview.org](https://www.techinterview.org/post/3233475331/ai-inference-infrastructure-interview-prep-2026-vllm-triton/)
