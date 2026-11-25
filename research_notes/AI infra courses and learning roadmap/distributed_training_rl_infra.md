# Distributed LLM Training and RL Post-Training Infrastructure: Learning Resources (verified 2026-11-24)

> **How the data was gathered (for the report writer).** Everything was checked on 2026-11-24. GitHub's API and HTML were blocked in this environment, so GitHub stars, last-push dates and latest releases come from the **ungh.cc GitHub API mirror** (`https://ungh.cc/repos/{owner}/{repo}` and `/releases/latest`). The mirror also resolves renamed repos. Star counts are a snapshot, so round them as "~". PyPI versions come from `https://pypi.org/pypi/{pkg}/json`. Paper titles, authors and dates come from the **arXiv API** (all 63 IDs queried resolved). Docs and handbook pages were checked with HTTP GETs and returned 200 unless flagged. GitHub repos were confirmed through the mirror. A few links taken from READMEs or search results were **not individually fetched**: the YouTube playlist, LMSYS blog posts listed in READMEs, the DeepSeek-V4 PDF, vendor networking docs, verl-recipe, the AReaL and torchforge docs sites, and the SemiAnalysis post. READMEs came from raw.githubusercontent.com. "Level", "prerequisites" and "why it stands out" are **my assessments** unless quoted. All resources are free and in English unless flagged.

## 1. What are the best end-to-end resources for learning distributed LLM training from first principles?

### Takeaway
Every candidate named in the brief is live. Two 2025 web books form the core curriculum, and they complement each other: Hugging Face's **Ultra-Scale Playbook** (GPU/PyTorch, 5D parallelism, picotron/nanotron code) and Google DeepMind's **How to Scale Your Model** (TPU/JAX-centric roofline and sharding math, plus a GPU chapter). Around them sit the regularly updated **PyTorch distributed tutorials** (DDP/FSDP2/TP/pipelining, all updated 2025), Stas Bekman's continuously maintained **ML Engineering Open Book** (hardware, network, storage, SLURM, fault tolerance), and hands-on **picotron**. Lilian Weng's post (2021, last updated 2022), EleutherAI's Transformer Math 101 (2023) and Sasha Rush's LLM Training Puzzles (static since Jan 2024) are still valuable but dated.

### Cited Findings

#### A. Handbooks and long-form guides
- **The Ultra-Scale Playbook: Training LLMs on GPU Clusters**: https://huggingface.co/spaces/nanotron/ultrascale-playbook
  - Maintainer: Hugging Face (nanotron org). Authors found in the page HTML: Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Thomas Wolf.
  - Published "Feb 19, 2025". The Space was last modified 2025-08-21 and has ~4.15k likes.
  - Format: interactive web book (static HF Space). Language: EN.
  - Top-level sections: High-Level Overview · First Steps: Training on One GPU · Data Parallelism · Tensor Parallelism · Context Parallelism · Pipeline Parallelism · Expert Parallelism · 5D Parallelism in a Nutshell · Finding the Best Training Configuration · Diving into the GPUs – Fusing, Threading, and Mixing · Conclusion · References · Appendix.
  - The page cites "4,000 scaling experiments" and setups up to 512 GPUs. — [Static page](https://nanotron-ultrascale-playbook.static.hf.space/index.html); [HF API metadata](https://huggingface.co/api/spaces/nanotron/ultrascale-playbook)
  - *Assessment:* Level beginner→advanced. Prereqs: PyTorch and Transformer basics. It is the single best end-to-end GPU-centric resource, and its code is runnable in picotron/nanotron.
- **How To Scale Your Model ("A Systems View of LLMs on TPUs")**: https://jax-ml.github.io/scaling-book/
  - Authors: Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, Reiner Pope (Google DeepMind).
  - Published Feb 4, 2025, with no changelog. The page calls itself "a draft and will continue to be revised". The repo `jax-ml/scaling-book` has ~1.5k★ and was last pushed 2026-10-19.
  - Chapters: Rooflines · How to Think About TPUs · Sharded Matrices and How to Multiply Them · All the Transformer Math You Need to Know · How to Parallelize a Transformer for Training · Training LLaMA 3 on TPUs · Transformer Inference · Serving LLaMA 3 on TPUs · Profiling TPU Code · Programming TPUs in JAX · Conclusions · **Ch. 12 "How to Think About GPUs"** (a "bonus"/"new" section).
  - Stated prerequisites: "a basic understanding of LLMs and the Transformer architecture", ideally some JAX. — [Site](https://jax-ml.github.io/scaling-book/); [repo stats](https://ungh.cc/repos/jax-ml/scaling-book)
  - *Assessment:* intermediate. It is the best resource for the *math* of collectives, sharding and rooflines, and it is accelerator-agnostic in spirit.
- **Machine Learning Engineering Open Book** (Stas Bekman): https://github.com/stas00/ml-engineering
  - ~19.7k★, last pushed 2026-11-24, so actively maintained. Format: Markdown book on GitHub.
  - Parts: Insights · Hardware (Compute, Storage, Network) · Orchestration (incl. SLURM) · Training · Inference · Development (Debugging, Testing) · Misc, plus "Lessons Learned from Training LLMs".
  - Verified chapter paths: `training/fault-tolerance`, `training/checkpoints`, `training/model-parallelism`, `training/performance`, `training/dtype.md`, `network/`, `network/benchmarks/`, `storage/`, `orchestration/slurm/`. — [README](https://github.com/stas00/ml-engineering); [stats](https://ungh.cc/repos/stas00/ml-engineering)
  - *Assessment:* intermediate/advanced practitioner's reference. It is the best free coverage of the "unglamorous" cluster side (network, storage, SLURM, debugging hangs).
- **The Smol Training Playbook: The Secrets to Building World-Class LLMs** (Hugging Face, HuggingFaceTB): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
  - Published "Oct. 30, 2025", last modified 2026-07-28, ~3.4k likes.
  - Chapters: Training Compass · Ablations · Architecture · Data Curation · The Training Marathon · Post-Training in 2025 · **Infrastructure—The Unsung Hero**.
  - Concrete numbers include "384 H100s for roughly a month" for 11T tokens at ~30% MFU. The infra chapter discusses NVLink (~786 GB/s bidirectional), NCCL, and NVLS boosting all-reduce "1.3× to 480 GB/s". — [Page](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook); [HF API](https://huggingface.co/api/spaces/HuggingFaceTB/smol-training-playbook)
  - *Assessment:* a companion to the Ultra-Scale Playbook that shows what a real run looks like.
- **How to Train Really Large Models on Many GPUs?** (Lilian Weng): https://lilianweng.github.io/posts/2021-09-25-train-large/
  - Blog post from 2021-09. In-page notes say "[Updated on 2022-03-13: add expert choice routing.]" and "[Updated on 2022-06-10]". — [Post](https://lilianweng.github.io/posts/2021-09-25-train-large/)
  - **Flag: dated.** It covers no FSDP2, context parallelism, FP8/FP4 or DualPipe. It is still a concise taxonomy of DP/PP/TP/MoE/memory-saving methods for beginners.
- **Transformer Math 101** (EleutherAI): https://blog.eleuther.ai/transformer-math/
  - Authors: Quentin Anthony, Stella Biderman, Hailey Schoelkopf. Dated Apr 18, 2023. Blog post. — [Post](https://blog.eleuther.ai/transformer-math/)
  - *Assessment:* beginner. It remains the canonical back-of-envelope reference for compute (6ND) and memory (weights, optimizer, activations, ZeRO). **Flag: 2023.** It predates FP8/MoE-era practice.

#### B. Official tutorials (PyTorch)
- **PyTorch Distributed overview**: https://docs.pytorch.org/tutorials/beginner/dist_overview.html (live). PyTorch latest is 2.15.0 (PyPI, 2026-10-28). — [PyPI](https://pypi.org/pypi/torch/json)
- **DDP tutorial**: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html. Author Shen Li, edited by Joe Zhu and Chirag Pandya. Created Apr 23, 2019, last updated Sep 23, 2025. — [Page](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **Getting Started with Fully Sharded Data Parallel (FSDP2)**: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
  - Authors: Wei Feng, Will Constable, Yifan Mao. Last updated Sep 02, 2025.
  - Covers `fully_shard`/DTensor sharding, implicit and explicit prefetching, `MixedPrecisionPolicy` (bf16 params, fp32 reduce), DCP checkpointing, and an FSDP1→FSDP2 migration guide.
  - States "FSDP1 is deprecated". — [Page](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- **Large Scale Transformer model training with Tensor Parallel (TP)**: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html. Authors Wanchao Liang and Tianyu Liu. Created Apr 19, 2024, last updated Jul 18, 2025. — [Page](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- **Distributed Pipeline Parallelism (torch.distributed.pipelining)**: https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html. Author Howard Huang. Created Jul 09, 2024, last updated Nov 05, 2025. — [Page](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)
- Repo `pytorch/tutorials`: ~9.5k★, last pushed 2026-11-06. — [stats](https://ungh.cc/repos/pytorch/tutorials)
- *Assessment:* beginner→intermediate. These are the most current "official" hands-on path, and they map 1:1 onto torchtitan's building blocks.
- **JAX equivalent:** "Distributed arrays and automatic parallelization" tutorial (live), https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html. JAX latest is 0.12.1 (2026-11-18). — [PyPI](https://pypi.org/pypi/jax/json)

#### C. Minimal educational codebases and exercises
- **picotron** (Hugging Face): https://github.com/huggingface/picotron
  - ~2.35k★. Last pushed 2025-08-26, no releases, not on PyPI.
  - "The minimalist & most-hackable repository for pre-training Llama-like models with 4D Parallelism (Data, Tensor, Pipeline, Context parallel)". `train.py`, `model.py` and `[data|tensor|pipeline|context]_parallel.py` are "all under **300** lines of code".
  - Reports "38% MFU on a LLaMA-2-7B model using 64 H100 GPUs" and "nearly 50% MFU on the SmolLM-1.7B model with 8 H100 GPUs". It is "primarily for educational purposes". — [README](https://github.com/huggingface/picotron); [stats](https://ungh.cc/repos/huggingface/picotron)
  - **Flag:** no commits since Aug 2025, effectively feature-frozen. That is fine for learning.
- **picotron_tutorial**: https://github.com/huggingface/picotron_tutorial
  - ~268★, last pushed 2025-11-24. A step-by-step "build Picotron from scratch" companion with a YouTube video series: Part 1 Model/Process Group Manager/Dataloader, Part 2 Tensor Parallel, and more. Playlist: https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S — [README](https://github.com/huggingface/picotron_tutorial)
  - *Assessment:* the best "implement every parallelism yourself" path, at beginner→intermediate level.
- **LLM Training Puzzles** (Sasha Rush): https://github.com/srush/LLM-Training-Puzzles
  - ~1.2k★, last pushed 2024-01-10. Repo tagline: "What would you do with 1000 H100s...". Notebook puzzles. — [stats](https://ungh.cc/repos/srush/LLM-Training-Puzzles)
  - **Flag:** static since Jan 2024, but the concepts are timeless. Sibling repo GPU-Puzzles has ~12.6k★.
- **nanotron**: https://github.com/huggingface/nanotron
  - ~2.9k★, last pushed 2026-10-08. The latest tagged release and PyPI version is **v0.4 (Mar 2024)**. The README points to the Ultra-Scale Playbook and to AI-generated DeepWiki docs. — [README](https://github.com/huggingface/nanotron); [PyPI](https://pypi.org/pypi/nanotron/json)
  - *Assessment:* the "next step up" from picotron (3D parallelism, production-ish). **Flag:** no tagged releases since 2024.
- **nanochat** (Karpathy): https://github.com/karpathy/nanochat. ~60.8k★, last pushed 2026-09-07, "The best ChatGPT that $100 can buy". It is single-node end-to-end and useful context, but not a parallelism tutor. — [stats](https://ungh.cc/repos/karpathy/nanochat)
- **Meta Lingua**: https://github.com/facebookresearch/lingua. ~4.8k★, **last pushed 2025-07-18**. **Flag: appears inactive.** — [stats](https://ungh.cc/repos/facebookresearch/lingua)

#### D. Community notes
- **Awesome-ML-SYS-Tutorial** (Chenyang Zhao, `zhaochenyang20`): https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial
  - ~7.7k★, last pushed 2026-10-21. Available in **English and Chinese**, with many notes mirrored on Zhihu in Chinese.
  - Sections: RLHF System Development Notes (slime, AReaL, System Design, verl, OpenRLHF, Algorithms) · SGLang Learning Notes · Scheduling and Routing · ML System Fundamentals (Transformers, CUDA & GPU, **Distributed Training & Communication**, Quantization). — [README](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)

### Inferences
- The Ultra-Scale Playbook and How to Scale Your Model do different jobs. The Playbook teaches *what each parallelism does and how to implement it on GPUs*. The scaling book teaches *how to reason quantitatively* about whether a given sharding is compute-, memory- or communication-bound. A learner should read both.
- For someone heading toward vLLM/SGLang serving, the most transferable material sits in four places: memory accounting (Transformer Math 101, scaling book ch. 4), collectives (scaling book ch. 3), TP/EP (Playbook) and Stas Bekman's network chapter. These ideas reappear directly in inference-time TP/EP and disaggregation.

#### Suggested learning order (synthesis; my recommendation)
1. **Accounting basics (1 week):** Transformer Math 101 → scaling book ch. 1 (rooflines) and ch. 4 (transformer math) → Playbook "First Steps: Training on One GPU".
2. **Collectives and data parallelism (1–2 weeks):** scaling book ch. 3 → PyTorch DDP tutorial → NCCL User Guide basics and nccl-tests (see Q5) → Playbook "Data Parallelism" (ZeRO-1/2/3) → ZeRO paper → FSDP2 tutorial.
3. **Model parallelism (2–3 weeks):** Playbook TP/CP/PP/EP sections → PyTorch TP and pipelining tutorials → canonical papers (Q3) → LLM Training Puzzles as exercises → **implement it yourself with picotron + picotron_tutorial videos**.
4. **Low precision and kernels:** Playbook "Diving into the GPUs" → mixed-precision/FP8/MXFP8/NVFP4 papers (Q3) → Transformer Engine / torchao.
5. **Production frameworks (Q2):** read **torchtitan** first (PyTorch-native, closest to the tutorials) → Megatron Core → DeepSpeed (ZeRO/offload) → MaxText with the scaling book (JAX/TPU view).
6. **Cluster and reliability (Q5):** Stas Bekman's network/storage/SLURM/fault-tolerance chapters → Smol Training Playbook infra chapter → Llama 3 §3 → ByteDance/Meta reliability papers → schedulers.
7. **RL post-training infra (Q4):** Anyscale overview (2025) → HF "Keep the Tokens Flowing" (2026) → TRL + vLLM RLHF docs → HybridFlow/verl → slime/Miles (SGLang-native) → AReaL/PipelineRL/prime-rl (async).
- Cross-reference for the report writer: university courses and GPU/CUDA fundamentals are covered by other researchers and should be slotted in before step 4.

### Gaps
- I could not confirm whether the Ultra-Scale Playbook has an official PDF or printed edition. The page HTML contains "PDF" strings and editorial comments about a "book version", but nothing conclusive.
- The full author list of the Ultra-Scale Playbook may include more names than the six I found by text search (for example Leandro von Werra, who is not found in the HTML). I could not verify the Smol Training Playbook's author list either.
- The scaling book publishes no changelog, so I cannot date when the GPU chapter was added beyond the fact that it is labeled "new".

## 2. Which production training frameworks should a learner study, and what docs, papers and tutorials exist for each?

### Takeaway
All the major frameworks are active as of Nov 2026: Megatron-Core v0.20, DeepSpeed 0.20, torchtitan v0.4, MaxText v0.2.6, Transformer Engine 2.21. The landscape has been **restructured**, though. NVIDIA's `NVIDIA/NeMo` repo now redirects to **`NVIDIA-NeMo/Speech`** (LLM work moved to Megatron-Bridge and AutoModel), and **Megatron-Bridge is being merged into Megatron-LM** (Bridge repo freeze Nov 30, 2026; full migration planned for Feb 2027). torchtune is no longer maintained. For a learner, **torchtitan** is the most readable production-grade codebase. **Megatron-Core** is the performance and MoE reference, **DeepSpeed** the ZeRO/offload reference, and **MaxText** the JAX/TPU reference.

### Cited Findings
- **Megatron-LM / Megatron Core** (NVIDIA): https://github.com/NVIDIA/Megatron-LM
  - ~18.5k★, last pushed 2026-11-23. Release `core_v0.20.0` (2026-10-29). PyPI `megatron-core` 0.20.1 (2026-11-19).
  - The README describes Megatron-LM as "a reference example ... ideal for research teams, **learning distributed training**, and quick experimentation". Megatron Core is "a composable library" with TP, PP, DP, EP, CP and FP16/BF16/FP8/FP4.
  - 2026 README news:
    - Megatron Bridge will merge into Megatron-LM (2026/11).
    - "Scale Bitwise-Deterministic Pretraining with NVIDIA Megatron Core" blog (2026/10).
    - Core 0.19 (2026/08).
    - DeepSeek-V3 pretraining on GB300 NVL72 at "1,648 TFLOPs/s per GPU" (2026/07).
    - DeepSeek-V4 initial support (2026/05).
    - Muon and other emerging optimizers (2026/04).
  - Docs: Megatron Core developer guide, https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html (live). — [README](https://github.com/NVIDIA/Megatron-LM); [stats](https://ungh.cc/repos/NVIDIA/Megatron-LM); [PyPI](https://pypi.org/pypi/megatron-core/json)
  - Key papers: Megatron-LM (arXiv 1909.08053) · Efficient Large-Scale LM Training on GPU Clusters (PTD-P, interleaved 1F1B; 2104.04473) · Reducing Activation Recomputation (sequence parallelism + selective recompute; 2205.05198) · MoE Parallel Folding (2504.14960, v3 Mar 2026). — [arXiv](https://arxiv.org/abs/1909.08053); [arXiv](https://arxiv.org/abs/2104.04473); [arXiv](https://arxiv.org/abs/2205.05198); [arXiv](https://arxiv.org/abs/2504.14960)
- **Unified Megatron-LM plan**:
  - Bridge training code consolidates into Megatron-LM's training loop, with "Copy the rest of Megatron-Bridge into Megatron-LM" under `megatron/bridge`.
  - Timeline: Megatron-Bridge repo freezes **Nov 30, 2026**. Core 0.21.0 (Dec 2026) unifies the training loop. Core 0.22.0 (Feb 2027) completes the migration. `pip install megatron-lm` will install Core, Bridge and Training. — [NVIDIA docs](https://docs.nvidia.com/nemo/megatron-bridge/nightly/unified-megatron.html)
- **Megatron-Bridge**: https://github.com/NVIDIA-NeMo/Megatron-Bridge
  - ~1.0k★, v0.7.1 (2026-11-20). HF↔Megatron bidirectional conversion and training recipes. Docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/ — [stats](https://ungh.cc/repos/NVIDIA-NeMo/Megatron-Bridge)
  - **Flag:** the repo freezes Nov 30, 2026 and moves into Megatron-LM.
- **NeMo**: `NVIDIA/NeMo` now resolves to `NVIDIA-NeMo/Speech` (~18.8k★, v3.0.0 2026-08-07).
  - README: "2026: This repo has pivoted to focus on audio, speech, and multimodal LLMs. For the final pre-split NeMo release ... see v2.7.3." The NeMo Framework user guide is still live at https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html. — [README](https://github.com/NVIDIA-NeMo/Speech); [stats](https://ungh.cc/repos/NVIDIA/NeMo)
  - **Flag for learners:** pre-2026 NeMo 2.0 LLM tutorials are effectively legacy.
- **NeMo AutoModel**: https://github.com/NVIDIA-NeMo/Automodel. ~1.1k★, v0.7.0 (2026-10-21). "Pytorch Distributed native training library for LLMs/VLMs with OO..." (the repo description is truncated in the mirror). Docs: https://docs.nvidia.com/nemo/automodel/latest/ — [stats](https://ungh.cc/repos/NVIDIA-NeMo/Automodel)
- **DeepSpeed**: https://github.com/deepspeedai/DeepSpeed (moved from `microsoft/`)
  - ~43.4k★, last pushed 2026-11-24. GitHub release v0.19.9 (2026-10-29), **PyPI 0.20.1 (2026-11-24)**.
  - 2026 news: Muon optimizer (2026/05) · SDMA offload of ZeRO-3 collectives on AMD (2026/05) · ASPLOS 2026 tutorial "Building Efficient Large-Scale Model Systems with DeepSpeed" (2026/03) · SuperOffload honorable mention at ASPLOS 2026 · "Core API updates: PyTorch-style backward and low-precision master states" (2025/12).
  - Monthly public office hours are hosted via the Linux Foundation's Zoom. Tutorials: https://www.deepspeed.ai/tutorials/ (live). — [README](https://github.com/deepspeedai/DeepSpeed); [PyPI](https://pypi.org/pypi/deepspeed/json)
  - Papers: ZeRO (1910.02054) · ZeRO-Offload (2101.06840) · ZeRO-Infinity (2104.07857) · ZeRO++ (2306.10209) · DeepSpeed Ulysses (2309.14509) · DeepSpeed-MoE (2201.05596). — [arXiv](https://arxiv.org/abs/1910.02054); [arXiv](https://arxiv.org/abs/2306.10209); [arXiv](https://arxiv.org/abs/2309.14509)
- **torchtitan** (PyTorch): https://github.com/pytorch/torchtitan
  - ~6.0k★, last pushed 2026-11-23. v0.4.0 (2026-10-28, also on PyPI).
  - Self-description: "a clean-room, PyTorch-native implementation of large-scale training. One readable codebase takes a model from pretraining through supervised fine-tuning to reinforcement learning, composing FSDP, tensor, pipeline, context, and expert parallelism without rewriting the model."
  - Features with per-topic docs:
    - FSDP2/HSDP/DDP (`torchtitan/distributed/FSDP.md`).
    - PP schedules 1F1B, Interleaved 1F1B and GPipe (`PIPELINE_PARALLEL.md`).
    - CP via all-gather KV and Ulysses.
    - EP with DeepEP/HybridEP, and Dist-MoE on Blackwell.
    - Full, selective and regional activation checkpointing (`ACTIVATION_REMAT.md`), plus activation offloading.
    - Regional `torch.compile` and CUDA graphs.
    - **MXFP8 and NVFP4 training**.
    - Distributed checkpointing with async saves.
    - Flight Recorder debugging.
  - News: **TitanRL** (2026/08), "a hackable RL stack" reusing TorchTitan models with vLLM generation and batch-invariant mode · AMD fork (2025/11) · SkyPilot tutorial (2025/10).
  - Paper: TorchTitan (ICLR 2025; arXiv 2410.06511, v3 Jun 2025). — [README](https://github.com/pytorch/torchtitan); [stats](https://ungh.cc/repos/pytorch/torchtitan); [arXiv](https://arxiv.org/abs/2410.06511)
- **MaxText** (Google, JAX): https://github.com/AI-Hypercomputer/maxtext
  - ~2.5k★, last pushed 2026-11-22. `maxtext-v0.2.6` (2026-11-11; PyPI 0.2.6). Docs: https://maxtext.readthedocs.io/en/latest/ (live).
  - README news: DeepSeek V4 Pro support (Oct 23, 2026), and a forthcoming simplified "m3 (Modern MaxText Models)" stack where "each model is self-contained and easy to read" (Nov 19, 2026). — [README](https://github.com/AI-Hypercomputer/maxtext); [stats](https://ungh.cc/repos/AI-Hypercomputer/maxtext)
- **nanotron**: see Q1. It is HF's production-ish 3D-parallel trainer and the codebase behind the Playbook. — [README](https://github.com/huggingface/nanotron)
- **Supporting libraries:**
  - **Transformer Engine** (FP8/FP4 kernels): ~3.6k★, v2.21.0 (2026-11-10). https://github.com/NVIDIA/TransformerEngine
  - **torchao** ("PyTorch native quantization for training and inference"): ~3.0k★, v0.18.0 (2026-08-03). https://github.com/pytorch/ao
  - **HF Accelerate**: ~10k★, v1.16.0 (2026-11-16). Docs: https://huggingface.co/docs/accelerate/index
  - **MS-AMP** (FP8-LM's library): `Azure/MS-AMP`, ~0.6k★, last pushed 2025-12-01. **Flag: low activity.**
  - — [stats](https://ungh.cc/repos/NVIDIA/TransformerEngine); [stats](https://ungh.cc/repos/pytorch/ao); [stats](https://ungh.cc/repos/huggingface/accelerate); [stats](https://ungh.cc/repos/Azure/MS-AMP)
- **Other frameworks (lower priority for learners):**
  - EleutherAI **GPT-NeoX**: ~7.5k★, pushed 2026-11-21, but the last tagged release is v2.0 (2023-03).
  - **ColossalAI**: ~41.4k★, v0.5.0 (2025-06).
  - **LLM Foundry**: ~4.5k★, last pushed 2026-03-25, last release 2025-07. **Flag: slowing.**
  - **Pai-Megatron-Patch** (Alibaba): ~1.6k★, last pushed 2025-12-15. **Flag: slowing.**
  - **veScale** (ByteDance, "PyTorch Distributed for Hyperscale Training of LLMs and RLs"): ~1.0k★, pushed 2026-11-11.
  - **AXLearn** (Apple): ~2.4k★, pushed 2026-07-08.
  - — [stats](https://ungh.cc/repos/EleutherAI/gpt-neox); [stats](https://ungh.cc/repos/hpcaitech/ColossalAI); [stats](https://ungh.cc/repos/mosaicml/llm-foundry); [stats](https://ungh.cc/repos/alibaba/Pai-Megatron-Patch); [stats](https://ungh.cc/repos/volcengine/veScale); [stats](https://ungh.cc/repos/apple/axlearn)
- **torchtune**: `meta-pytorch/torchtune`, ~5.8k★, last release v0.6.1 (2025-04-07). README: "⚠️ **Torchtune is no longer actively maintained:** torchtune development wound down in 2025" (see issue #2883). — [README](https://github.com/meta-pytorch/torchtune)
  - **Flag: unmaintained. Do not recommend.**

### Inferences
- **Best study order for frameworks:**
  1. torchtitan. Its feature docs map directly to the PyTorch tutorials, it has MXFP8/NVFP4, EP and fault-tolerance hooks, and it now extends into RL via TitanRL.
  2. Megatron Core. It is the de facto reference for maximum-MFU dense and MoE training, and the backend of slime, Miles, ROLL and NeMo RL.
  3. DeepSpeed. Read it for ZeRO and offload concepts; OpenRLHF still uses it.
  4. MaxText, read alongside the scaling book.
- Learners should **avoid starting from NeMo 2.0 LLM tutorials**. NVIDIA's own LLM path in 2026 is Megatron-LM (absorbing Bridge) plus AutoModel.
- Version churn is high: Megatron Core shipped 0.19 in Aug and 0.20 in Oct 2026, and DeepSpeed released again today. Notes should link to docs "latest" rather than pin versions.

### Gaps
- I did not verify torchtitan's or Megatron's official documentation sites beyond the README and doc paths (torchtitan has no separate docs site, as far as the README shows).
- NeMo AutoModel's internal parallelism design (DTensor-based?) was not verified.
- NVIDIA's "NeMo 2.0 → Megatron Bridge Migration Guide" exists in the docs sidebar but was not read.

## 3. Canonical papers and best explainers per technique, and what's new in 2025–2026

### Takeaway
The canonical paper set is stable. It runs ZeRO → Megatron TP → sequence parallelism with selective recompute, GPipe → PipeDream/1F1B → interleaved → zero-bubble → DualPipe, ring attention, Ulysses and USP for context, GShard/Switch → DeepSpeed-MoE for experts, and mixed precision → FP8 formats. The 2025–2026 frontier is defined by five things:
- DeepSeek-V3's stack: no TP, 16-way PP with DualPipe, 64-way EP, ZeRO-1, FP8.
- Block-scaled low precision: MXFP8 recipes and NVFP4 pretraining (2025), adopted in torchtitan, Megatron and Transformer Engine.
- MoE-specific parallel mappings and communication: Parallel Folding, DeepEP.
- Muon-optimizer support in big frameworks.
- Reliability and determinism engineering at 10k–100k GPU scale.
DeepSeek-V4 (Apr 2026) adds FP4 QAT for experts and CP for compressed attention, but I could only verify it through secondary notes.

### Cited Findings

#### Canonical papers (all verified via the arXiv API)
- **Data parallelism and sharding:**
  - ZeRO: Rajbhandari et al., 2019, arXiv 1910.02054.
  - ZeRO-Offload: 2101.06840. ZeRO-Infinity: 2104.07857. ZeRO++: 2306.10209.
  - PyTorch FSDP: Zhao et al., 2023, arXiv 2304.11277.
  - — [ZeRO](https://arxiv.org/abs/1910.02054); [ZeRO++](https://arxiv.org/abs/2306.10209); [FSDP](https://arxiv.org/abs/2304.11277)
- **Tensor and sequence parallelism:**
  - Megatron-LM: Shoeybi et al., 2019, 1909.08053.
  - "Reducing Activation Recomputation in Large Transformer Models": Korthikanti et al., 2022, 2205.05198. This introduced Megatron sequence parallelism and selective recompute.
  - Colossal-AI's "Sequence Parallelism: Long Sequence Training from System Perspective": Li et al., 2105.13120.
  - — [arXiv](https://arxiv.org/abs/1909.08053); [arXiv](https://arxiv.org/abs/2205.05198); [arXiv](https://arxiv.org/abs/2105.13120)
- **Pipeline parallelism:**
  - GPipe: Huang et al., 1811.06965.
  - PipeDream: Harlap et al., 1806.03377.
  - PipeDream-2BW: "Memory-Efficient Pipeline-Parallel DNN Training", 2006.09503.
  - PTD-P / interleaved 1F1B: Narayanan et al., 2104.04473.
  - Zero Bubble PP: Qi et al., 2401.10241; code `sail-sg/zero-bubble-pipeline-parallelism`, ~0.5k★, last pushed 2025-05-07.
  - DualPipe: DeepSeek-V3 report 2412.19437; repo `deepseek-ai/DualPipe`, ~3.0k★, last pushed 2026-01-14. The README calls it a "bidirectional pipeline parallelism algorithm ... achiev[ing] full overlap of forward and backward computation-communication phases" and includes schedule diagrams.
  - — [GPipe](https://arxiv.org/abs/1811.06965); [ZB-PP](https://arxiv.org/abs/2401.10241); [DualPipe README](https://github.com/deepseek-ai/DualPipe); [stats](https://ungh.cc/repos/deepseek-ai/DualPipe)
- **Context and sequence parallelism for long context:**
  - Ring Attention: Liu, Zaharia, Abbeel, 2310.01889.
  - DeepSpeed Ulysses: 2309.14509.
  - USP (Unified Sequence Parallelism): Fang & Zhao, 2405.07719. Code `feifeibear/long-context-attention`, ~0.7k★, pushed 2026-09-16.
  - `zhuzilin/ring-flash-attention`: ~1.1k★, last pushed 2025-09-10.
  - — [arXiv](https://arxiv.org/abs/2310.01889); [arXiv](https://arxiv.org/abs/2405.07719); [stats](https://ungh.cc/repos/feifeibear/long-context-attention); [stats](https://ungh.cc/repos/zhuzilin/ring-flash-attention)
- **MoE and expert parallelism:**
  - GShard: 2006.16668. Switch Transformers: 2101.03961. DeepSpeed-MoE: 2201.05596.
  - MoE Parallel Folding (Megatron Core): 2504.14960.
  - DeepEP ("an efficient expert-parallel communication library"): ~10.5k★, v1.2.1 (2025-09), pushed 2026-11-20.
  - EPLB: ~1.5k★, static since 2025-03.
  - — [GShard](https://arxiv.org/abs/2006.16668); [Switch](https://arxiv.org/abs/2101.03961); [Folding](https://arxiv.org/abs/2504.14960); [DeepEP stats](https://ungh.cc/repos/deepseek-ai/DeepEP)
- **Activation checkpointing:** "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016, 1604.06174) and the Korthikanti 2022 selective-recompute paper. The torchtitan docs cover full, selective and regional AC. — [arXiv](https://arxiv.org/abs/1604.06174); [torchtitan README](https://github.com/pytorch/torchtitan)
- **Mixed and low precision:**
  - Mixed Precision Training: Micikevicius et al., 1710.03740.
  - FP8 Formats for Deep Learning: 2209.05433.
  - FP8-LM: 2310.18313.
  - **Recipes for Pre-training LLMs with MXFP8**: Mishra et al., 2506.08027, v2 Aug 2025.
  - **Pretraining Large Language Models with NVFP4**: NVIDIA, 2509.25149, v2 Mar 2026.
  - — [arXiv](https://arxiv.org/abs/1710.03740); [arXiv](https://arxiv.org/abs/2209.05433); [arXiv](https://arxiv.org/abs/2506.08027); [arXiv](https://arxiv.org/abs/2509.25149)

#### Best explainers per technique
- **Ultra-Scale Playbook** has one section each for DP (incl. ZeRO), TP (incl. SP), CP, PP, EP, "5D in a nutshell", configuration search, and fused kernels/mixed precision. — [Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html)
- **How to Scale Your Model**: ch. 3 (sharded matmuls/collectives) and ch. 5 (how to parallelize a transformer for training) give the cost models. — [Scaling book](https://jax-ml.github.io/scaling-book/)
- **PyTorch tutorials:** FSDP2, TP and pipelining (see Q1). **Stas Bekman's** `training/model-parallelism` and `training/dtype.md`. — [ML Eng book](https://github.com/stas00/ml-engineering)
- **Awesome-ML-SYS-Tutorial**: "Deep Dive into DeepSeek MoE with Classic Secondary Development of EP on FSDP", plus FSDP and Megatron notes in an RL context. — [README](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)

#### New in 2025–2026
- **DeepSeek-V3 training stack**, verified in the report text:
  - "trained on a cluster equipped with 2048 NVIDIA H800 GPUs" (NVLink/NVSwitch within nodes, InfiniBand across nodes).
  - "requires only 2.788M H800 GPU hours for its full training".
  - Uses "16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)" in its in-house HAI-LLM framework.
  - Memory optimizations allowed training "without using costly Tensor Parallelism (TP)".
  - Introduces "an FP8 mixed precision training framework and, for the first time, validate[s] its effectiveness on an extremely large-scale model".
  - — [DeepSeek-V3 report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- **DeepSeek hardware reflections:** "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures" (Zhao et al., arXiv 2505.09343, v2 Dec 2025). — [arXiv](https://arxiv.org/abs/2505.09343)
- **DeepSeek Feb 2025 open-source infra repos** (good study artifacts):
  - DualPipe; DeepEP; DeepGEMM (~9.2k★, v2.1.1.post3, 2025-10).
  - 3FS distributed file system (~10.5k★, pushed 2026-11-17).
  - EPLB, and `profile-data` ("Analyze computation-communication overlap in V3/R1", static since 2025-03).
  - — [3FS](https://ungh.cc/repos/deepseek-ai/3FS); [DeepGEMM](https://ungh.cc/repos/deepseek-ai/DeepGEMM); [profile-data](https://ungh.cc/repos/deepseek-ai/profile-data)
- **DeepSeek-V4 (Apr 2026)**, from secondary sources only. The primary report is "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (PDF on HF: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf). A reading-notes blog (Apr 25, 2026) describes:
  - MXFP4 expert weights with FP32 master weights quantized to FP4 (E2M1) and dequantized to FP8 (E4M3) for compute (QAT with STE).
  - Muon with hybrid ZeRO bucket assignment.
  - Context parallelism adapted to compressed attention.
  - TileLang kernels, and EP "waves" with "up to a 1.96× speedup" in RL rollouts.
  - — [Gradients & Tides notes](https://jianyuh.github.io/llm/2026/04/25/DeepSeek-V4-Infra.html); [HF blog](https://huggingface.co/blog/deepseekv4)
  - **Flag: unverified against the primary PDF.**
- **Llama 3 (2024)** is the reference for 4D parallelism at 16K GPUs: "we use 4D parallelism", trained "on up to 16K H100 GPUs". — [Llama 3 paper (2407.21783)](https://arxiv.org/abs/2407.21783)
- **Framework adoption of new precisions and optimizers:**
  - torchtitan supports MXFP8 and NVFP4 training.
  - Megatron Core lists FP8 and FP4, and added Muon (Apr 2026).
  - DeepSpeed added Muon (May 2026).
  - Megatron published bitwise-deterministic pretraining guidance (Oct 2026).
  - — [torchtitan](https://github.com/pytorch/torchtitan); [Megatron-LM](https://github.com/NVIDIA/Megatron-LM); [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- **Large-scale MoE throughput:** Megatron Core reports DeepSeek-V3 671B pretraining on GB300 NVL72 at "1,648 TFLOPs/s per GPU" (NVIDIA blog, 2026/07, via the README). — [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM)
- **Fault tolerance at 10k+ GPU scale:** see Q5 (ByteRobust, FlashRecovery, Fault-Tolerant HSDP on 100,000 GPUs, Llama 3 interruption statistics).

### Inferences
- A 2026 learner should treat **"no-TP, PP+EP+ZeRO-1 with comm/compute overlap"** (the DeepSeek pattern) and **"FSDP2 + TP + PP + CP" (4D, the Llama 3/torchtitan pattern)** as the two reference designs to compare.
- MXFP8/NVFP4 block-scaled formats are the active low-precision frontier. Most of these learning materials are papers and framework docs, not tutorials.
- There is a clear **2026 trend toward determinism**: Megatron's bitwise-deterministic pretraining, TitanRL's batch-invariant mode, and the Thinking Machines nondeterminism post (Q4). This matters for RL, where training and rollout numerics must match.

### Gaps
- The DeepSeek-V4 primary report was not read. Parallelism degrees, cluster size, GPU hours and any DualPipe successor are unverified. Secondary sources also disagree in detail: one reports 32T/33T pretraining tokens for V4-Flash/Pro.
- I found no single up-to-date explainer that compares all PP schedules (1F1B, interleaved, ZB, DualPipe/DualPipeV) side by side beyond the Playbook's PP section and the DualPipe README. That should be verified.
- I found no tutorial-style FP4 training resource. Only papers and framework docs exist.

## 4. RL post-training infrastructure: architectures, use of vLLM/SGLang, docs, and which framework is best for learners

### Takeaway
The field has converged on **separating rollout (inference engine: vLLM or SGLang) from training (FSDP2/Megatron/DeepSpeed)**. Designs differ on two axes:
- **Placement:** colocated with time-sharing and resharding (verl's 3D-HybridEngine), or disaggregated pools.
- **Synchrony:** synchronous, one-step-off, or fully asynchronous with staleness control (AReaL, PipelineRL, prime-rl, slime/Miles).
NCCL broadcast is the default weight-sync mechanism. Two comparative overviews are the best starting points: Anyscale's (Jul/Sep 2025) and the more current **Hugging Face "Keep the Tokens Flowing" (Mar 10, 2026, 16 libraries × 7 axes)**. My recommendation for a learner aiming at SGLang/vLLM serving:
1. Learn the loop with **TRL** or **verifiers/prime-rl**.
2. Study **verl** (most popular, with a paper, docs and community code walkthroughs).
3. Study **slime/Miles** (lean Megatron+SGLang).
4. Read **AReaL/PipelineRL** for async.

### Cited Findings

#### Comparative overviews and explainers
- **"Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries"**: https://huggingface.co/blog/async-rl-training-landscape
  - Hugging Face blog, March 10, 2026. Authors: Amine Dirhoussi, Quentin Gallouédec, Kashif Rasul, Lewis Tunstall, Edward Beeching, Albert Villanova del Moral, Nouamane Tazi, Leandro von Werra, Sergio Paniego.
  - Libraries surveyed: AReaL, ART, Atropos, MILES, NeMo-RL, OAT, open-instruct, PipelineRL, PRIME-RL, ROLL, SkyRL, SLIME, TorchForge, Tunix, verl, verifiers-rl.
  - "Every library in this survey has independently converged on the same architectural principle" (separating inference and training). In colocated mode "inference and training cannot overlap", while disaggregated mode lets them "run concurrently".
  - Findings by axis:
    - Orchestration: Ray in 8 of 16 libraries. Native Python in PRIME-RL, PipelineRL and AReaL. Monarch in TorchForge. HTTP microservices in Atropos.
    - Weight sync: "NCCL ... broadcast is the default"; alternatives are CUDA IPC, filesystem, HTTP and JAX cross-mesh.
    - Staleness: version rejection, depth bounding, or importance-sampling correction, trending toward hybrids.
    - Partial rollouts: implicit continuation (PipelineRL), abort and retry (SkyRL, SLIME), save and resume (verl).
    - LoRA: 8 of 13 applicable libraries sync adapter-only deltas.
    - Backends: FSDP2, DeepSpeed, Megatron, JAX. "Distributed MoE support is the emerging differentiator."
  - The post was written to guide a new async trainer for TRL. — [HF blog](https://huggingface.co/blog/async-rl-training-landscape)
- **"Open Source RL Libraries for LLMs"** (Anyscale): https://www.anyscale.com/blog/open-source-rl-libraries-for-llms
  - Tyler Griggs and Philipp Moritz. July 1, 2025, updated Sep 1, 2025. Compares TRL, verl, OpenRLHF, RAGEN, AReaL, Verifiers, ROLL, NeMo-RL, SkyRL, slime.
  - Colocation "can help reduce GPU usage, but it is less flexible".
  - Recommendations: verl is "arguably a good choice" for large-scale performance-critical work. TRL suits "text-based RL ... without environment interactions". Verifiers suits researchers who "value simplicity and flexibility the most".
  - **Flag: partly outdated.** It marked verl's async support as missing or in progress, but verl now ships fully-async and one-step-off trainers in `verl-recipe` (see below). — [Anyscale](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)
- **Awesome-ML-SYS-Tutorial, "RLHF System Development Notes"** (EN + Chinese):
  - **slime:** source code walkthrough; FSDP2 backend; unified FP8 for MoE RL; INT4 QAT RL ("Squeezing 1TB Model Rollout into a Single H200"); speculative decoding in RL; colocate vs disaggregated setup guide.
  - **AReaL:** code walkthrough.
  - **System design:** "Deep Thoughts on RL Systems: In-Depth Understanding of Weight Update Mechanism", plus FSDP and Megatron notes and EP on FSDP.
  - **verl:** source code walkthroughs (initialization, rollout, AgentLoop, parameter reference), multi-turn profiling, and the server-based rollout design.
  - **OpenRLHF:** notes on integrating SGLang into OpenRLHF.
  - The author warns that "A large number of RL conclusions derived from papers are based on RL infrastructure in the open-source community that may be extremely flawed." — [README](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)
- **vLLM RLHF docs** (live): https://docs.vllm.ai/en/latest/training/rlhf/ — [page](https://docs.vllm.ai/en/latest/training/rlhf/)
- **"Defeating Nondeterminism in LLM Inference"** (Horace He, Thinking Machines Lab, Sep 10, 2025): https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/. It underpins the batch-invariant modes now appearing in RL stacks (e.g., TitanRL). — [post](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); [torchtitan README](https://github.com/pytorch/torchtitan)
- **More surveys and blogs:**
  - "Rollout Efficiency in Reinforcement Learning for Reasoning LLMs: A Taxonomy and Future Directions" (Gholipour et al., arXiv 2609.25463, Sep 2026; TMLR per search listing).
  - "RL Infrastructure for the Mathematically Inclined" (Chenggong Zhang, personal blog, May 12, 2026).
  - SemiAnalysis "RL Systems: Mind the Gap" (June 2026). **Paywalled.**
  - — [arXiv](https://arxiv.org/abs/2609.25463); [blog](https://chenggong-zhang.github.io/rl-infra-survey/); [SemiAnalysis](https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching)

#### Frameworks (status as of 2026-11-24)
- **verl** (repo moved from ByteDance's `volcengine/verl` to **`verl-project/verl`**; old URL redirects): https://github.com/verl-project/verl
  - ~24.3k★, pushed 2026-11-24. v0.9.1 (2026-09-20, PyPI 0.9.1).
  - "the open-source version of HybridFlow" (EuroSys 2025; arXiv 2409.19256). A "hybrid-controller programming model", and "Efficient actor model resharding with 3D-HybridEngine", which "Eliminates memory redundancy ... during transitions between training and generation".
  - Integrates "FSDP, Megatron-LM, vLLM, SGLang".
  - 2026/11: "Mainline PPO training now uses V1 exclusively; the V0 trainer has been removed. Fully async and one-step-off trainers have moved to verl-recipe, pinned to verl v0.9.1."
  - Also: RL-Insight observability (2026/07) and VeRL-Tinker, which runs the Tinker Cookbook loop on your own GPUs (2026/08).
  - Docs: https://verl.readthedocs.io/en/latest/ (HybridFlow page live). — [README](https://github.com/verl-project/verl); [stats](https://ungh.cc/repos/volcengine/verl); [HybridFlow](https://arxiv.org/abs/2409.19256); [docs](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
- **OpenRLHF**: https://github.com/OpenRLHF/OpenRLHF
  - ~10.2k★, pushed 2026-11-03. v0.11.2 (2026-09-14).
  - "Ray + vLLM distributed architecture" with DeepSpeed and a "unified agent-based design paradigm". 2026: VLM RLHF and multi-turn VLM RL (v0.10, 2026/4).
  - Paper arXiv 2405.11143 (v6 Oct 2025). Docs: https://openrlhf.readthedocs.io/en/latest/
  - The README also promotes NVIDIA's "Molt" (`NVIDIA-NeMo/labs-molt`). — [README](https://github.com/OpenRLHF/OpenRLHF); [arXiv](https://arxiv.org/abs/2405.11143)
- **slime** (THUDM; the README positions it "Alongside the GLM family" and also supports Qwen, DeepSeek and Llama 3): https://github.com/THUDM/slime
  - ~8.9k★, pushed 2026-11-05. v0.4.0 (2026-10-09).
  - "connecting Megatron with SGLang". "Megatron arguments are available directly; SGLang arguments use the `--sglang-` prefix". Includes rollout-only and train-only debugging modes.
  - Docs: https://thudm.github.io/slime/ (live). LMSYS launch blog (Jul 9, 2025): https://lmsys.org/blog/2025-07-09-slime/ — [README](https://github.com/THUDM/slime); [stats](https://ungh.cc/repos/THUDM/slime)
- **Miles** (RadixArk): https://github.com/radixark/miles
  - ~3.4k★, v0.2.0 (2026-11-12). "enterprise-facing reinforcement learning framework".
  - v0.1 tech report "Miles v0.1: Production-Level Post-Training" (arXiv 2609.08368, Sep 2026).
  - LMSYS blogs include "Updating 1 T parameters in seconds: P2P weight transfer in large-scale distributed RL" (2026/04) and day-0 RL support for new models. Awesome-ML-SYS files Miles train–inference-mismatch notes under its slime section. — [README](https://github.com/radixark/miles); [arXiv](https://arxiv.org/abs/2609.08368)
- **AReaL** (originally `inclusionAI/AReaL`; repo now **`areal-project/AReaL`**): https://github.com/areal-project/AReaL
  - ~5.9k★, v2.1.1 (2026-11-04). "Built on a **fully asynchronous RL training paradigm**".
  - AReaL 2.0 (2026/07/01) refactored into a "microservice architecture" with separate agent and weight-update services. Paper: arXiv 2505.24298 (v5 Mar 2026).
  - Docs now live under `areal-project.github.io/AReaL/` (linked from the README). **Flag:** the old `inclusionai.github.io/AReaL/` URL redirects to a 404. The PyPI `areal` package is at 1.0.4 (Jun 2026), behind the GitHub release.
  - The community meeting is held in Chinese. — [README](https://github.com/areal-project/AReaL); [arXiv](https://arxiv.org/abs/2505.24298); [PyPI](https://pypi.org/pypi/areal/json)
- **TRL** (Hugging Face): https://github.com/huggingface/trl. ~19.8k★, v1.18.2 (2026-11-19). Docs: https://huggingface.co/docs/trl/index — [stats](https://ungh.cc/repos/huggingface/trl)
- **NeMo RL** (NVIDIA): https://github.com/NVIDIA-NeMo/RL
  - ~2.1k★, v0.8.0 (2026-11-06). Docs: https://docs.nvidia.com/nemo/rl/latest/index.html, plus a design doc at https://docs.nvidia.com/nemo/rl/latest/design-docs/design-and-philosophy.html
  - Nov 2026 README news shows GRPO recipes on the AutoModel backend with vLLM or SGLang generation, including asynchronous MXFP8 vLLM generation.
  - **Flag:** the PyPI `nemo-rl` is a 0.0.0 placeholder, so install from source. — [README](https://github.com/NVIDIA-NeMo/RL); [PyPI](https://pypi.org/pypi/nemo-rl/json)
  - **NeMo-Aligner** is the predecessor: "As of 5/15/2025, this repository is no longer actively maintained. We recommend switching to NeMo RL". **Flag: unmaintained.** — [README](https://github.com/NVIDIA/NeMo-Aligner)
- **SkyRL** (originally `NovaSky-AI/SkyRL`; repo now **`skyrl-project/SkyRL`**): https://github.com/skyrl-project/SkyRL
  - ~2.6k★, `skyrl-v0.5.0` (2026-11-17). Docs moved to https://docs.skyrl.ai/docs
  - Unified `skyrl` library with a backend for the **Tinker API** (`skyrl-tx`). Harbor integration for terminal agents (2026/02). MXFP8 RL recipes (2026/10).
  - The PyPI `skyrl-train` (0.3.1, Jan 2026) is superseded by `skyrl` 0.5.0. — [README](https://github.com/skyrl-project/SkyRL); [PyPI](https://pypi.org/pypi/skyrl/json)
- **ROLL** (Alibaba): https://github.com/alibaba/ROLL
  - ~3.5k★, v0.4.0 (2026-09-29). "multi-role distributed architecture with Ray", integrating "Megatron-Core, SGLang and vLLM". FSDP2 strategy added 2026/02.
  - Docs: https://alibaba.github.io/ROLL/
  - Papers: ROLL (arXiv 2506.06122); **RollArt: Disaggregated Multi-Task Agentic RL Training at Scale** (OSDI'26 per README; arXiv 2512.22560); "Let It Flow" (2512.24873). — [README](https://github.com/alibaba/ROLL); [arXiv](https://arxiv.org/abs/2512.22560)
- **prime-rl** (Prime Intellect): https://github.com/PrimeIntellect-ai/prime-rl
  - ~2.3k★, v0.10.0 (2026-10-13). "Fully asynchronous RL", "FSDP2 for training and vLLM for inference", targets "1T+ MoE models on 1000+ GPUs". Supports EP and CP, plus "One-line SLURM deployment".
  - Companion **verifiers** (RL environments + evals): ~4.8k★, v0.4.1.
  - INTELLECT-2 paper (decentralized RL): arXiv 2505.07291. — [README](https://github.com/PrimeIntellect-ai/prime-rl); [verifiers stats](https://ungh.cc/repos/PrimeIntellect-ai/verifiers); [arXiv](https://arxiv.org/abs/2505.07291)
- **PipelineRL** (ServiceNow): https://github.com/ServiceNow/PipelineRL
  - ~0.4k★, last pushed 2026-08-05. "asynchronous reinforcement learning implementation with in-flight weight updates ... After each optimizer step, updated weights are broadcast to the inference servers".
  - Paper arXiv 2509.19128. — [README](https://github.com/ServiceNow/PipelineRL); [arXiv](https://arxiv.org/abs/2509.19128)
- **torchforge + Monarch** (Meta PyTorch):
  - torchforge ("PyTorch-native post-training at scale") was announced Oct 22, 2025. ~0.7k★, last pushed 2026-09-09. It requires Monarch, vLLM and torchtitan. Docs: https://meta-pytorch.org/torchforge
  - Monarch (single-controller actor framework with "supervision trees"): ~1.1k★, v0.8.0 (2026-11-13).
  - **Flag:** torchforge looks early-stage (PyPI 0.0.1 from Jul 2025, slower commits). torchtitan's in-tree **TitanRL** (2026/08) is a newer PyTorch-native option. — [blog](https://pytorch.org/blog/introducing-torchforge/); [README](https://github.com/meta-pytorch/torchforge); [Monarch](https://github.com/meta-pytorch/monarch)
- **Weight-sync middleware: checkpoint-engine** (Moonshot AI): https://github.com/MoonshotAI/checkpoint-engine
  - ~1.0k★, v0.4.2 (2026-07-04). "middleware to update model weights in LLM inference engines -- a critical step in reinforcement learning".
  - A `ParameterServer` colocated with inference engines does **Broadcast** (synchronous, fastest) or **P2P** (for dynamically added instances). verl uses "NCCL + checkpoint-engine buckets" per the HF survey. — [README](https://github.com/MoonshotAI/checkpoint-engine); [HF blog](https://huggingface.co/blog/async-rl-training-landscape)
- **Other active RL stacks:**
  - allenai **open-instruct**: ~3.9k★, pushed 2026-11-24.
  - **EasyR1** (verl-based multimodal): ~5.2k★.
  - **rLLM**: ~5.9k★.
  - OpenPipe **ART**: ~10.9k★, v0.5.24.
  - HF **open-r1**: ~26.5k★, pushed 2026-10-26.
  - **ms-swift**: ~16.1k★.
  - **tinker-cookbook** (Thinking Machines; open cookbook for the Tinker service): ~4.3k★, v0.7.1.
  - — [stats](https://ungh.cc/repos/allenai/open-instruct); [stats](https://ungh.cc/repos/hiyouga/EasyR1); [stats](https://ungh.cc/repos/rllm-org/rllm); [stats](https://ungh.cc/repos/OpenPipe/ART); [stats](https://ungh.cc/repos/thinking-machines-lab/tinker-cookbook)
- **Outdated or unmaintained:**
  - CarperAI **trlx**: last pushed 2024-01-08, last release 2023-06.
  - **DeepSpeed-Chat** paper (2023, arXiv 2308.01320; code in DeepSpeedExamples). It predates the vLLM/SGLang-rollout era.
  - **NeMo-Aligner**: unmaintained.
  - — [trlx stats](https://ungh.cc/repos/CarperAI/trlx); [arXiv](https://arxiv.org/abs/2308.01320)

#### Key RL-systems papers (verified via arXiv)
- HybridFlow (2409.19256) · OpenRLHF (2405.11143) · ReaL parameter reallocation (2406.14088) · RLHFuse stage fusion (2409.13221)
- Asynchronous RLHF (Noukhovitch et al., ICLR 2025; 2410.18252) · StreamRL disaggregated stream generation (2504.15930) · LlamaRL (Meta; 2505.24034) · AReaL (2505.24298) · PipelineRL (2509.19128)
- **Laminar** async post-training with relay workers (Sheng et al., 2510.12633) · Periodic Asynchrony (2511.18871) · RollArt (2512.22560) · Miles v0.1 (2609.08368)
- Algorithm and system context: DeepSeekMath/GRPO (2402.03300) · DeepSeek-R1 (2501.12948) · DAPO (2503.14476) · Kimi k1.5 (2501.12599) · Kimi K2 (2507.20534) · MiniMax-M1 (2506.13585)
- — [arXiv list](https://arxiv.org/abs/2510.12633); [arXiv](https://arxiv.org/abs/2410.18252); [arXiv](https://arxiv.org/abs/2505.24034)

### Inferences
- **Which is best for learners?** My judgment:
  - (a) **Concepts first:** TRL GRPO with vLLM, or verifiers + prime-rl. These have the smallest conceptual surface, and the HF survey describes native-Python designs as "minimal dependencies, easy to debug".
  - (b) **Then verl.** It has the largest community, a peer-reviewed design paper, extensive docs, and English and Chinese source walkthroughs in Awesome-ML-SYS. Its colocated HybridEngine is the clearest way to see *resharding between training and serving layouts*.
  - (c) **For a learner whose goal is SGLang serving, slime (then Miles) is the most aligned.** It is SGLang-native with direct Megatron args, and the SGLang team's notes explain weight updates, FP8/INT4 rollouts and speculative decoding inside RL.
  - (d) **AReaL / PipelineRL / prime-rl** for fully asynchronous designs and staleness control.
  - (e) torchtitan **TitanRL** for a PyTorch-native, determinism-focused view.
- RL infra is where training and serving skills meet. The serving-side topics to master are:
  - Weight sync (NCCL broadcast, CUDA IPC, P2P, checkpoint-engine).
  - Engine sleep/wake and memory release in colocated mode.
  - Partial rollouts and abort/resume.
  - Train–inference numerical mismatch and determinism.
  - Rollout throughput (speculative decoding, FP8/INT4 rollouts).
- The Anyscale (2025) table should be read alongside the HF (2026) survey, because async support changed substantially within a year.

### Gaps
- I could not text-verify the Notion post at https://fengyao.notion.site/off-policy-rl (JS-rendered; URL is live). From memory, it is Feng Yao et al.'s Aug 2025 "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" on train–inference mismatch and truncated importance sampling. Verify before citing.
- Kimi K2's RL-infra section (colocated checkpoint engine) and MiniMax-M1's infra details were not read in this session.
- NVIDIA "Molt" (`NVIDIA-NeMo/labs-molt`), promoted in the OpenRLHF README, was not verified.
- There is no neutral benchmark comparing these frameworks' throughput. Published speedups come from each project's own papers.

## 5. Cluster-level concerns: collectives, datacenter networking, storage and checkpointing, schedulers, reliability

### Takeaway
The best free learning path on cluster concerns combines Stas Bekman's open book (network, storage, SLURM, fault-tolerance, checkpoint chapters), the Smol Training Playbook's infrastructure chapter, the infrastructure sections of the Llama 3 and DeepSeek-V3 reports, and a short list of primary systems papers:
- Collectives: Demystifying NCCL (2025/2026).
- Networking: Meta's RoCE paper (SIGCOMM'24) and Rail-only.
- Reliability: MegaScale (2024), ByteRobust (2025), FlashRecovery (2025), Fault-Tolerant HSDP on 100,000 GPUs (2026).
For schedulers, the primary resources are project docs: Slurm 26.11 (released today), Slinky (Slurm on k8s), Kueue, Volcano, KAI Scheduler, JobSet, LeaderWorkerSet and Kubeflow Trainer. I found no neutral comparison tutorial.

### Cited Findings

#### Collective communication
- **NCCL**: https://github.com/NVIDIA/nccl. ~5.3k★, **v2.32.7-1 released 2026-11-24**. NCCL User Guide (live): https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html. Benchmarking is done with **nccl-tests**: ~1.7k★, pushed 2026-10-05. — [stats](https://ungh.cc/repos/NVIDIA/nccl); [stats](https://ungh.cc/repos/NVIDIA/nccl-tests)
- **"Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms"** (Hu et al., arXiv 2507.04786, v3 Mar 2026). Judging by its title, it is the deepest available analysis of NCCL internals. I did not read the body, but my understanding is that it covers the Simple/LL/LL128 protocols and ring/tree algorithms. — [arXiv](https://arxiv.org/abs/2507.04786)
- **Explainers:**
  - Scaling book ch. 3 (collective cost models).
  - Stas Bekman's `network/` and `network/benchmarks/` chapters (all-reduce benchmarks).
  - Smol Training Playbook infra chapter (NVLink/NVLS numbers).
  - — [scaling book](https://jax-ml.github.io/scaling-book/); [ML Eng network](https://github.com/stas00/ml-engineering/blob/master/network/README.md); [Smol Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- **Alternatives and research stacks:** Microsoft **MSCCL++** (~0.6k★, v0.10.0, 2026-07) and **UCCL** (~1.6k★, v0.1.1, 2026-05). — [stats](https://ungh.cc/repos/microsoft/mscclpp); [stats](https://ungh.cc/repos/uccl-project/uccl)
- **Meta torchcomms** is now **deprecated**: "Its features have been upstreamed into PyTorch Distributed (c10d); new code should use `torch.distributed` directly", and "NCCLX is deprecated for OSS usage". **Flag: deprecated.** — [README](https://github.com/meta-pytorch/torchcomms)

#### AI datacenter networking
- **Meta's SIGCOMM 2024 RoCE paper**, "RDMA over Ethernet for Distributed Training at Meta Scale". The title is from memory; the PDF filename `sigcomm24-final246.pdf` confirms the venue. The engineering blog post (Aug 5, 2024) and a free PDF are hosted by Meta. The ACM DL page returned 403 here, but the PDF is free. — [Meta blog](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/); [PDF](https://engineering.fb.com/wp-content/uploads/2024/08/sigcomm24-final246.pdf)
- **"Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters"** (Wang et al., MIT/Meta; arXiv 2307.12169; HOTI 2024).
  - Defines rail-optimized networks: GPUs of the same rank sit under shared "rail switches", with spine layers forming a full-bisection Clos.
  - Argues the spine can be dropped, "reducing the network cost by 38% to 77% and network power consumption by 37% to 75%", with "8.2%–11.2% throughput overhead" for MoE all-to-all. — [arXiv](https://arxiv.org/abs/2307.12169)
- **Llama 3 paper §3.3:** network design, including a congestion-control discussion that mentions DCQCN, and 4D parallelism at 16K H100s. **DeepSeek-V3 §3.1:** H800 nodes with NVLink/NVSwitch intra-node and InfiniBand inter-node. — [Llama 3](https://arxiv.org/abs/2407.21783); [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- **Vendor docs** (useful but commercial):
  - Juniper JVD "Backend GPU rail-optimized stripe architecture".
  - IBM Developer "virtual rail architecture" (notes that rail designs can crash a job if an uplink or rail switch fails).
  - Dell "Understanding LLM GPU Clusters Fabrics Traffic for Networkers" blog series.
  - — [Juniper](https://www.juniper.net/documentation/us/en/software/jvd/jvd-ai-dc-apstra-nvidia-weka/backend_gpu_rail_optimized_stripe_architecture.html); [IBM](https://developer.ibm.com/articles/virtual-rail-architecture/); [Dell](https://www.dell.com/en-us/blog/understanding-llm-gpus-clusters-fabrics-traffic-for-networkers-part-2/)

#### Storage and checkpointing
- Stas Bekman's `storage/` and `training/checkpoints/` chapters. — [ML Eng book](https://github.com/stas00/ml-engineering)
- PyTorch **DCP** is covered in the FSDP2 tutorial (`get/set_model_state_dict`). torchtitan implements distributed checkpointing "including async saves" and HF↔DCP conversion. — [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html); [torchtitan](https://github.com/pytorch/torchtitan)
- DeepSeek **3FS**: "A high-performance distributed file system designed to address the challenges of AI training and inference workloads". ~10.5k★, pushed 2026-11-17. — [stats](https://ungh.cc/repos/deepseek-ai/3FS)

#### Schedulers and orchestration
- **Slurm** (SchedMD): https://github.com/SchedMD/slurm. ~4.5k★, **release `slurm-26-11-0-1` on 2026-11-24**. Docs: https://slurm.schedmd.com/documentation.html. Stas Bekman's `orchestration/slurm` chapter is the best practitioner intro. — [stats](https://ungh.cc/repos/SchedMD/slurm)
- **Slinky slurm-operator**: "Run Slurm on Kubernetes, by SchedMD". ~0.4k★, v1.3.0 (2026-11-19). — [README](https://github.com/SlinkyProject/slurm-operator)
- **Kueue** (kubernetes-sigs): ~3.2k★, v0.20.7 (2026-11-19). Docs: https://kueue.sigs.k8s.io/. Features include fair sharing with cohorts and borrowing, hierarchical quotas, elastic workloads, and DRA device quotas. — [README](https://github.com/kubernetes-sigs/kueue)
- **Volcano** (CNCF batch system): ~6.1k★, v1.16.0 (2026-10-14). Docs: https://volcano.sh/ — [stats](https://ungh.cc/repos/volcano-sh/volcano)
- **KAI Scheduler** (from NVIDIA; repo now `kai-scheduler/KAI-Scheduler`): ~1.7k★, v0.19.0 (2026-11-11). Topology-aware and hierarchical gang scheduling, integrated with Grove/Dynamo. KubeCon NA 2025 and EU 2026 talks. — [README](https://github.com/kai-scheduler/KAI-Scheduler)
- **Kubernetes APIs:**
  - **JobSet**: ~0.3k★, v0.13.0, "k8s native API for distributed ML training".
  - **LeaderWorkerSet**: ~0.9k★, v1.0.0 (2026-10-16).
  - **Kubeflow Trainer**: ~2.3k★, v2.3.0 (2026-08-07). Docs: https://www.kubeflow.org/docs/components/trainer/
  - — [JobSet](https://ungh.cc/repos/kubernetes-sigs/jobset); [LWS](https://ungh.cc/repos/kubernetes-sigs/lws); [Trainer](https://ungh.cc/repos/kubeflow/trainer)
- **Cluster tooling:**
  - **SkyPilot**: ~10.9k★, v0.14.3; has a torchtitan tutorial.
  - **NVIDIA GPU Operator**: v26.7.3.
  - **NVIDIA DeepOps** ("Tools for building GPU clusters"): still releasing, 26.11 on 2026-11-13.
  - — [SkyPilot](https://ungh.cc/repos/skypilot-org/skypilot); [GPU Operator](https://ungh.cc/repos/NVIDIA/gpu-operator); [DeepOps](https://ungh.cc/repos/NVIDIA/deepops)
- **Ray** (orchestration for most RL frameworks; 8 of 16 per the HF survey): ~44.4k★, 2.60.1 on PyPI (2026-11-23). — [stats](https://ungh.cc/repos/ray-project/ray); [HF blog](https://huggingface.co/blog/async-rl-training-landscape)

#### Reliability and failure handling
- **Llama 3:**
  - "During a 54-day snapshot period of pre-training, we experienced a total of 466 job interruptions", of which 47 were planned and "The remaining 419 were unexpected".
  - "Approximately 78% of the unexpected interruptions are attributed to confirmed hardware issues ... or suspected hardware-related issues like silent data corruption".
  - Despite this, "we achieved higher than 90% effective training time". — [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)
- **MegaScale** (ByteDance, NSDI'24; arXiv 2402.15627): >10,000-GPU training with diagnosis and recovery tooling. — [arXiv](https://arxiv.org/abs/2402.15627)
- **"Robust LLM Training Infrastructure at ByteDance" (ByteRobust)** (arXiv 2509.16293, v4 Oct 2025; SOSP'25 per search result).
  - Per the abstract snippet, it is deployed on a platform with "over 200,000 GPUs" and achieves "97% ETTR for a three-month training job on 9,600 GPUs".
  - It cites Meta reporting hardware failures "approximately every 2.78 hours" on 16,000 GPUs. — [arXiv](https://arxiv.org/abs/2509.16293)
- **FlashRecovery** (arXiv 2509.03047): limits lost computation to a single training step and removes the need for periodic checkpointing, validated at ~10,000 devices (per abstract snippet). — [arXiv](https://arxiv.org/abs/2509.03047)
- **"Training LLMs with Fault Tolerant HSDP on 100,000 GPUs"** (Salpekar et al., arXiv 2602.00277, Jan 30, 2026). Data-parallel replicas are the unit of fault tolerance: only the failed replica restarts while the others continue (per abstract snippet). — [arXiv](https://arxiv.org/abs/2602.00277)
- **Meta, "Revisiting Reliability in Large-Scale Machine Learning Research Clusters"** (Kokolis et al., arXiv 2410.21680, v2 Feb 2025). **Alibaba C4** (anomaly detection and comm optimization; arXiv 2406.04594). — [arXiv](https://arxiv.org/abs/2410.21680); [arXiv](https://arxiv.org/abs/2406.04594)
- **Open-source fault-tolerance tools:**
  - **torchft** (now `meta-pytorch/torchft`): ~0.6k★, v0.3.1 (2026-11-03). Per-step fault tolerance: "Fault Tolerant HSDP", LocalSGD and DiLoCo (experimental), a Lighthouse coordinator, and torchtitan integration.
  - **NVIDIA Resiliency Extension**: ~0.4k★, v0.7.0 (2026-09-16).
  - Stas Bekman's `training/fault-tolerance` chapter.
  - Megatron's "Scale Bitwise-Deterministic Pretraining" blog (Oct 2026), for reproducibility.
  - — [torchft README](https://github.com/meta-pytorch/torchft); [NVRx](https://ungh.cc/repos/NVIDIA/nvidia-resiliency-ext); [ML Eng FT](https://github.com/stas00/ml-engineering/blob/master/training/fault-tolerance/README.md); [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### Inferences
- For a serving-focused learner, the transferable cluster skills are:
  - Reading NCCL and topology behavior: nccl-tests, NVLink/NVLS vs. IB/RoCE.
  - Understanding rail-optimized fabrics, which carry EP all-to-all in both training and disaggregated serving.
  - Gang/topology-aware scheduling on Kubernetes: KAI, Kueue, LWS. These are increasingly shared between training and multi-node serving (e.g., KAI's integration with Grove/Dynamo, and LWS's "DisaggregatedSet" in its description).
- Reliability literature has moved from "checkpoint more often" (2024) to **"never stop the whole job"** (2025–2026): replica-level fault-tolerant HSDP, step-level recovery, and torchft. A learner should read Llama 3 §3, then ByteRobust, then the 100K-GPU FT-HSDP paper, in that order.

### Gaps
- I did not verify Alibaba's HPN (SIGCOMM'24) paper or a neutral, up-to-date InfiniBand vs. RoCE explainer. The sources I found were vendor blogs.
- I found no neutral, current comparison of Kueue vs. Volcano vs. KAI vs. Slurm/Slinky for training. Each project's docs are the only source.
- The specific affiliation of the "Fault Tolerant HSDP on 100,000 GPUs" authors was not verified. The search engine's summary misattributed it, and I did not read the paper.
- NCCL release notes (what changed in 2.28–2.32) were not reviewed.
- SemiAnalysis cluster and networking deep dives exist but are largely paywalled and were not used.
