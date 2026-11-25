# GPU Programming, ML Kernel Development and AI Accelerator Hardware: Non-University Learning Resources (state as of 2026-11-24)

*How this was checked: GitHub star counts and last-push dates were pulled on 2026-11-24 through the public ungh.cc mirror of the GitHub API, because direct GitHub API access was blocked in this environment. Stars are approximate. "Last push" means the most recent push to any branch. READMEs were read through the jsDelivr GitHub CDN. All content is in English unless noted. The "Level" columns are my own assessment, not a claim from the source.*

---

## Q1. What are the most recommended resources (2024–2026) for learning CUDA from scratch through performant ML kernels, and in what order?

### Takeaway
A stable, mostly free learning path stands out for 2026. Build a performance mental model first: Horace He's "Brrrr" essay, Modal's GPU Glossary, and the roofline and GPU chapters of "How to Scale Your Model". Next, write your first kernels interactively with GPU Puzzles, NVIDIA DLI's free intro courses, or the 12-hour freeCodeCamp course. Then go deep with PMPP, GPU MODE lectures and the CUDA Programming Guide (now v13.5). After that, reproduce the GEMM worklogs (Boehm, then Gordić, Shankhdhar and Colfax) and the softmax and attention kernels, and finally move to Hopper/Blackwell-era tools. PMPP is the main paid item. It had a 5th edition in 2026. Several classics are no longer actively updated but remain canonical: llm.c, GPU-Puzzles and Boehm's 2022 worklog.

### Cited Findings

#### Books and official NVIDIA material
- **PMPP 5th edition (2026).** "Programming Massively Parallel Processors: A Hands-on Approach", 5th ed., by Wen-mei W. Hwu, David B. Kirk and Izzat El Hajj (Morgan Kaufmann/Elsevier, ISBN 9780443439001, 680 pp.). It adds new chapters on filtering, wavefront algorithms, a multi-GPU API and advanced matrix-multiplication optimizations. It also expands coverage of warp-level programming, cooperative groups, CUDA C++ atomics, and multi-GPU programming with NCCL and NVSHMEM. Release dates conflict across listings: Elsevier shows Feb 27, 2026 and Jun 3, 2026; retailers show about Jun 19, 2026; an inspection-copy page shows Sep 1, 2026. — [Elsevier](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1); [Elsevier catalog](https://shop.elsevier.com/books/catalog/isbn/9780443439001); [Elsevier inspection copy](https://www.inspectioncopy.elsevier.com/book/details/9780443439001)
- **PMPP 4th edition.** ISBN 9780323912310. It is still the edition the GPU MODE resource list links to. GPU MODE Lecture 2 is a recap of PMPP chapters 1–3. An official PMPP YouTube channel (@pmpp-book) hosts an "Applied Parallel Programming" playlist, which is university-style but the canonical companion to the book. — [Elsevier 4th ed.](https://shop.elsevier.com/books/catalog/isbn/9780323912310); [GPU MODE resource-stream](https://github.com/gpu-mode/resource-stream); [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- **NVIDIA CUDA Programming Guide.** Current version CUDA **v13.5**, "Last updated on Oct 31, 2026". It is organized in five parts:
  - Introduction
  - Programming GPUs in CUDA: Intro to CUDA C++, **Intro to CUDA Python**, Writing SIMT Kernels, **Writing Tile Kernels**, Asynchronous Execution, Unified and System Memory, NVCC
  - Advanced CUDA, including Programming Systems with Multiple GPUs
  - CUDA Features: 23 chapters, including CUDA Graphs, Cooperative Groups, Asynchronous Barriers, Pipelines, Asynchronous Data Copies, Cluster Launch Control and Virtual Memory Management
  - Technical Appendices: compute capabilities, floating point, and the C++ memory and execution models

  Archived versions 13.3.0, 13.3.1 and 13.4.1 also carry a "Writing Tile Kernels" chapter. — [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html); [13.4.1 archive](https://docs.nvidia.com/cuda/archive/13.4.1/cuda-programming-guide/02-basics/writing-tile-kernels.html)
- **NVIDIA DLI self-paced courses:**
  - "An Even Easier Introduction to CUDA": free, about 1 hour, covers CUDA C++, nvcc and nvprof. — [NVIDIA self-paced training](https://www.nvidia.com/en-eu/training/online/)
  - "Getting Started with Accelerated Computing in Modern CUDA C++": complimentary, about 6 hours. — [NVIDIA Developer Program](https://developer.nvidia.com/join-nvidia-developer-program)
  - "Fundamentals of Accelerated Computing with CUDA Python": about 8 hours, **$90**. — [NVIDIA self-paced training](https://www.nvidia.com/en-eu/training/online/)
  - "Optimizing CUDA Machine Learning Codes With Nsight Profiling Tools": 2 hours, **$30**. — [NVIDIA self-paced training](https://www.nvidia.com/en-eu/training/online/)
  - A free NVIDIA Developer Program account is required. — [unstop guide](https://unstop.com/blog/nvidia-free-self-paced-learning-courses/amp)
- **NVIDIA Accelerated Computing Hub.** Repo `NVIDIA/accelerated-computing-hub`, about 2.25k stars, last push 2026-11-23. It is an open collection of GPU-programming tutorials runnable on NVIDIA Brev or Google Colab:
  - CUDA C++ tutorial (L40S/L4/T4)
  - CUDA Python tutorials, including "CuPy, cuDF, CCCL & Kernels – 8 Hours", "CuPy, SIMT & Tile – 8 Hours" and "cuda.core & CCCL – 2 Hours"
  - CUDA Tile tutorial (A10G or newer Ampere/Ada/Blackwell)
  - PyHPC (NumPy/CuPy/mpi4py) and nvmath-python
  - Warp
  - GPU deployment tutorial

  — [ACH README](https://github.com/NVIDIA/accelerated-computing-hub)
- **Stephen Jones (NVIDIA CUDA architect) talks:**
  - "How CUDA Programming Works", GTC Fall 2022 (session A41101). It is tagged 101-level and described as "not a course on CUDA programming… a foundation on what works, what doesn't work, and why". The on-demand video may need an NVIDIA Developer login. — [NVIDIA On-Demand](https://www.nvidia.cn/on-demand/session/gtcfall22-a41101)
  - NVIDIA's blog lists his series of foundational talks: how GPU computing works, how CUDA programming works, and how to write a CUDA program. — [NVIDIA blog](https://developer.nvidia.com/blog/advanced-strategies-for-high-performance-gpu-programming-with-nvidia-cuda/)
  - GTC 2022 "CUDA: New Features and Beyond" is on YouTube. — [YouTube via GPU MODE resource-stream](https://www.youtube.com/watch?v=SAm4gwkj2Ko)
  - A GTC 2025 edition of "CUDA: New Features and Beyond" exists. — [third-party listing](https://conference-intel.pages.dev/c/gtc-2025/gtc25-016/)
  - He is listed among GTC 2026 speakers. — [NVIDIA GTC speakers](https://www.nvidia.com/en-us/gtc/speakers/)
- **Other free official and lecture material linked by GPU MODE as "1st contact":**
  - NVIDIA "GPU Performance Background User's Guide"
  - OLCF NVIDIA CUDA Training Series (recordings plus exercises)
  - "An Easy Introduction to CUDA C and C++"
  - Older books "CUDA by Example" (free PDF) and "The CUDA Handbook"

  — [GPU MODE resource-stream](https://github.com/gpu-mode/resource-stream); [NVIDIA GPU Performance Background guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html); [OLCF CUDA Training Series](https://www.olcf.ornl.gov/cuda-training-series/)

#### Community lecture series and video courses
- **GPU MODE (formerly CUDA MODE).** A community that started as a reading-group Discord founded by Mark Saroufim and Andreas Köpf, later renamed to cover more hardware and languages. Recordings are on the YouTube channel @GPUMODE; code and slides are in `gpu-mode/lectures` (about 7.1k stars, last push 2026-09-09). Lecture numbering reaches at least **Lecture 114**. — [GPU MODE lectures README](https://github.com/gpu-mode/lectures)

  Selected titles:

  | Area | Lectures |
  |---|---|
  | Basics | 1 Profiling and Integrating CUDA kernels in PyTorch; 2 PMPP Ch. 1–3 recap; 3 Getting Started with CUDA; 4 Intro to Compute and Memory Architecture; 8 CUDA Performance Checklist; 9 Reductions |
  | Kernels and libraries | 12 Flash Attention; 14 Practitioner's Guide to Triton; 15 CUTLASS; 16 Hands-on profiling; 18 Fused Kernels; 20/21/24 Scan; 23 Tensor Cores; 29 Triton Internals; 36 CUTLASS and FlashAttention 3; 37 Intro to SASS and GPU Microarchitecture |
  | Multi-GPU | 17 NCCL; 67 NCCL and NVSHMEM; 78 Iris: Multi-GPU Programming in Triton |
  | Serving | 22 Speculative Decoding in vLLM; 35 SGLang Performance Optimization; 40 FlashInfer; 79 Mirage Mega Kernels |
  | Other | 25 Composable Kernel (AMD); 41 CUDA Docs for Humans; 42 Mosaic GPU; 75 GPU Programming Fundamentals + ThunderKittens; 84 Numerics and AI; 106 HF kernels |
  | CuTe and Gluon | 57 CuTe; 86 Intro to CuTeDSL; 103 CuTe Layout Algebra; 104 Gluon; 114 PyCuTe |

- **GPU MODE Discord.** One listing shows about 31,280 members (undated snapshot). — [Discord server listing](https://discord.com/servers/1189498204333543425)
- **GPU MODE resource-stream.** About 2.4k stars, last push 2026-06-15. A curated link list covering first contact with CUDA, papers and case studies, books (PMPP, CUDA by Example, The CUDA Handbook) and courses. Its "Practice" entry is GPU Puzzlers by Adnan Aziz and Anupam Bhatnagar, and a "CUDA Grandmasters" section covers Tri Dao, Tim Dettmers and Sasha Rush. It also lists the 12-hour freeCodeCamp CUDA video. — [resource-stream README](https://github.com/gpu-mode/resource-stream)
- **freeCodeCamp CUDA course (Elliot Arledge).** About 12 hours of video (11 h 55 min). Course repo `Infatoshi/cuda-course` has about 4.2k stars, last push 2026-03-11. — [freeCodeCamp](https://www.freecodecamp.org/news/learn-cuda-programming/); [Class Central](https://classcentral.com/course/freecodecamp-cuda-programming-course-high-performance-computing-with-gpus-326216); [repo README](https://github.com/Infatoshi/cuda-course)
  - Syllabus: DL ecosystem, setup, C/C++ review, gentle intro to GPUs, first kernels, CUDA APIs (cuBLAS/cuDNN), optimizing matmul, Triton, PyTorch CUDA extensions, and a final project (an MLP for MNIST in CUDA).
  - Prerequisites: Python is required; differentiation and linear algebra are recommended. It targets Ubuntu and any NVIDIA GTX/RTX/datacenter GPU or a cloud GPU.
  - Its stated aim is to give a foundation for understanding llm.c.
  - freeCodeCamp also published a follow-up, "CUDA Programming for NVIDIA H100s", with about a 24-hour watch time. — [freeCodeCamp H100 course](https://www.freecodecamp.org/news/cuda-programming-for-nvidia-h100s/)

#### Canonical practitioner blogs and worklogs
- **Horace He, "Making Deep Learning Go Brrrr From First Principles".** The page references 2022. It splits runtime into **compute, memory bandwidth and overhead**, and calls operator fusion "the most important optimization in deep learning compilers". Its example: `x.cos().cos()` fused cuts global reads and writes from 4 to 2. Its compute-intensity sweep shows the workload is memory-bound below roughly 64 multiplications per element. — [horace.io](https://horace.io/brrr_intro.html)
- **Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog".** Published Dec 31, 2022, on an RTX A6000 (some A100 plots). The FP32 SIMT kernels progress as follows:

  | Kernel | % of cuBLAS |
  |---|---|
  | Naive | 1.3% |
  | GMEM coalescing | 8.5% |
  | SMEM caching | 12.8% |
  | 1D blocktiling | 36.5% |
  | 2D blocktiling | 68.7% |
  | Vectorized | 78.4% |
  | Autotuned | 84.8% |
  | Warptiling | **93.7%** |

  A "Work in Progress: Kernel 11" section was never finished. Repo `siboehm/SGEMM_CUDA` has about 1.4k stars, last push 2025-09-02. — [siboehm.com](https://siboehm.com/articles/22/CUDA-MMM)
- **Aleksa Gordić, "Inside NVIDIA GPUs: Anatomy of high performance matmul kernels".** About Sept 29, 2025, per a mirror. It focuses on Hopper/H100 in four parts:
  - GPU memory fundamentals: gmem/smem/L1/L2, and power throttling versus speed-of-light (SOL)
  - SASS and PTX
  - a near-SOTA warp-tiling synchronous kernel
  - asynchronous Hopper kernels with tensor cores and TMA, persistent kernels and Hilbert-curve scheduling

  Secondary summaries report about 317 TFLOP/s. I could not reach the original page (aleksagordic.com/blog/matmul) with my tools. — [mirror](https://yuanlehome.github.io/blog/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels); [reading notes](https://monishver11.github.io/blog/2025/aleksagordic-gpu-blog-notes)
- **Pranjal Shankhdhar, "Outperforming cuBLAS on H100: a Worklog".** The final kernel beats cuBLAS by about 7% at N=4096 in a single dependency-free C++ file. Summaries cite 764 TFLOPs and Hilbert-curve tile scheduling. It builds on Boehm's A6000 work. The sequel, "Outperforming cuBLAS on NVFP4", reports +4.7% at 8192³. Repo `pranjalssh/fast.cu` has about 630 stars, last push 2026-08-15. — [H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog); [NVFP4 sequel](https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4)
- **Colfax Research CUTLASS tutorials:**
  - The Hopper WGMMA tutorial is based on Pradeep Ramani's CUTLASS 3.5.1 wgmma example. — [WGMMA tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
  - The Blackwell series has three parts:
    - Part 1: tcgen05 MMA and Tensor Memory
    - Part 2: clusters, TMA multicast and CTA pairs
    - Part 3: low-precision, block-scaled MMA
  - Hopper's WGMMA is deprecated on Blackwell in favor of `tcgen05.mma` (called "UMMA" in CUTLASS). Consumer Blackwell (CC 12.0) lacks Tensor Memory.
  - [Blackwell Part 1](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/); [Blackwell Part 2](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)
- **ThunderKittens.** The repo has a step-by-step educational GEMM series under `kernels/gemm/educational_h100`. — [ThunderKittens README](https://github.com/HazyResearch/ThunderKittens)

#### Catalog: foundational and kernel-progression resources

| Resource | Maintainer | Latest edition / last activity | Format and cost | Level (assessed) | Prerequisites | Key topics; why it stands out |
|---|---|---|---|---|---|---|
| [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | Horace He | 2022, no revisions found | Blog, free | Beginner | PyTorch basics | Compute vs. bandwidth vs. overhead, fusion. The best short mental model. |
| [Modal GPU Glossary](https://modal.com/gpu-glossary) | Modal (`modal-labs/gpu-glossary`, about 790 stars) | Repo push 2026-09-04 | Web reference, free | Beginner to intermediate | None | Four sections: Device Hardware, Device Software, Host Software, Performance. Covers the roofline model, arithmetic intensity, occupancy, coalescing, bank conflicts, TMA, the CUDA Tile model and CuTe DSL. — [glossary](https://modal.com/gpu-glossary) |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Google DeepMind authors (`jax-ml/scaling-book`, about 1.5k stars) | Repo push 2026-10-19; GPU chapter Aug 18, 2025 | Online book, free | Intermediate | Linear algebra, basic ML | Rooflines (Part 1), TPUs (Part 2), GPUs (Part 12). — [GPU chapter](https://jax-ml.github.io/scaling-book/gpus/) |
| PMPP 5th ed. | Hwu, Kirk, El Hajj | 2026 | Book, **paid** | Beginner to advanced | C | The standard textbook; now covers NCCL/NVSHMEM and advanced matmul. — [Elsevier](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1) |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | NVIDIA | v13.5, Oct 31, 2026 | Docs, free | All | C++ or Python | Authoritative reference; now includes a Python intro and tile kernels. |
| DLI "Even Easier Intro" and "Modern CUDA C++" | NVIDIA DLI | Listing undated | Self-paced, free (about 1 h and 6 h) | Beginner | C++ | Official hands-on start. — [DLI](https://www.nvidia.com/en-eu/training/online/); [Dev Program](https://developer.nvidia.com/join-nvidia-developer-program) |
| [Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) | NVIDIA (about 2.25k stars) | Push 2026-11-23 | Notebooks, free (Colab/Brev) | Beginner to intermediate | Python or C++ | Free 2–8 h CUDA Python and C++ syllabi plus a CUDA Tile tutorial. |
| [GPU MODE lectures](https://github.com/gpu-mode/lectures) | GPU MODE community (about 7.1k stars) | Push 2026-09-09 | YouTube and repo, free | Beginner to advanced | Python, some C++ | The broadest living lecture series (114+). |
| [freeCodeCamp CUDA course](https://www.freecodecamp.org/news/learn-cuda-programming/) | Elliot Arledge (`Infatoshi/cuda-course`, about 4.2k stars) | Push 2026-03-11 | 12 h video, free | Beginner | Python, C basics | End-to-end beginner course through matmul, Triton and extensions. |
| [Boehm SGEMM worklog](https://siboehm.com/articles/22/CUDA-MMM) | Simon Boehm (`siboehm/SGEMM_CUDA`, about 1.4k stars) | Dec 2022; repo push 2025-09 | Blog and code, free | Intermediate | CUDA basics | The classic optimization ladder. **Pre-tensor-core FP32 only.** |
| [Inside NVIDIA GPUs](https://yuanlehome.github.io/blog/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels) (mirror) | Aleksa Gordić | About Sept 2025 | Long blog, free | Intermediate to advanced | Boehm-level CUDA | Hopper memory system, SASS/PTX, TMA and tensor-core async kernels. |
| [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) | Pranjal Shankhdhar (`fast.cu`, about 630 stars) | Repo push 2026-08-15 | Substack, free | Advanced | Hopper basics | Beats cuBLAS. An NVFP4/Blackwell sequel exists. |
| [Colfax CUTLASS tutorials](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) | Colfax Research | Hopper and Blackwell series (dates not captured) | Tutorials, free | Advanced | C++ templates, CuTe | WGMMA, TMEM/tcgen05, CTA pairs, block scaling. |
| [llm.c](https://github.com/karpathy/llm.c) | Andrej Karpathy (about 31.5k stars) | **Last push 2025-06-26 (dormant)** | Repo, free | Intermediate | C, CUDA | GPT-2 training in raw C/CUDA. Great to read, but no longer evolving. |
| [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) | tspeterkim (about 1.2k stars) | **Last push 2024-12-30** | Repo, free | Intermediate | CUDA basics | FlashAttention forward pass in about 100 lines of CUDA. |
| [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) | xlite-dev (about 12.2k stars) | Push 2026-11-24 | Repo or "open-source book", free | Beginner to advanced | CUDA basics | Hundreds of kernels and notes: BF16/FP8/FP4, HGEMM, CuTe. |
| [learn-cuda](https://github.com/gau-nernst/learn-cuda) | gau-nernst (about 530 stars) | Push 2026-10-03 | Repo, free | Intermediate | PyTorch, CUDA | "Learn CUDA with PyTorch". |
| [cuda-samples](https://github.com/NVIDIA/cuda-samples) | NVIDIA (about 9.8k stars) | Push 2026-11-04 | Repo, free | All | C++ | Official feature samples. |
| [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | BBuf (about 3.4k stars) | Push 2026-11-02 | Repo, free | Intermediate | CUDA | Optimization notes; I did not check what language the content is in. |

#### Softmax, reduction and attention-specific resources
- **Triton tutorials.** The official tutorial list is: Vector Addition, **Fused Softmax**, Matrix Multiplication, Low-Memory Dropout, Layer Normalization, **Fused Attention**, Libdevice, Group GEMM, Persistent Matmul, and Block Scaled Matrix Multiplication. — [triton-lang.org tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- **Triton-Puzzles.** The puzzles build up "from trivial examples… to real algorithms like Flash Attention and Quantized neural networks". — [Triton-Puzzles README](https://github.com/gpu-mode/Triton-Puzzles)
- **Reference implementations to read:**
  - `Dao-AILab/flash-attention`: about 25.3k stars, last push 2026-11-23. — [repo](https://github.com/Dao-AILab/flash-attention)
  - `linkedin/Liger-Kernel` ("Efficient Triton Kernels for LLM Training"): about 6.7k stars, last push 2026-11-21. — [repo](https://github.com/linkedin/Liger-Kernel)
  - `Dao-AILab/quack` ("A Quirky Assortment of CuTe Kernels"): about 1.2k stars, last push 2026-11-22. — [repo](https://github.com/Dao-AILab/quack)
  - `flashinfer-ai/flashinfer` ("Kernel Library for LLM Serving"): about 7.0k stars. — [repo](https://github.com/flashinfer-ai/flashinfer)
  - `deepseek-ai/FlashMLA`: about 13.3k stars. — [repo](https://github.com/deepseek-ai/FlashMLA)
  - `deepseek-ai/DeepGEMM` ("clean and efficient BLAS kernel library"): about 9.2k stars. — [repo](https://github.com/deepseek-ai/DeepGEMM)

### Inferences
- **Recommended learning order.** The phases and time estimates below are my synthesis.
  1. **Phase 0, mental model (1–2 weeks, $0, no GPU):**
     - Horace He's essay
     - the Modal GPU Glossary, Device Hardware and Performance sections
     - "How to Scale Your Model" Part 1 (rooflines) and Part 12 (GPUs)
     - NVIDIA's GPU Performance Background guide
     - optionally Tensor Puzzles to sharpen PyTorch broadcasting
  2. **Phase 1, first kernels (2–4 weeks, $0):**
     - GPU-Puzzles (Numba in Colab) first
     - then DLI "An Even Easier Introduction to CUDA"
     - then **either** the freeCodeCamp 12-hour course **or** the free DLI "Modern CUDA C++" course
     - optionally the Accelerated Computing Hub CUDA C++ and CUDA Python notebooks
     - start easy LeetGPU problems
  3. **Phase 2, systematic foundations (1–3 months):**
     - PMPP (the 5th ed. if buying; the 4th ed. is fine), chapters through memory, tiling, reduction, scan, convolution and matmul
     - paired with GPU MODE lectures 1–9, 16 (profiling), 20–24 (scan, tensor cores) and 37 (SASS)
     - Stephen Jones' "How CUDA Programming Works"
     - keep the CUDA Programming Guide v13.5 as the reference
  4. **Phase 3, classic performant kernels (1–2 months):**
     - reproduce Boehm's SGEMM ladder
     - then reductions, softmax and layernorm: GPU MODE Lecture 9, the Triton fused-softmax and layer-norm tutorials, and Liger kernels as production references
     - then attention: flash-attention-minimal, then the Triton fused attention tutorial, then the final Triton-Puzzles, then GPU MODE Lecture 12
     - submit to GPU MODE, LeetGPU or Tensara leaderboards to get feedback on real hardware
  5. **Phase 4, tensor-core era (2–3 months; needs Hopper/Blackwell time):**
     - Gordić's anatomy post
     - Shankhdhar's H100 worklog and `fast.cu`
     - Colfax WGMMA, then the Colfax Blackwell series
     - the ThunderKittens educational H100 GEMMs
     - the CuTe DSL notebooks and tcgen05 guide
     - the Gluon TMA, WGMMA and tcgen05 tutorials
     - NVFP4 problems from the GPU MODE hackathon
  6. **Phase 5, serving-oriented specialization:**
     - read FlashInfer, FlashMLA and DeepGEMM source
     - GPU MODE lectures 22 (vLLM speculative decoding), 35 (SGLang), 40 (FlashInfer) and 84 (numerics)
     - multi-GPU communication: lectures 17, 67 and 78, plus DeepEP
- **Prioritize living resources.** GPU MODE, the CUDA Programming Guide, the Accelerated Computing Hub, LeetCUDA and the CUTLASS docs are all updated in 2026. Use dormant classics such as llm.c, GPU-Puzzles, Boehm's worklog and flash-attention-minimal for understanding, not for current best practice.
- **Boehm's worklog is now an intro.** It is FP32 SIMT on Ampere, so the 2025–26 Hopper/Blackwell worklogs (Gordić, Shankhdhar, Colfax) are now the step up to "performant" kernels.
- **The paid DLI CUDA Python course has free overlap.** The $90 course overlaps heavily in topic with the free Accelerated Computing Hub CUDA Python syllabi, so a self-learner can likely skip the paid course. This is an inference from topic overlap; I did not compare the content line by line.

### Gaps
- I did not find the PMPP 5th edition list price, and its exact release date conflicts across Elsevier and retailer listings.
- I did not re-verify the current version or date of the CUDA C++ Best Practices Guide (docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
- The DLI "Even Easier Introduction" listing still mentions nvprof, which suggests the listing text may be dated. I could not confirm what the current course content uses.
- I could not fetch the original Gordić post. Its date and throughput numbers come from mirrors and summaries.
- I could not confirm the title of Stephen Jones' GTC 2026 talk, or whether it is freely viewable.
- The release date of the freeCodeCamp H100 course is not verified.
- The GPU MODE Discord member count is from an undated listing.
- Colfax article publication dates were not captured.

---

## Q2. What is the current landscape of kernel-authoring languages/DSLs, what learning materials exist for each, and which are worth learning first in 2026?

### Takeaway
By late 2026 the landscape has consolidated around tile-based programming at several levels of abstraction:

- **Triton**: the largest ecosystem, with official tutorials and puzzles. Its lower-level dialect **Gluon** has official Hopper/Blackwell tutorials.
- **CUTLASS 4.x / CuTe**: both C++ and the **CuTe Python DSL** (docs up to 4.8.0).
- **NVIDIA cuTile / CUDA Tile**: launched Dec 2025 and now part of the CUDA Programming Guide. Hardware support is still limited.
- **ThunderKittens 2.0**: full Blackwell support, plus HipKittens for AMD.
- **TileLang**: strong on NVIDIA and AMD.
- **Helion**: PyTorch Foundation-hosted, 1.x in 2026, and compiles to Triton.
- **Mojo**: runs on NVIDIA, AMD and Apple GPUs.

For a serving-oriented learner, learn Triton first, CUDA C++ fundamentals alongside it, then CuTe DSL (or Gluon) for Hopper/Blackwell. cuTile, Helion and TileLang are worth tracking rather than learning first.

### Cited Findings

#### Triton and Gluon
- **Triton.** Repo `triton-lang/triton` has about 20.6k stars, last push 2026-11-24. The official tutorial set runs from vector addition to fused attention, persistent matmul and block-scaled matmul. — [repo stats via mirror](https://github.com/triton-lang/triton); [tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- **Triton learning materials:**
  - Triton-Puzzles: no GPU needed thanks to the Triton interpreter.
  - GPU MODE lectures 14 (Practitioner's Guide to Triton), 29 (Triton Internals), 34 (Low Bit Triton Kernels) and 78 (Iris: multi-GPU programming in Triton).
  - NVIDIA's blog on Triton for Blackwell.
  - An AMD fork exists at `ROCm/triton`.

  — [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles); [GPU MODE lectures](https://github.com/gpu-mode/lectures); [NVIDIA blog](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- **Gluon.** Gluon is "a GPU programming language based on the same compiler stack as Triton… lower-level… gives the user more control", exposing layouts, shared memory, warp specialization and target-specific features. It assumes basic familiarity with Triton. The official tutorials cover:
  - Introduction
  - Tensor Layouts
  - Async Copy
  - TMA
  - Warp-Group MMA
  - The 5th Generation TensorCore (tcgen05, with Tensor Memory; requires a Blackwell GPU)

  GPU MODE Lecture 104 covers "Gluon: Tile-Based GPU Programming with Low-Level Control". — [Gluon overview](https://triton-lang.org/main/gluon/); [Gluon intro](https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html); [tcgen05 tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html); [GPU MODE lectures](https://github.com/gpu-mode/lectures)

#### CUTLASS, CuTe and CuTe DSL
- **The CUTLASS repo.** `NVIDIA/cutlass` has about 10.8k stars, last push 2026-11-23. It now describes itself as "CUDA Templates and **Python DSLs** for High-Performance Linear Algebra". Documentation exists through version 4.8.0. — [CUTLASS 4.8.0 Python DSL overview](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/overview.html)
- **CuTe DSL setup and tutorials:**
  - Start with the Quick Start, then the DSL Programming Model: JIT functions, kernels, static values and control flow.
  - It supports Linux x86_64/aarch64 and Python 3.10–3.14.
  - The tutorial notebooks run from `00_quickstart` through layouts, layout algebra, tensors, algorithms, MMA atoms, a GEMM tutorial and predication.
  - A tcgen05 MMA programming guide places the accumulator in Tensor Memory and claims 2–4× the throughput of Hopper WGMMA, depending on dtype.

  — [Quick Start 4.7](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/quick_start.html); [guides index](https://docs.nvidia.com/cutlass/4.6.2/media/docs/pythonDSL/guides.html); [tcgen05 guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)
- **NVIDIA blogs and talks on CuTe DSL.** The launch post introduced CuTe DSL as beta in CUTLASS 4. A follow-up post shows C++ and DSL code with "identical programming models and programming patterns". There is a GTC 2025 talk, "Enable Tensor Core Programming in Python with CUTLASS 4.0", and an LLVM 2025 talk. — [NVIDIA blog](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/); [talks list](https://docs.nvidia.com/cutlass/4.4.0/media/docs/pythonDSL/cute_dsl_general/resources.html)
- **Community CuTe material.** Colfax tutorials (see Q1); QuACK (CuTe DSL kernels by Dao-AILab); GPU MODE lectures 15, 36, 57, 86, 103 and 114. — [GPU MODE lectures](https://github.com/gpu-mode/lectures); [quack](https://github.com/Dao-AILab/quack)

#### NVIDIA cuTile / CUDA Tile
- **Launch and documentation.** NVIDIA's "Simplify GPU Programming with NVIDIA CUDA Tile in Python" was published 2025-12-04. It links docs, video tutorials and samples, and a follow-up blog covers high-performance matmul. — [NVIDIA blog](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python); [matmul blog](https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile)
- **Requirements conflict:**
  - The quickstart covers vector addition and the `@ct.kernel` / `@ct.function` decorators, installs with `pip install cuda-tile`, and requires CUDA Toolkit 13.1+, driver r580+ and Python 3.10–3.13. It lists **compute capability 10.x or 12.x**, i.e. Blackwell. — [cuTile quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)
  - The README says "the `tileiras` compiler (version 13.2) only supports Blackwell GPU and Ampere/Ada GPU. Hopper GPU will be supported in the coming versions." — [cuTile README](https://github.com/NVIDIA/cutile-python)
- **Repos and other materials:**
  - `NVIDIA/cutile-python`: about 2.2k stars, created 2025-06, last push 2026-11-21.
  - `NVIDIA/TileGym`: about 850 stars, created 2025-11-13, last push 2026-11-24. It offers "kernel tutorials and examples for tile-based GPU programming", benchmarks and LLM integration examples, with READMEs in English, Simplified Chinese, Traditional Chinese, Japanese and French.
  - `NVIDIA/cuda-tile`: the MLIR-based CUDA Tile IR, about 1.1k stars.
  - `cuTile.jl` brings the model to Julia.
  - The CUDA Programming Guide has a "Writing Tile Kernels" chapter.
  - The Accelerated Computing Hub has a CUDA Tile tutorial.

  — [TileGym README](https://github.com/NVIDIA/TileGym); [cuTile.jl blog](https://developer.nvidia.com/blog/cutile-jl-brings-nvidia-cuda-tile-based-programming-to-julia); [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)

#### ThunderKittens (and HipKittens, ParallelKittens)
- **ThunderKittens 2.0.** The README dates the release to Jan 11, 2026; the release blog by Stuart Sul and Chris Ré is dated Feb 19, 2026. It brings "full support for Blackwell GPUs along with MXFP8 and NVFP4". Ampere is no longer actively supported, and the project is built and tested mainly for Hopper and Blackwell. The repo has about 3.8k stars, last push 2026-09-12. — [README](https://github.com/HazyResearch/ThunderKittens); [TK 2.0 blog](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- **Learning materials:**
  - the educational H100 GEMM series in the repo
  - blogs: "GPUs Go Brrr" (May 2024), "ThunderKittens Now on Blackwells!" (Mar 2025) and ParallelKittens (multi-GPU, Nov 2025)
  - GPU MODE Lecture 75
  - HipKittens for AMD (blog Nov 11, 2025; MLSys 2026), repo `HazyResearch/HipKittens` with about 490 stars, last push 2026-10-15

  — [GPUs Go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk); [TK Blackwell](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell); [ParallelKittens](https://hazyresearch.stanford.edu/blog/2025-11-17-pk); [HipKittens blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)

#### TileLang
- **Repo and tutorials.** `tile-ai/tilelang` has about 9.0k stars, last push 2026-11-23. The README includes a GEMM tutorial covering layout, L2-cache swizzling and pipelining. Examples cover GEMM, dequantized GEMM, FlashAttention and linear attention. `T.gemm` dispatches to CuTe on NVIDIA and HIP on AMD. — [TileLang mirror README](https://github.com/coderonion/tilelang); [paper](https://arxiv.org/pdf/2504.17577)
- **Paper results.** The paper reports speedups of 1.36×, 1.41× and 1.70× over FlashAttention-3, Triton and PyTorch respectively. GEMM is roughly at parity with vendor libraries on RTX 4090, A100, H100 and MI300X. — [arXiv 2504.17577](https://arxiv.org/pdf/2504.17577)
- **AMD tutorial.** AMD's ROCm blog (Jan 2026) shows FlashAttention on MI300X in "fewer than 80 lines" versus 500+ lines of CUDA. — [ROCm blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html)
- **Third-party tutorial.** A July 2026 Colab tutorial covers GEMM, fused softmax, FlashAttention and autotuning. — [MarkTechPost](https://www.marktechpost.com/2026/07/25/designing-high-performance-gpu-kernels-with-tilelang-tensor-core-gemm-fused-softmax-flashattention-and-autotuning/)

#### Helion (PyTorch)
- **Status:**
  - It describes itself as "A Python-embedded DSL that makes it easy to write fast, scalable ML kernels with minimal boilerplate". Repo `pytorch/helion` has about 1.0k stars, last push 2026-11-24.
  - It became a PyTorch Foundation-hosted project, announced Apr 7, 2026.
  - It compiles to Triton, with TileIR and more backends "coming soon". PTC EU 2026 slides list Pallas as next and CuTe DSL and Metal later.
  - A PyPI mirror lists helion 1.1.0 released May 20, 2026.
  - It hosted a PLDI 2026 tutorial (Jun 15, 2026).

  — [Linux Foundation press release](https://www.linuxfoundation.org/press/pytorch-foundation-welcomes-helion-as-a-foundation-hosted-project-to-standardize-open-portable-and-accessible-ai-kernel-authoring); [PyTorch blog](https://pytorch.org/blog/helion/); [PTC Paris 2026 slides](https://hosted-files.sched.co/pytorchconferenceeu2026/b8/_PTC%20Paris-2026_%20Helion.pdf); [PyPI mirror](https://simple-repository.app.cern.ch/project/helion); [HPCwire](https://www.hpcwire.com/aiwire/2026/04/07/pytorch-foundation-welcomes-helion-as-new-project-for-ai-kernel-development/)
- **Learning material.** Red Hat Developer's explainer (Apr 24, 2026); the PyTorch dev-discuss Helion category; a GPU MODE Helion B200 competition, whose submissions are exported in the `GPUMODE/kernelbot-data` dataset. — [Red Hat](https://developers.redhat.com/articles/2026/04/24/200-lines-15-how-helion-rewriting-rules-gpu-programming); [dev-discuss](https://dev-discuss.pytorch.org/t/about-the-helion-category/3315.md); [kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data/tree/refs%2Fpr%2F12)

#### Mojo (Modular)
- **Repos.** `modular/modular` has about 30.2k stars, last push 2026-11-21. "Mojo GPU Puzzles, Edition 1" is at puzzles.modular.com; its repo has about 450 stars, last push 2026-11-24. — [mojo-gpu-puzzles README](https://github.com/modular/mojo-gpu-puzzles)
- **Hardware support.** The puzzles run on Apple silicon GPUs, NVIDIA Blackwell, AMD MI325X and MI355X "among others". On Apple M2–M5, almost all puzzles work, excluding the PyTorch and NVIDIA-specific ones. Puzzle 34 needs SM90+. Pre-Turing NVIDIA GPUs are unsupported. The debugging puzzles are NVIDIA-only. — [Modular 25.6 blog](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple); [EuroLLVM 2026 slides](https://www.llvm.org/devmtg/2026-04/slides/quick_talk/quick_talk_panchenko.pdf); [puzzles how-to](https://puzzles.modular.com/howto.html)

#### Other options
- **Numba-CUDA.** `NVIDIA/numba-cuda`, about 300 stars. GPU-Puzzles uses Numba, which "directly maps Python code to CUDA kernels". — [GPU-Puzzles README](https://github.com/srush/GPU-Puzzles)
- **JAX Pallas / Mosaic GPU.** Covered in GPU MODE Lecture 42. — [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- **Hugging Face `kernels`.** "Build compute kernels and load them from the Hub"; about 790 stars; GPU MODE Lecture 106. — [repo](https://github.com/huggingface/kernels)
- **NVIDIA CCCL / cuda.compute.** An NVIDIA team used it to top GPU MODE leaderboards. — [NVIDIA blog](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute)
- **KernelBench.** "Can LLMs Write GPU Kernels?", Torch to CUDA plus more DSLs; about 1.3k stars, last push 2026-03-24. It belongs to a 2026 trend of benchmarks and contests for AI-written kernels, alongside KernelBench-Verified (Jul 2026), SOL-ExecBench (Mar 2026) and the MLSys 2026 CUDA kernel-agent contest. — [KernelBench-Verified](https://www.alphaxiv.org/abs/2607.16241); [SOL-ExecBench](https://alphaxiv.org/abs/2603.19173); [SyFI MLSys 2026](https://syfi.cs.washington.edu/blog/2026-05-28-mlsys-kernel-contest/)

#### Summary table

| DSL | Abstraction | Hardware | Best first material | Repo stars / last push |
|---|---|---|---|---|
| Triton | Block/tile, Python | NVIDIA, AMD | Triton-Puzzles, then official tutorials | about 20.6k / 2026-11-24 |
| Gluon | Lower-level Triton (layouts, TMA, warp specialization) | NVIDIA Hopper/Blackwell focus | Official Gluon tutorials; GPU MODE L104 | (in Triton repo) |
| CuTe DSL / CUTLASS 4.x | Layout algebra, Python and C++ | NVIDIA (Ampere to Blackwell) | CuTe notebooks, tcgen05 guide, Colfax, QuACK | about 10.8k / 2026-11-23 |
| cuTile (CUDA Tile) | Tile, Python (also Julia) | Blackwell, Ampere/Ada; Hopper "coming" (sources conflict) | NVIDIA blog, quickstart, TileGym, ACH notebook | about 2.2k / 2026-11-21 |
| ThunderKittens 2.0 | C++ embedded tile primitives | Hopper, Blackwell (Ampere legacy); HipKittens for AMD | Educational H100 GEMM series; Hazy blogs | about 3.8k / 2026-09-12 |
| TileLang | Tile DSL (TVM-based) | NVIDIA, AMD | README GEMM tutorial, examples, AMD blog | about 9.0k / 2026-11-23 |
| Helion | High-level PyTorch-like, autotuned; lowers to Triton | Wherever Triton runs (more backends planned) | PyTorch blog, Red Hat explainer, PLDI tutorial | about 1.0k / 2026-11-24 |
| Mojo | New systems language | NVIDIA, AMD, Apple | Mojo GPU Puzzles | about 30.2k (modular) / 2026-11-21 |

### Inferences
- **What to learn first, for a vLLM/SGLang-oriented learner in 2026:**
  1. **Triton.** It has the biggest ecosystem, free no-GPU puzzles, a complete official tutorial ladder, and works on NVIDIA and AMD. Helion also compiles to it.
  2. **Enough CUDA C++ to read production kernels.** FlashInfer, FlashMLA and DeepGEMM are referenced by serving engines; GPU MODE has dedicated lectures on SGLang, FlashInfer and vLLM.
  3. **CuTe DSL for Hopper/Blackwell performance work.** NVIDIA's own docs, the NVFP4 competition lecture and QuACK all push the Python CuTe path, so it is the most transferable "expert" skill. Gluon is the alternative for people already committed to Triton.
- **cuTile is promising but not a first choice yet.** It is first-party and now documented in the CUDA Programming Guide, but the hardware-support restrictions (Hopper missing per the README) and its youth (launched Dec 2025) make it something to learn after Triton and CuTe.
- **ThunderKittens and TileLang are good second languages.** ThunderKittens' small C++ codebase and educational GEMMs teach Hopper/Blackwell structure. TileLang is notable for AMD portability.
- **Mojo is the most accessible for some learners.** It is the only listed option whose puzzles run on Apple silicon GPUs, which helps learners with Macs, but its industry use in serving stacks is less established. I have no usage data to quantify this.

### Gaps
- I found no authoritative 2026 survey quantifying which DSLs vLLM and SGLang kernels actually use. The serving-engine researcher should confirm the share of Triton, CUDA, CuTe and FlashInfer kernels.
- cuTile's current Hopper support is unresolved. The docs quickstart says CC 10.x/12.x, while the README says Ampere/Ada/Blackwell with Hopper "coming". Check the release notes for CUDA 13.5.
- The latest Triton release number, the CUTLASS release cadence beyond docs version 4.8.0, and an official tilelang.com tutorial index were not captured.
- I did not verify whether Meta's Triton low-level extensions (TLX) are public. The `facebookexperimental/triton` repo describes itself only as a mirror.

---

## Q3. Which practice platforms, puzzles and competitions exist, and what do they cost?

### Takeaway
Almost everything is free:

- Sasha Rush's puzzle series. Triton-Puzzles now lives under the GPU MODE org and needs no GPU.
- Mojo GPU Puzzles.
- Tensara: free and open source, but its site was unreachable on 2026-11-24.
- GPU MODE's KernelBot leaderboards: free runs on real datacenter GPUs via popcorn-cli.

LeetGPU has a free tier and a Pro tier that third parties report at $6.99/month. Competitions in 2025–26 had large prizes: NVIDIA's Blackwell NVFP4 hackathon, and AMD's $1.1M GPU MODE "E2E Model Speedrun".

### Cited Findings

#### Puzzles

| Puzzle set | Maintainer | Stars / last push | Cost and hardware | Level (assessed) | Notes |
|---|---|---|---|---|---|
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | Sasha Rush | about 12.6k / **2024-09-01** | Free; Colab GPU runtime; Numba | Beginner | "In a few hours… go from basics to understanding the real algorithms that power 99% of deep learning". A walkthrough video exists. A CUDA C++ port by dshah3 is linked in the [resource-stream](https://github.com/gpu-mode/resource-stream). |
| [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | Sasha Rush with Tejas Ramesh and Keren Zhou; now GPU MODE org | about 2.7k / 2026-04-01 | Free; **no GPU needed** (Triton interpreter, Triton-Viz); Colab | Beginner to intermediate | Builds up to Flash Attention and quantized NNs; Discord channel #triton-puzzles. — [README](https://github.com/gpu-mode/Triton-Puzzles) |
| [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) | Sasha Rush | about 4.4k / **2024-07-15** | Free, CPU | Beginner | PyTorch broadcasting and vectorization. |
| [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles) | Sasha Rush | about 1.2k / **2024-01-10** | Free, CPU | Intermediate | "What would you do with 1000 H100s…": parallelism and memory. |
| [Mojo GPU Puzzles](https://puzzles.modular.com/) | Modular | about 450 / 2026-11-24 | Free; needs an NVIDIA (Turing+), AMD or Apple M2–M5 GPU | Beginner to advanced | "Edition 1". Cross-vendor. Puzzle 34 needs SM90. — [README](https://github.com/modular/mojo-gpu-puzzles); [how-to](https://puzzles.modular.com/howto.html) |
| [GPU Puzzlers](http://www.gpupuzzlers.com/) | Adnan Aziz and Anupam Bhatnagar | Unknown | Unknown | Unknown | Listed under "Practice" in the GPU MODE resource-stream; current status not verified. — [resource-stream](https://github.com/gpu-mode/resource-stream) |

#### Practice and benchmarking platforms
- **LeetGPU** (leetgpu.com):
  - **Cost.** A free tier plus a **Pro plan at $6.99/month** that "adds faster GPUs and higher limits", per a third-party comparison. The pricing page is JavaScript-rendered and I could not read prices from it. — [TensorTonic comparison](https://www.tensortonic.com/guides/leetcode-for-machine-learning)
  - **Languages.** A third-party guide says it supports CUDA, Triton, PyTorch, tinygrad and Mojo with free GPU access. — [AI Merge guide](https://read.theaimerge.com/p/the-mlai-engineers-starter-guide)
  - **Challenge repo.** `AlphaGPU/leetgpu-challenges` has about 1.25k stars, last push 2026-11-20. It provides starters for CUDA, Mojo, CuTe DSL, Triton, PyTorch and JAX. The reported time is the median of 15 timed runs, with a wall-clock limit of about 40 s. The license is CC BY-NC-ND 4.0. — [repo](https://github.com/AlphaGPU/leetgpu-challenges)
  - **History.** It launched as a "Show HN" in April 2025. Earlier, the team described CPU-side GPU emulation with Accel-Sim for a cycle-accurate mode. — [HN](https://news.ycombinator.com/item?id=43673762); [accel-sim group](https://groups.google.com/g/accel-sim/c/zUZhawBYk2Q)
- **Tensara** (tensara.org):
  - Its README describes 60+ challenges in CUDA, Triton and Mojo, run on "actual GPUs (T4, H100, A100, etc.)", with per-GPU leaderboards, baseline comparisons and a CLI. Modal sponsors the benchmarking. — [README](https://github.com/tensara/tensara)
  - A 2025 launch post called it "100% free and open-source", and another page cites 73 problems. — [Product Hunt](https://www.producthunt.com/products/tensara); [project page](https://someshs-website.pages.dev/work/tensara)
  - Repo about 240 stars, last push 2026-07-21. **On 2026-11-24 both tensara.org and tensara.org/problems returned HTTP 522**, a Cloudflare origin-unreachable error, in my checks.
- **GPU MODE leaderboards / KernelBot:**
  - The backend `gpu-mode/kernelbot` (formerly discord-cluster-manager; about 120 stars, last push 2026-11-23) runs submissions on GPUs donated as GitHub Actions runners. Users submit via `popcorn-cli` (about 190 stars, last push 2026-09-18). Problems live in `gpu-mode/reference-kernels` (about 330 stars, last push 2026-11-23). — [kernelbot README](https://github.com/gpu-mode/kernelbot)
  - NVIDIA's CCCL team reported the most first-place finishes across B200, H100, A100 and L4 leaderboards (PrefixSum, VectorAdd, Histogram, Sort, Grayscale). — [NVIDIA blog](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute)
  - Submission data is published on Hugging Face. — [GPUMODE/kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data/tree/refs%2Fpr%2F12)
- **CUDA Online Judge.** An NVIDIA forum post announces this platform for learning CUDA "without GPU hardware". I did not verify details. — [NVIDIA forum](https://forums.developer.nvidia.com/t/introducing-cuda-online-judge-learn-cuda-programming-without-gpu-hardware/355666)

#### Competitions (2025–2026)
- **Blackwell NVFP4 Kernel Hackathon** (GPU MODE + NVIDIA):
  - Run on B200 GPUs, with submissions via popcorn-cli.
  - Kernel #1, NVFP4 batched GEMV, ran Nov 10–28, 2025. Kernel #2 was NVFP4 GEMM (Nov 29–Dec 19), then NVFP4 gated dual GEMM and grouped GEMM.
  - Prizes per problem: 1st, DGX Spark plus a GTC 2026 pass; 2nd, RTX 5090 plus a pass; 3rd, RTX 5080. The grand prize was a Dell Pro Max with GB300. Sesterce donated DGX B200 compute.
  - A public solutions repo (`Laz4rz/nvfp4`) builds on the reference-kernels code.

  — [rules PDF](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf); [NVIDIA forum](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092); [participant write-up](https://ramshankar07.substack.com/p/gpu-mode-nvfp4-blackwell-challenge); [solutions repo](https://github.com/Laz4rz/nvfp4)
- **GPU MODE Virtual Hackathon: E2E Model Speedrun** (sponsored by AMD; announced Mar 9, 2026):
  - Qualifiers ran Mar 6–Apr 6, 2026, on MXFP4 MoE, MLA decode and MXFP4 GEMM.
  - The top 10 teams went to finals (Apr 7–May 15), with awards on May 18.
  - The prize pool was **$1.1M**.

  — [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/new-gpumode-virtual-hackathon--e2e-model-speedrun.html)
- **MLSys 2026 CUDA Kernel Agent Contest.** An AI-agent kernel contest, won by UW's SyFI team. — [SyFI blog](https://syfi.cs.washington.edu/blog/2026-05-28-mlsys-kernel-contest/)

### Inferences
- **Suggested practice ladder:**
  1. Triton-Puzzles and GPU-Puzzles: free, little or no GPU.
  2. LeetGPU and Tensara easy and medium problems: free real-GPU timing.
  3. GPU MODE reference-kernels and leaderboards: free, competition-grade problems on B200, H100 and MI-series GPUs, mirroring real serving kernels such as NVFP4 GEMM, MLA decode and MoE.
  4. Past competition problems with public write-ups, used as capstone projects.
- **Competitions teach serving-relevant work.** Recent problem sets (NVFP4/MXFP4 GEMM, MoE, MLA decode) map directly onto LLM-serving hot paths, so they are unusually relevant for a vLLM/SGLang-oriented learner.
- **Treat Tensara as at-risk.** Given the outage and no repo push since July 2026, have LeetGPU or GPU MODE as alternatives.

### Gaps
- I could not confirm LeetGPU's current official pricing, GPU types per tier, or whether B200 is offered; the page is JavaScript-only.
- I could not confirm whether Tensara's outage is temporary.
- The final standings of the AMD E2E Model Speedrun and the full NVFP4 hackathon results were not found.
- I found no current list of open GPU MODE leaderboards and their hardware as of Nov 2026, because the gpumode.com site is JavaScript-rendered.
- The details and status of GPU Puzzlers and CUDA Online Judge are unverified.

---

## Q4. Which resources explain GPU/accelerator hardware for ML accessibly?

### Takeaway
The best free hardware explainers in 2026 are:

- **Fundamentals:** Modal's GPU Glossary and "How to Scale Your Model" (rooflines, TPUs, and a GPU chapter covering SMs, TMEM, NVLink/NVSwitch, InfiniBand and NVL72).
- **Hopper/Blackwell features:** Colfax's tutorials and NVIDIA's CuTe tcgen05 and Gluon tutorials.
- **Networking:** Stas Bekman's "ML Engineering" network chapter.
- **Low precision:** NVIDIA's NVFP4 explainer plus the FP8 and OCP MX specs.
- **AMD:** ROCm blogs, AMD lab notes and HipKittens.
- **Independent analysis:** Chips and Cheese.

SemiAnalysis is mostly paywalled; its InferenceX benchmark site is free.

### Cited Findings

#### SM, warp execution, memory hierarchy and roofline
- **Modal GPU Glossary.** Device Hardware (SM, Tensor Core, TMA, register file, GPU RAM), Device Software (warp, thread block, memory hierarchy, shared memory, CUDA Tile model), and Performance (roofline model, arithmetic intensity, occupancy, memory coalescing, bank conflicts, warp divergence). — [Modal](https://modal.com/gpu-glossary)
- **"How to Scale Your Model", Part 12 "How to Think About GPUs".** By Jacob Austin, Swapnil Patil, Adam Paszke (Google DeepMind) and Reiner Pope (MatX), Aug 18, 2025. — [scaling-book GPUs](https://jax-ml.github.io/scaling-book/gpus/)
  - Chip architecture: SMs, Tensor Cores, warp schedulers. The H100 has 132 SMs and the B200 has 148.
  - Memory hierarchy: registers, SMEM, about 50 MB of L2, HBM, and Blackwell TMEM.
  - Spec tables from V100 to B200.
  - Networking: NVLink/NVSwitch nodes, InfiniBand fat trees and SuperPods, scale-up versus scale-out, and GB200 NVL72.
  - Collectives, including SHARP.
  - Parallelism rooflines with DeepSeek-V3 and LLaMA-3 examples.
  - It builds on Part 1 (rooflines) and Part 2 (TPUs).
- **Other architecture introductions:**
  - GPU MODE Lecture 4 (Compute and Memory Architecture), Lecture 23 (Tensor Cores) and Lecture 37 (SASS and GPU Microarchitecture). — [GPU MODE lectures](https://github.com/gpu-mode/lectures)
  - The NVIDIA GPU Performance Background User's Guide. — [NVIDIA docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
  - Horace He's compute, bandwidth and overhead framing. — [horace.io](https://horace.io/brrr_intro.html)

#### Hopper/Blackwell features (TMA, WGMMA, tcgen05, TMEM, clusters)
- **Colfax.** WGMMA on Hopper; Blackwell tcgen05 and TMEM (Part 1); clusters, TMA multicast and CTA pairs (Part 2); block-scaled low-precision MMA (Part 3). WGMMA is deprecated on Blackwell, and consumer Blackwell (CC 12.0) has no TMEM. — [Colfax Part 1](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/); [Colfax Part 2](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)
- **NVIDIA and Triton guides.** The CUTLASS tcgen05 guide (2–4× WGMMA throughput). The Gluon tutorials on TMA, WGMMA and tcgen05. The CUDA Programming Guide feature chapters on asynchronous data copies, pipelines and Cluster Launch Control. — [tcgen05 guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html); [Gluon tcgen05](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html); [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- **Gordić's post.** Covers the Hopper memory system, power throttling versus SOL, SASS/PTX and TMA-based async kernels. — [mirror](https://yuanlehome.github.io/blog/inside-nvidia-gpus-anatomy-of-high-performance-matmul-kernels)

#### Low-precision formats (FP8, MXFP8/MXFP4, NVFP4)
- **NVFP4 vs MXFP4:**
  - NVFP4 uses E2M1 elements with a **16-element block** and an **FP8 E4M3 scale**. OCP MXFP4 uses **32-element blocks** with a power-of-two **E8M0** scale.
  - Average storage is about 4.5 bits per element for NVFP4 versus 4.25 for MXFP4.
  - Blackwell supports both NVFP and MXFP formats; AMD supports MXFP4.

  — [NVIDIA NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/); [arXiv 2509.23202](https://arxiv.org/pdf/2509.23202)
- **Accuracy evidence is mixed:**
  - NVIDIA Research reports that MXFP4 needed 36% more tokens than NVFP4 to reach the same pretraining loss. — [NVIDIA Research](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/index.html)
  - Another study finds an enhanced MXFP4 gets within about 1% of NVFP4 accuracy. — [arXiv 2509.23202](https://arxiv.org/html/2509.23202v2)
- **Specs and references:**
  - "FP8 Formats for Deep Learning" (arXiv 2209.05433). — [arXiv](https://arxiv.org/abs/2209.05433)
  - The OCP Microscaling (MX) v1.0 spec. The URL returned a 301 redirect, so the document was not read. — [OCP](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  - NVIDIA TransformerEngine, "including using 8-bit and 4-bit floating point" (about 3.6k stars, last push 2026-11-24). — [repo](https://github.com/NVIDIA/TransformerEngine)
  - The Triton "Block Scaled Matrix Multiplication" tutorial. — [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
  - GPU MODE lectures 7 (Advanced Quantization), 30 (Quantized Training), 34 (Low Bit Triton Kernels), 69 (Quartet 4-bit training) and 84 (Numerics and AI). — [GPU MODE lectures](https://github.com/gpu-mode/lectures)

#### Multi-GPU interconnects (NVLink/NVSwitch, InfiniBand, RoCE)
- **Stas Bekman, "Machine Learning Engineering Open Book".** `stas00/ml-engineering`, about 19.7k stars, last push 2026-11-24. Its network chapter "Inter-Node and Intra-Node Networking Hardware" covers:
  - Cluster networks: frontend, backend and out-of-band.
  - RDMA networking.
  - Intra-node: all-to-all versus peer-to-peer bandwidth, PCIe, NVLink, NVLink-C2C, NVSwitch, Infinity Fabric/xGMI, Gaudi, NeuronLink.
  - Inter-node: InfiniBand, EFA, HPE Slingshot, GPUDirect-TCPX, Omni-Path, UALink, UB Link.
  - SHARP.
  - Why inter-node bandwidth matters for training.

  — [ml-engineering network chapter](https://github.com/stas00/ml-engineering/tree/master/network)
- **Production RoCE at scale.** Meta's engineering post "RoCE networks for distributed AI training at scale" (Aug 2024). — [Engineering at Meta](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/)
- **Collective-communication programming:**
  - GPU MODE lectures 17 (NCCL), 67 (NCCL and NVSHMEM), 70 (fault-tolerant collectives) and 78 (multi-GPU Triton). — [GPU MODE lectures](https://github.com/gpu-mode/lectures)
  - `NVIDIA/nccl-tests` (about 1.7k stars). — [repo](https://github.com/NVIDIA/nccl-tests)
  - DeepEP, "an efficient expert-parallel communication library" (about 10.5k stars). — [repo](https://github.com/deepseek-ai/DeepEP)
  - ParallelKittens (multi-GPU kernels). — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-17-pk)
  - PMPP 5th ed. adds NCCL and NVSHMEM chapters. — [Elsevier](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1)

#### TPUs
- **"How to Scale Your Model" Part 2, "All About TPUs".** This is the core free TPU-hardware text; the GPU chapter builds on and compares against it. — [scaling-book](https://jax-ml.github.io/scaling-book/gpus/)
- **Google TPU Research Cloud (TRC).** Free access to "a cluster of more than 1,000 Cloud TPU devices". It is open to anyone with a research interest, covers only TPUs (not other GCP services), and asks to be acknowledged in publications. — [TRC](https://sites.research.google/trc/about/); [TRC FAQ](https://sites.research.google/trc/faq/)
- **Pallas and Mosaic GPU.** GPU MODE Lecture 42 covers Mosaic GPU, and Helion plans a Pallas backend. — [GPU MODE lectures](https://github.com/gpu-mode/lectures); [PTC 2026 slides](https://hosted-files.sched.co/pytorchconferenceeu2026/b8/_PTC%20Paris-2026_%20Helion.pdf)

#### AMD Instinct / ROCm
- **ROCm blogs and notebooks:**
  - "Matrix Core Programming on AMD CDNA3 and CDNA4 architecture": HIP intrinsics for low-precision MFMA; it points to the CDNA ISA guides' "Matrix Arithmetic Instructions" chapter. — [ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
  - "FP8 GEMM Optimization on AMD CDNA4 Architecture". — [ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
  - An AI Developer Hub notebook on FP8 GEMM in HIP, covering intrinsics and inline assembly. — [ROCm AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/v17.0/notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4.html)
- **AMD lab notes** (GPUOpen; code in AMD/amd-lab-notes). It includes a Matrix Cores note, and AMD training decks list further titles such as "Reading AMD GPU ISA" and "Register pressure in AMD CDNA 2 GPUs". — [GPUOpen AMD lab notes](https://gpuopen.com/learn/amd-lab-notes); [matrix cores note](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme); [AMD training deck](https://www.ccs.tsukuba.ac.jp/wp-content/uploads/sites/14/2025/09/22.-Additional_AMD_Training_Materials.pdf)
- **HipKittens** (Hazy Research, Nov 2025; MLSys 2026):
  - It reports AITER and PyTorch SDPA GQA-backward kernels reaching only 30% and 24% of state of the art on MI355X.
  - It argues that wave specialization, the dominant NVIDIA pattern, is wrong for AMD, and favors 8-wave/4-wave schedules plus chiplet-aware cache reuse.
  - The work was productionalized in AITER.
  - A companion post, "AMD GPUs go brrr" (Nov 9, 2025), covers the hardware.

  — [HipKittens blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk); [AMD GPUs go brrr](https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr); [arXiv 2511.08083](https://arxiv.org/abs/2511.08083v1)
- **AMD repos:**

  | Repo | Stars | Note |
  |---|---|---|
  | `ROCm/hip` | about 4.4k | — |
  | `ROCm/rocm-examples` | about 330 | last push 2026-11-24 |
  | `ROCm/aiter` | about 620 | "AI Tensor Engine for ROCm" |
  | `ROCm/composable_kernel` | — | **Marked "[DEPRECATED] Moved to ROCm/rocm-libraries"** |

  GPU MODE Lecture 25 covers "Speaking Composable Kernel". — [GPU MODE lectures](https://github.com/gpu-mode/lectures)

#### Independent architecture analysis
- **Chips and Cheese.** "Nvidia's B200: Keeping the CUDA Juggernaut Rolling ft. Verda (formerly DataCrunch)" was published 2025-12-15. It notes B200's move to two reticle-sized dies presented as one GPU. Its CDNA 4 coverage reports that the MI355X has fewer CUs than the MI300X but a large vector-throughput lead over Blackwell, and about 40 MB of LDS versus about 33 MB of shared memory on B200. Articles are currently served from `old.chipsandcheese.com`. — [B200 article](https://old.chipsandcheese.com/?p=37649); [CDNA 4 article](https://old.chipsandcheese.com/?p=36728)
- **SemiAnalysis:**
  - The newsletter has free and paid tiers; the paid tier gives "full access to all articles". Industry models are separate institutional products. — [SemiAnalysis FAQ](https://semianalysis.com/faq/); [newsletter about](https://newsletter.semianalysis.com/about)
  - Its InferenceX site publishes free inference-benchmark comparisons. For example, on DeepSeek R1 at about 87 tok/s/user, MI355X is about 14% cheaper per token than B200. — [InferenceX DeepSeek R1 B200 vs MI355X](https://inferencex.semianalysis.com/compare/deepseek-r1-b200-vs-mi355x)
- **Spec comparison.** Tom's Hardware reports MI355X has 288 GB HBM3E versus 192 GB on B200. — [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/inside-the-instinct-mi355x)

### Inferences
- **Hardware reading order for a serving-focused learner:**
  1. Modal Glossary: Device Hardware plus Performance.
  2. Scaling-book Parts 1, 2 and 12.
  3. Horace He's essay.
  4. Low-precision primers: the NVFP4 blog and the FP8 paper, which matter for FP8/FP4 serving.
  5. The ml-engineering network chapter.
  6. Colfax and Gluon Blackwell material, only once writing kernels.
- **AMD learning is less consolidated.** Material is spread across ROCm blogs, lab notes, HipKittens and AITER. The HipKittens finding that NVIDIA-style warp specialization doesn't carry over is a key conceptual warning for anyone porting serving kernels.
- **Analysis-outlet value.** Chips and Cheese provides free microarchitecture-level analysis. SemiAnalysis' free InferenceX data is more useful to a serving engineer than its mostly paywalled newsletter.

### Gaps
- I did not verify current official NVIDIA architecture whitepapers or tuning guides (Hopper/Blackwell/Blackwell Ultra/Rubin) or their URLs.
- I did not find a dedicated, free, up-to-date InfiniBand/RoCE primer beyond the ml-engineering chapter and Meta's post. NVIDIA networking docs were not checked.
- I did not read the OCP MX spec PDF; its URL redirected.
- I did not find SemiAnalysis subscription prices.
- I did not check whether Chips and Cheese has moved to a new domain or platform, beyond the "old." subdomain observation.
- I did not verify Google Cloud's official TPU architecture docs.

---

## Q5. How can a learner without a local GPU practice cheaply or for free, and what are typical costs?

### Takeaway
A learner can get through Phases 0–3 for about $0:

- Triton-Puzzles need no GPU.
- Colab's free tier usually gives a T4 (unnamed, with 12-hour sessions); Kaggle gives about 30 GPU-hours per week.
- LeetGPU, Tensara and GPU MODE run small kernels on real datacenter GPUs for free.
- Modal's Starter plan includes $30/month of credit, enough for roughly 7.6 H100-hours or 4.8 B200-hours at list prices.

Hopper/Blackwell-specific work (TMA, WGMMA, tcgen05, NVFP4) needs paid or competition-provided time. B200 on Modal costs $6.25/hour.

### Cited Findings

#### Free and cheap compute options

| Option | What you get | Limits and caveats | Source |
|---|---|---|---|
| Google Colab (free) | GPU not named by Google; users report a T4 | 12 h max sessions, idle timeouts, no published quota. "Colab Pro for Education" (free year for US students and faculty) is **closed to new signups**. | [gpuperhour (Sep 21, 2026)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) |
| Kaggle Notebooks | T4 (×2) or P100 class; about 30 GPU-h/week commonly reported (not officially published) | 12 h CPU/GPU sessions, 9 h TPU sessions. Linked Colab Pro/Pro+ accounts add 15/30 GPU-h/week. One source claims the P100 was retired Sep 15, 2026, which is unverified. The reset day is disputed. | [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits); [Spheron](https://www.spheron.network/blog/free-gpu-cloud-credits-2026/) |
| Modal Starter | $0 plan plus **$30/month free credits**, up to 3 seats | Per-second list prices: B200 $6.25/h, H200 $4.54/h, H100 $3.95/h, A100-80GB $2.50/h, A100-40GB $2.10/h, L40S $1.95/h, A10 $1.10/h, L4 $0.80/h, T4 $0.59/h. Academics can get up to $10k in credits. | [Modal pricing](https://modal.com/pricing) |
| Lightning AI (free) | Historically 15 credits/month (about 22–80 GPU-h depending on GPU), one Studio with 4 h restarts | **In flux.** A tracker reports a Sep 21, 2026 switch to a one-time grant of up to 30 credits. Another reports a Nov 9, 2026 cut to 50 GPU-hours. Neither is confirmed on Lightning's own page. | [usagepricing tracker](https://www.usagepricing.com/blueprint/activity/lightning-ai-2026-09-21-gpu-rate-card-reprice); [pricingsaas diff](https://pricingsaas.com/companies/lightning/diffs/2026W46); [aicreditmart](https://aicreditmart.com/ai-credits-providers/lightning-ai-free-plan-22-gpu-hours-month-guide-2026/) |
| Hugging Face ZeroGPU | 5 GPU-min/day free on a shared RTX PRO 6000 Blackwell | Gradio Spaces demos only, 60 s function limit; not for kernel development. PRO is $9/month. | [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) |
| AMD Developer Cloud via the AMD AI Developer Program | Free program; AMD pages say **$100 credits (about 50 h)**; MI300X 1× (192 GB) or 8× VMs | **Conflict:** a June 2026 AMD article says $50 valid for 90 days. An older evaluation route gave 25 MI300X hours, used up 8× faster on 8-GPU VMs. | [AMD claim-credits article](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-claim-amd-cloud-credits.html); [AMD cloud access](https://www.amd.com/en/developer/resources/cloud-access.html); [AMD Developer Cloud](https://amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) |
| Google TPU Research Cloud | Free Cloud TPUs for research projects (1,000+ devices) | Application required; TPUs only. | [TRC](https://sites.research.google/trc/about/) |
| NVIDIA Brev / Colab with Accelerated Computing Hub | Prebuilt notebook environments (T4/L4/L40S/A10G) | Brev is paid per instance. I did not verify Brev prices. | [ACH README](https://github.com/NVIDIA/accelerated-computing-hub) |
| LeetGPU / Tensara / GPU MODE KernelBot | Real-GPU benchmarking of small kernels for free. GPU MODE leaderboards have included B200, H100, A100 and L4; Tensara lists T4, A100 and H100. | Limited to platform problems and time limits (about 40 s on LeetGPU). Tensara was down on 2026-11-24. | [leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges); [Tensara README](https://github.com/tensara/tensara); [NVIDIA blog](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute) |
| No-GPU learning | Triton-Puzzles (Triton interpreter); Tensor Puzzles and LLM-Training Puzzles (CPU) | Interpreter mode teaches semantics, not performance. | [Triton-Puzzles README](https://github.com/gpu-mode/Triton-Puzzles) |

- **Closed programs.** Amazon SageMaker Studio Lab is closed to new customers. Google Cloud's $300 free trial cannot add GPUs. — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- **Market rate example.** B200 SXM6 at $5.31/h spot and $11.06/h on-demand on Spheron, as of Nov 18, 2026 (one provider; third-party). — [Spheron](https://www.spheron.network/openai-triton-kernel-gpu-cloud-2026/)
- **Student credits.** The GitHub Student Developer Pack includes a $100 Azure credit and a Camber plan with 5 GPU-hours. — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

### Inferences
- **What $30/month of Modal credit buys** (my arithmetic from Modal's list prices): about 50 T4-hours, about 37 L4-hours, about 7.6 H100-hours, or about 4.8 B200-hours per month. That is ample for Phase 3 and enough for a focused Phase 4 Hopper/Blackwell project if runs are short and profiled carefully.
- **A plausible budget:**

  | Phases | Monthly cost | What it covers |
  |---|---|---|
  | 0–2 | $0 | Colab or Kaggle, Triton interpreter, LeetGPU free tier |
  | 3 | $0–$10 | Kaggle T4s plus free leaderboard runs, optionally LeetGPU Pro ($6.99) |
  | 4 | $0 (Modal credits) up to about $50–100 | Extra H100/B200 hours on Modal or spot providers |

  Entering GPU MODE competitions gives free B200 or MI-series benchmark runs on serious problems.
- **Ampere/Ada is enough for most early cuTile and ACH tutorial work.** The free-tier GPUs are T4, L4 or P100 class. Hopper or Blackwell features (TMA, WGMMA, tcgen05, TMEM, NVFP4) cannot be practiced on free notebook GPUs and need Modal-type rentals or competition infrastructure.
- **Re-check free tiers before recommending them.** Lightning AI, Kaggle and AMD terms changed or conflict in 2026.

### Gaps
- I found no official Google statement of the Colab free-tier GPU type or quota.
- Kaggle's official weekly quota and the P100 retirement claim are unverified.
- Lightning AI's current free-tier terms are unclear.
- The AMD AI Developer Program credit amount conflicts ($100 vs $50).
- I did not verify NVIDIA Brev pricing, or Lambda, RunPod and Vast.ai consumer-GPU rental rates.
- I did not check whether LeetGPU's free tier includes Hopper or Blackwell GPUs.
