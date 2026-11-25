# Production LLM Deployment Stack, Benchmarking, Observability, MLOps and Non-University Deployment Courses (state as of 2026-11-24)

## 1. What does a production LLM serving stack look like in 2026, layer by layer, and which open-source projects dominate each layer?

### Takeaway
By late 2026 the stack around the engine has settled on Kubernetes. vLLM/SGLang engines sit inside a serving control plane: KServe's LLMInferenceService, Ray Serve LLM, or a distributed framework (llm-d, NVIDIA Dynamo, AIBrix). In front sits inference-aware routing, meaning the Gateway API Inference Extension (GIE) InferencePool plus an endpoint picker, usually below a general AI gateway such as LiteLLM or Agent Router (formerly Envoy AI Gateway). Multi-host work is scheduled with LeaderWorkerSet/DisaggregatedSet, Kueue, KAI, Volcano or Grove. Autoscaling uses queue/KV signals through KEDA/HPA, Dynamo Planner or llm-d autoscaling. KV-cache tiers come from LMCache, Mooncake, NIXL and Dynamo KVBM, and observability from Prometheus/Grafana, DCGM and the still-unstable OTel GenAI conventions. Several 2026 milestones show this layer moving from experimental to production-grade: Dynamo 1.0 (Mar), KServe LLMInferenceService GA (Mar), llm-d joining CNCF (Mar) and reaching 1.0 (Oct), LWS 1.0 (Oct), GIE v1.x, the close of WG Serving (Feb), and the AI Conformance program.

### Cited Findings

#### Data-collection note
- GitHub stars, last-push dates and latest-release tags in this file come from the public GitHub-API mirror ungh.cc, queried 2026-11-24, because direct GitHub API access was blocked in this session. Figures are rounded. Example: llm-d had 5,340 stars and 952 forks and was pushed 2026-11-23. — [ungh.cc llm-d](https://ungh.cc/repos/llm-d/llm-d)
- The mirror follows GitHub renames, which exposed several 2026 repo moves: envoyproxy/ai-gateway → theagentrouter/agent-router; bentoml/llm-inference-handbook → modular/llm-inference-handbook; llm-d-inference-scheduler → llm-d/llm-d-router; llm-d-incubation/workload-variant-autoscaler → llm-d/llm-d-autoscaling; ray-project/kuberay → kuberay-project/kuberay; NVIDIA/KAI-Scheduler → kai-scheduler/KAI-Scheduler; substratusai/kubeai → kubeai-project/kubeai; sgl-project/ome → ome-projects/ome; InferenceMAX → SemiAnalysisAI/InferenceX. — [Agent Router repo](https://github.com/theagentrouter/agent-router); [llm-d-router](https://github.com/llm-d/llm-d-router); [InferenceMAX repo notice](https://github.com/InferenceMAX/InferenceMAX)

#### Ecosystem and governance milestones that shape the stack
- Kubernetes WG Serving was disbanded on 2026-02-26 after meeting its goal. Its work moved to permanent homes: GIE to SIG Network; multi-host/LWS to SIG Apps; autoscaling and fast bootstrap to SIG Node or SIG Scheduling; DRA requirements to WG Device Management; inference-perf to SIG Scalability; benchmarking and best practices to the llm-d project. The post says llm-d and AIBrix are now the better forums for driving requirements to SIGs. — [CNCF blog, Yuan Tang, 2026-02-26](https://www.cncf.io/blog/2026/02/26/kubernetes-wg-serving-concludes-following-successful-advancement-of-ai-inference-support/)
- CNCF launched the Certified Kubernetes AI Conformance Program on 2025-11-11. Stricter v1.35 requirements ("Kubernetes AI Requirements", KARs) followed in March 2026, when the number of certified platforms nearly doubled. Coverage says the requirements include Gateway API support for model-aware routing, and that Dynamic Resource Allocation (DRA) was among the first mandatory items. The program now sits under SIG Architecture. — [CNCF 2025 launch](https://www.cncf.io/announcements/2025/11/11/cncf-launches-certified-kubernetes-ai-conformance-program-to-standardize-ai-workloads-on-kubernetes/); [CNCF 2026-03-24](https://www.cncf.io/announcements/2026/03/24/cncf-nearly-doubles-certified-kubernetes-ai-platforms/); [heise](https://heise.de/-11074526); [TechTarget](https://www.techtarget.com/searchitoperations/news/366634474/CNCF-Kubernetes-AI-program-faces-scrutiny-from-IT-analysts); [Kubernetes blog 2026-10-29](https://kubernetes.io/blog/2026/10/29/sig-arch-ai-conformance-2026/); [cncf/k8s-ai-conformance](https://github.com/cncf/k8s-ai-conformance)
- KServe became a CNCF incubating project on 2025-11-11. llm-d was accepted into the CNCF Sandbox on 2026-03-24. — [CNCF KServe incubating](https://www.cncf.io/blog/2025/11/11/kserve-becomes-a-cncf-incubating-project/); [CNCF welcome llm-d](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/)
- KubeCon + CloudNativeCon North America 2026 (Nov 9–12, Salt Lake City) added a dedicated "AI Inference + Agentic" track covering model serving, GPU scheduling and observability. — [CNCF announcement 2026-08-10](https://www.cncf.io/announcements/2026/08/10/cncf-reveals-kubecon-cloudnativecon-north-america-2026-schedule-adds-new-ai-inference-agentic-track/)
- Engine context only (engines are covered by other researchers). vLLM: v0.34.0 (2026-11-16), ~97.5k stars. SGLang: v0.5.23 (2026-11-02), ~39k stars. TensorRT-LLM: latest release v1.2.1 (2026-04-20), ~15.1k stars. llama.cpp: ~136k stars, latest release tag v1.0.0 (2026-11-13). Ollama: v0.40.18 (2026-11-24), ~188k stars. — [vLLM](https://github.com/vllm-project/vllm); [SGLang](https://github.com/sgl-project/sglang); [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM); [llama.cpp](https://github.com/ggml-org/llama.cpp); [Ollama](https://github.com/ollama/ollama)
- Hugging Face TGI is officially in maintenance mode, taking only minor bug-fix, docs and lightweight maintenance PRs. Secondary sources date the switch to 2025-12-11 and point new deployments to vLLM/SGLang. TGI's last release was v3.3.7 (2025-12-19) and its last push 2026-03-21. — [HF TGI docs](https://huggingface.co/docs/text-generation-inference/index); [Spheron migration guide](https://www.spheron.network/blog/migrate-tgi-to-vllm-sglang-2026/); [TGI repo](https://github.com/huggingface/text-generation-inference)
- 20 submitters in MLPerf Inference v6.0 used vLLM, up from 5 in v5.1. — [Red Hat blog](https://www.redhat.com/en/blog/red-hat-ai-tops-mlperf-inference-v60-vllm-qwen3-vl-whisper-and-gpt-oss-120b)

#### Layer map with dominant open-source projects (stars/last push/latest release per GitHub metadata, 2026-11-24)

| Layer | Project (URL) | Maintainer / governance | Latest release (date) | Stars / last push | Notes |
|---|---|---|---|---|---|
| GPU enablement | [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) | NVIDIA | v26.7.3 (2026-11-17) | ~3.0k / 2026-11-24 | Drivers, device plugin, toolkit on K8s |
| GPU telemetry | [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter) | NVIDIA | 4.9.0 (2026-11-06) | ~1.9k / 2026-11-06 | GPU metrics into Prometheus |
| Serving control plane | [KServe](https://github.com/kserve/kserve) | CNCF incubating | v0.21.0 (GitHub tag 2026-09-25; release blog 2026-11-04) | ~6.3k / 2026-11-23 | LLMInferenceService CRD built on llm-d |
| Serving (Python-native) | [Ray / Ray Serve LLM](https://github.com/ray-project/ray) | Anyscale / Ray project | ray-2.60.0 (2026-11-04) | ~44.4k / 2026-11-24 | `pip install "ray[llm]"`; vLLM/SGLang backends |
| Ray on K8s | [KubeRay](https://github.com/kuberay-project/kuberay) | kuberay-project org | v1.7.3 (2026-11-23) | ~2.8k / 2026-11-24 | Operator for RayService |
| Model serving framework | [BentoML](https://github.com/bentoml/BentoML) | BentoML (acquired by Modular, 2026-02-10) | v1.4.40 (2026-10-23) | ~9.0k / 2026-11-17 | General model-serving framework |
| Multi-framework server | [Triton / NVIDIA Dynamo-Triton](https://github.com/triton-inference-server/server) | NVIDIA | v2.74.0 (2026-11-02) | ~11.2k / 2026-11-24 | Renamed "Dynamo-Triton" 2025-03-18 |
| Distributed LLM serving | [llm-d](https://github.com/llm-d/llm-d) | CNCF Sandbox | v1.0.0 (2026-10-20) | ~5.3k / 2026-11-24 | See Q2 |
| Distributed LLM serving | [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | NVIDIA (open source) | v1.6.2 (2026-11-05) | ~8.6k / 2026-11-24 | See Q2 |
| Distributed LLM serving | [AIBrix](https://github.com/vllm-project/aibrix) | vLLM project (ByteDance origin) | v0.8.0 (2026-10-13) | ~5.2k / 2026-11-23 | See Q2 |
| Reference Helm stack | [vLLM production-stack](https://github.com/vllm-project/production-stack) | vLLM project / LMCache team | vllm-stack-0.1.14 (2026-11-07) | ~2.7k / 2026-11-07 | See Q2 |
| K8s operators (lighter) | [KAITO](https://github.com/kaito-project/kaito) | CNCF Sandbox (Microsoft origin) | v0.13.0 (2026-10-24) | ~1.0k / 2026-11-24 | Workspace CRD; AKS add-on |
| | [KubeAI](https://github.com/kubeai-project/kubeai) | kubeai-project | Helm chart models-0.25.1 (2026-11-10) | ~1.3k / 2026-11-23 | "AI Inference Operator for Kubernetes" |
| | [OME](https://github.com/ome-projects/ome) | ome-projects (SGLang ecosystem) | v1.2.2 (2026-08-05) | ~530 / 2026-11-24 | LLM-serving operator |
| | [llmaz](https://github.com/InftyAI/llmaz) | InftyAI | v0.1.4 (2025-06-10) | ~310 / 2026-08-27 | Low release activity |
| Inference routing API | [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) | Kubernetes SIG Network | v1.6.2 (2026-09-17) | ~810 / 2026-11-23 | InferencePool GA since v1.0.0 |
| Endpoint picker | [llm-d-router](https://github.com/llm-d/llm-d-router) (formerly llm-d-inference-scheduler) | llm-d | v1.0.0 (2026-10-16) | ~480 / 2026-11-24 | Prefix/load/KV-aware scheduling |
| vLLM-native router | [vllm-project/router](https://github.com/vllm-project/router) | vLLM project | v0.1.17 (2026-11-04) | ~535 / 2026-11-24 | "High-performance and light-weight router" |
| Semantic/model routing | [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | vLLM project | v0.4.0 (2026-09-27) | ~6.5k / 2026-11-24 | Mixture-of-Models routing |
| AI gateway (L7) | [LiteLLM](https://github.com/BerriAI/litellm) | BerriAI | v1.110.3 (2026-11-24) | ~63.8k / 2026-11-24 | Repo describes "Rust core with Python SDK" |
| AI gateway (Envoy) | [Agent Router (ex-Envoy AI Gateway)](https://github.com/theagentrouter/agent-router) | Agentic AI Foundation (since Sept 2026) | v1.3.0 (2026-11-18) | ~2.4k / 2026-11-23 | Built on Envoy Gateway |
| AI gateway | [kgateway](https://github.com/kgateway-dev/kgateway) | kgateway project | v2.4.7 (2026-10-16) | ~5.7k / 2026-11-19 | "Cloud-Native API Gateway and AI Gateway" |
| AI gateway | [Portkey gateway](https://github.com/Portkey-AI/gateway) | Portkey | v1.15.2 (2026-01-12) | ~13.4k / 2026-11-20 | Commercial-backed OSS gateway |
| Multi-host workload API | [LeaderWorkerSet + DisaggregatedSet](https://github.com/kubernetes-sigs/lws) | SIG Apps | v1.0.0 (2026-10-16) | ~910 / 2026-11-15 | DisaggregatedSet (KEP-766) for P/D roles |
| Gang/topology scheduling | [Grove](https://github.com/ai-dynamo/grove) | NVIDIA (Dynamo) | v0.2.1-beta1 (2026-11-11) | ~300 / 2026-11-24 | PodCliqueSet/PodClique |
| Queueing/quotas | [Kueue](https://github.com/kubernetes-sigs/kueue) | kubernetes-sigs | v0.20.7 (2026-11-19) | ~3.2k / 2026-11-24 | Job queueing |
| AI scheduler | [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) | kai-scheduler org (NVIDIA origin) | v0.19.0 (2026-11-11) | ~1.7k / 2026-11-24 | GPU-aware scheduler |
| Batch scheduler | [Volcano](https://github.com/volcano-sh/volcano) | CNCF | v1.16.0 (2026-10-14) | ~6.1k / 2026-11-24 | Gang scheduling (AIBrix integrates) |
| Distributed jobs | [JobSet](https://github.com/kubernetes-sigs/jobset) | kubernetes-sigs | v0.13.1 (2026-11-18) | ~350 / 2026-11-18 | Mostly training/HPC |
| Multi-cloud compute | [SkyPilot](https://github.com/skypilot-org/skypilot) | SkyPilot | v0.14.3 (2026-11-16) | ~10.9k / 2026-11-23 | `sky serve` autoscaled replicas |
| Multi-cloud compute | [dstack](https://github.com/dstackai/dstack) | dstack | 0.22.8 (2026-11-12) | ~2.3k / 2026-11-20 | Orchestration across clouds |
| Event autoscaling | [KEDA](https://github.com/kedacore/keda) | CNCF | v2.21.1 (2026-11-15) | ~10.7k / 2026-11-23 | KServe 0.21 wires it directly |
| SLO autoscaler | [llm-d-autoscaling](https://github.com/llm-d/llm-d-autoscaling) | llm-d | v1.0.0 (2026-10-13) | ~60 / 2026-11-24 | "Variant optimization autoscaler" |
| KV-cache layer | [LMCache](https://github.com/LMCache/LMCache) | LMCache | v0.5.8 (2026-11-22) | ~12.5k / 2026-11-23 | KV offload/sharing |
| KV-cache / transfer | [Mooncake](https://github.com/kvcache-ai/Mooncake) | kvcache-ai (Kimi) | v0.3.14 (2026-11-03) | ~7.1k / 2026-11-24 | KV-centric serving platform |
| KV transfer library | [NIXL](https://github.com/ai-dynamo/nixl) | NVIDIA | v1.5.2 (2026-10-29) | ~1.4k / 2026-11-22 | Used by Dynamo and llm-d |
| KV scheduling/offload | [llm-d-kv-cache](https://github.com/llm-d/llm-d-kv-cache) | llm-d | v1.0.0 (2026-10-16) | ~200 / 2026-11-24 | KV-cache-aware scheduling |
| Quantization | [LLM Compressor](https://github.com/vllm-project/llm-compressor) | vLLM project (Red Hat) | 0.15.0 (2026-10-29) | ~4.0k / 2026-11-24 | Compress for vLLM serving |
| App-level LLM observability | [Langfuse](https://github.com/langfuse/langfuse) | Langfuse | v4.84.0 (2026-11-23) | ~37.8k / 2026-11-24 | Tracing/evals; now also an AI gateway |
| | [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Arize | client v3.8.0 (2026-11-23) | ~12.4k / 2026-11-24 | Observability & evaluation |
| | [OpenLLMetry](https://github.com/traceloop/openllmetry) | Traceloop | 0.63.1 (2026-11-16) | ~7.6k / 2026-11-23 | OpenTelemetry-based |
| MLOps platform | [MLflow](https://github.com/mlflow/mlflow) | Databricks/LF | v3.18.0 (2026-11-04) | ~29k / 2026-11-24 | Now positioned for "agents, LLMs, and ML models" |
| MLOps platform | [ZenML](https://github.com/zenml-io/zenml) | ZenML | 0.99.0 (2026-11-19) | ~5.6k / 2026-11-24 | Pipelines to agents |
| MLOps platform | [Kubeflow](https://github.com/kubeflow/kubeflow) | CNCF | n/a | ~16k / 2026-09-30 | "The Cloud Native AI Platform" |

Sources for the table: each project's GitHub link above (metadata via [ungh.cc](https://ungh.cc/repos/llm-d/llm-d)).

#### Layer details
- **Serving control plane: KServe.** v0.17 (2026-03-13) promoted LLMInferenceService from experimental to a production-ready CRD built on llm-d. v0.18 (2026-04-29) added multi-node inference without Ray, LWS-based autoscaling for multi-node workloads, OpenAI Responses API routing, a vLLM v0.19 upgrade and llm-d v0.6 integration. v0.19 (2026-06-14) added static LoRA adapters, model-name routing and graceful shutdown. v0.20 (2026-08-06) added confidential model serving, traffic splitting/canary and KV-cache offloading. v0.21 (2026-11-04) wires KEDA directly to LLMInferenceService, including scale-to-zero, runs the vLLM tokenizer as a standalone deployment, and adds canary traffic splitting on RawDeployment HTTPRoutes plus LoRA support in LocalModelCache. — [KServe blog index](https://kserve.github.io/website/blog); [KServe 0.17](https://kserve.github.io/website/blog/kserve-0.17-release); [KServe 0.18](https://kserve.github.io/website/blog/kserve-0.18-release)
- KServe manifests appear as `serving.kserve.io/v1alpha1` in early docs and as `v1alpha2` in an August 2026 Red Hat article, so learners must match their CRD version. — [KServe llmisvc overview](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview); [Red Hat 2026-08-05](https://developers.redhat.com/articles/2026/08/05/trace-kubernetes-resources-llm-d-model-serving)
- Thoughtworks Technology Radar placed KServe in the "Trial" ring (Nov 2026). — [Thoughtworks Radar KServe](https://www.thoughtworks.com/es-cl/radar/platforms/kserve)
- **Ray Serve LLM** provides an OpenAI-compatible API for chat, completions and embeddings. It supports multi-node and multi-model deployment with autoscaling, tensor/pipeline/expert/data-parallel attention, prefill–decode disaggregation, prefix-aware routing, multi-LoRA, vLLM and SGLang backends, and built-in metrics with Grafana dashboards. — [Ray Serve LLM docs](https://docs.ray.io/en/latest/serve/llm/index.html)
- **Triton / Dynamo-Triton.** Since 2025-03-18 NVIDIA has branded Triton Inference Server as "NVIDIA Dynamo-Triton", but code and containers keep the Triton name. NVIDIA's pages variously call Dynamo the successor to Triton or a complement to it. — [NVIDIA Dynamo-Triton page](https://developer.nvidia.com/dynamo-triton)
- **BentoML.** Modular announced its acquisition of BentoML on 2026-02-10. BentoML stays Apache-2.0. — [Modular blog](https://modular.com/blog/bentoml-joins-modular); [BentoML blog](https://bentoml.com/blog/bentoml-is-joining-modular)
- **Routing layer: GIE.** InferencePool has been GA since v1.0.0. The alpha InferenceModel API was replaced by InferenceObjective. Since v1.5.0, `endpointPickerRef` is optional, so an InferencePool can work without a user-managed Endpoint Picker (EPP). GKE documents migrating from v1alpha2 to v1. — [GIE InferencePool API](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/); [GKE migration guide](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/migrate-gke-inference-gateway)
- The GIE implementations page lists Alibaba Cloud ACK, Istio (v1.28.0+ recommended), agentgateway (Rust), Higress and NGINX Gateway Fabric. GKE Inference Gateway is documented by Google, and NGINX documents its own support. — [GIE implementations](https://gateway-api-inference-extension.sigs.k8s.io/implementations/gateways/); [GKE Inference Gateway tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-with-gke-inference-gateway); [NGINX docs](https://docs.nginx.com/nginx-gateway-fabric/how-to/gateway-api-inference-extension/)
- Background: the June 2025 Kubernetes blog introduction to GIE remains the conceptual primer. — [Kubernetes blog 2025-06-05](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/)
- **AI gateway: Agent Router.** Envoy AI Gateway was renamed "Agent Router" and moved to the Agentic AI Foundation (AAIF) in September 2026 (AAIF dates it Sept 9; TrueFoundry Sept 10). Code, maintainers, the Apache-2.0 license, CRDs and the `aigw` CLI were unchanged, and images, the Helm chart and the Go module path still use envoyproxy/ai-gateway names. v1.2.0 (Oct 9, 2026 per Tetrate) was the first release under the new name. — [AAIF blog](https://aaif.io/blog/agent-router-joins-aaif); [Tetrate](https://tetrate.io/blog/agent-router-joins-aaif); [TrueFoundry review](https://www.truefoundry.com/blog/envoy-ai-gateway-review)
- **vLLM Semantic Router.** It routes across mixed fleets (local vLLM plus cloud APIs) using signals and safety/privacy policies, and the project is moving toward building "Mixture-of-Models" systems. Releases: v0.1 "Iris" (Jan 2026), v0.2 "Athena" (Mar 2026), v0.3 "Themis" (Jun 2026). GitHub shows v0.4.0 (2026-09-27). — [vLLM blog 2026-07-21](https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom); [arXiv 2603.04444](https://arxiv.org/abs/2603.04444); [repo](https://github.com/vllm-project/semantic-router)
- **vLLM Router.** vLLM published a separate router described as "high-performance and prefill/decode aware". — [vLLM blog](https://vllm.ai/blog/vllm-router-release)
- **Orchestration: LWS and DisaggregatedSet.** DisaggregatedSet is a controller/CRD (KEP-766) that extends LWS for disaggregated inference. Each role, such as prefill or decode, becomes its own child LeaderWorkerSet with default per-role rollout maxSurge 0 / maxUnavailable 1. The API is `disaggregatedset.x-k8s.io/v1`. — [LWS DisaggregatedSet concept](https://lws.sigs.k8s.io/docs/concepts/disaggregatedset/); [API ref](https://lws.sigs.k8s.io/docs/reference/disaggregatedset.v1/)
- A maintainer post says DisaggregatedSet reached v1 alongside llm-d's v1. GitHub shows lws v1.0.0 on 2026-10-16, with the repo now describing itself as "LeaderWorkerSet & DisaggregatedSet". — [LinkedIn post](https://cn.linkedin.com/in/yankay); [LWS repo](https://github.com/kubernetes-sigs/lws)
- **Autoscaling guidance (GKE).** Use queue size as the main HPA metric for throughput/cost: start at a threshold of 3–5 and raise it until latency is acceptable. Use batch size for latency-sensitive workloads. Avoid GPU utilization, which overprovisions and is cost-inefficient in Google's benchmarks. The HPA default tolerance is 0.1. — [GKE autoscaling best practices](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling); [GKE blog: tuning HPA](https://cloud.google.com/blog/products/containers-kubernetes/tuning-the-gke-hpa-to-run-inference-on-gpus); [GKE how-to](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/autoscaling)
- **Multi-cloud serving: SkyPilot.** `sky serve` runs vLLM as autoscaled replicas, with `replica_policy` min/max replicas and `target_qps_per_replica`, load balancing, readiness probes and spot instances, across clouds and Kubernetes. — [vLLM docs: SkyPilot](https://docs.vllm.ai/en/stable/deployment/frameworks/skypilot/)
- **Observability: vLLM.** The OpenAI-compatible server exposes Prometheus metrics at `/metrics`. vLLM ships a Prometheus/Grafana docker-compose example and two dashboards ("Performance Statistics" and "Query Statistics"), also in Perses format. Metric names have changed across versions, so docs must match the deployed version. — [vLLM Prometheus/Grafana example](https://docs.vllm.ai/en/stable/examples/observability/prometheus_grafana/); [vLLM dashboards](https://docs.vllm.ai/en/stable/examples/observability/dashboards/)
- A community Grafana dashboard (ID 25043) points to the production-stack Helm dashboard. — [Grafana dashboard 25043](https://grafana.com/grafana/dashboards/25043)
- **Observability: OpenTelemetry GenAI semantic conventions.** These were still "Development" (not stable) as of mid-to-late 2026. On 2026-06-12 (core semconv v1.42.0) they moved to a separate repo, semantic-conventions-genai, which had no official release as of 2026-08-21. Vendors such as Datadog implement them anyway. No public stabilization date was found. — [dev.to analysis](https://dev.to/azena-ai/opentelemetrys-genai-semantic-conventions-are-not-stable-yet-heres-what-actually-shipped-in-2026-3mke); [semantic-conventions-genai repo](https://github.com/open-telemetry/semantic-conventions-genai); [OTel GenAI spans spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/); [Dash0 explainer, updated 2026-09-14](https://www.dash0.com/knowledge/opentelemetry-genai-semantic-conventions-explained)
- **Distributed tracing in llm-d.** llm-d published an end-to-end and fine-grained tracing post ("Seeing Through the Stack"; date not shown on the index). Red Hat published "Trace Kubernetes resources in llm-d model serving" on 2026-08-05. — [llm-d blog](https://llm-d.ai/blog); [Red Hat](https://developers.redhat.com/articles/2026/08/05/trace-kubernetes-resources-llm-d-model-serving)

### Inferences
- "Router" now means two layers that learners should keep separate. The upper layer is an L7 AI gateway (LiteLLM, Agent Router, Portkey, kgateway/agentgateway) handling provider abstraction, auth, quotas and fallbacks. The lower layer is inference-aware endpoint picking (GIE EPP implementations such as llm-d-router, Dynamo's EPP, the AIBrix gateway plugin, vLLM Router) handling prefix-cache, KV and load-aware placement. Production designs typically stack them.
- Many 2024–2025 tutorials name components that have since been renamed or deprecated: Envoy AI Gateway → Agent Router, InferenceModel → InferenceObjective, GenAI-Perf → AIPerf, InferenceMAX → InferenceX, TGI → maintenance mode, Triton → Dynamo-Triton. A self-learner should check every tutorial's date and API version before use.
- The KServe GitHub "latest release" date (2026-09-25) and its v0.21 announcement blog (2026-11-04) differ. This is probably a tag published before the blog, or a release-candidate versus final distinction; I have not verified which.

### Gaps
- No neutral adoption or market-share data for each layer, such as how many production clusters run llm-d versus Dynamo versus KServe. Star counts are only a weak proxy.
- I could not verify licenses for every repo, and could not retrieve the GIE Endpoint Picker/Body-Based Router concept pages (the implementations page fetched did not describe them).
- I did not verify whether any InferencePool/EPP conformance list includes GKE, kgateway or Agent Router as of Nov 2026. The fetched implementations page lists only ACK, Istio, agentgateway, Higress and NGINX Gateway Fabric.

## 2. How do the Kubernetes-native distributed inference projects (llm-d, Dynamo, AIBrix, production-stack, KServe, Ray Serve) compare in architecture and maturity, and what quickstarts, tutorials, workshops and talks exist?

### Takeaway
- **llm-d (CNCF Sandbox, 1.0 in Oct 2026)** is the "upstream-Kubernetes-native" option. It is composable "well-lit paths" on vLLM/SGLang, GIE, LWS and NIXL, and is now also KServe's LLM engine.
- **NVIDIA Dynamo (1.0 in Mar 2026, v1.6.2 now)** is a vertically integrated, engine-agnostic framework with its own frontend, KV router, Planner, KVBM, NIXL, Grove and Operator. It can also sit behind GIE.
- **AIBrix (v0.8, Oct 2026)** is a broad ByteDance-derived control plane covering gateway, StormService multi-node/P-D, LoRA, batch, console and GPU sharing.
- **vLLM production-stack** is the lightest reference Helm chart and the easiest place to start.
- **KServe** is the CNCF model-serving control plane/CRD (LLMInferenceService built on llm-d).
- **Ray Serve LLM** is the Python-first alternative.

Learning material is plentiful but scattered across project docs, blogs, Red Hat Developer articles, cloud blueprints, bi-weekly vLLM office hours and KubeCon/PyTorch/GTC talks.

### Cited Findings

#### Comparison matrix

| Dimension | llm-d | NVIDIA Dynamo | AIBrix | vLLM production-stack | KServe (LLMInferenceService) | Ray Serve LLM |
|---|---|---|---|---|---|---|
| Origin / governance | Red Hat-led, with IBM, Google, NVIDIA and others; >500 contributors from >25 orgs; CNCF Sandbox since 2026-03-24 ([llm-d blog](https://llm-d.ai/blog); [CNCF](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/)) | NVIDIA open source; production GA at GTC 2026 ([NVIDIA news](https://nvidianews.nvidia.com/news/dynamo-1-0)) | ByteDance origin, under vllm-project ([vLLM blog](https://vllm.ai/blog/aibrix-release); [arXiv](https://arxiv.org/html/2504.03648v1)) | vllm-project / LMCache team; first release 2025-01-22 ([vLLM blog](https://vllm.ai/blog/stack-release)) | CNCF incubating ([CNCF](https://www.cncf.io/blog/2025/11/11/kserve-becomes-a-cncf-incubating-project/)) | Ray / Anyscale ([docs](https://docs.ray.io/en/latest/serve/llm/index.html)) |
| Maturity (latest) | v1.0 on 2026-10-20: stable v1 config API, multi-tenant QoS, zero-downtime upgrades, agent-aware serving ([llm-d blog](https://llm-d.ai/blog); [README](https://cdn.jsdelivr.net/gh/llm-d/llm-d@main/README.md)) | v1.0.0 GA 2026-03-12; v1.6.2 on 2026-11-05 ([release notes](https://docs.nvidia.com/dynamo/v1.4.0/reference/releases/v1-0-0); [repo](https://github.com/ai-dynamo/dynamo)) | v0.8.0 on 2026-10-12 ([AIBrix posts](https://aibrix.github.io/posts/)) | Helm chart 0.1.14 on 2026-11-07; 2026 roadmap issue #855 ([repo](https://github.com/vllm-project/production-stack); [roadmap](https://github.com/vllm-project/production-stack/issues/855)) | v0.21 on 2026-11-04 ([KServe blog](https://kserve.github.io/website/blog)) | ray-2.60.0 on 2026-11-04 ([repo](https://github.com/ray-project/ray)) |
| Core architecture | Integrates vLLM/SGLang with K8s; prefix-cache and load-aware routing; tiered KV offload to CPU/disk; P/D disaggregation; wide EP; session-aware routing; SLO-aware autoscaling; Batch API ([README](https://cdn.jsdelivr.net/gh/llm-d/llm-d@main/README.md)) | Frontend → KV-aware Router → prefill/decode workers; Planner (throughput- or load-based scaling); KVBM (block reuse/offload across tiers); Operator reconciles DynamoGraphDeployment; Grove PodCliqueSet for multinode ([architecture](https://docs.nvidia.com/dynamo/latest/design-docs/overall-architecture)) | Gateway (routing, session pinning), StormService/RoleSet (multi-node, P/D), ModelClaim/ModelAdapter (lifecycle, LoRA), batch platform, console, Volcano gang scheduling ([v0.8.0](https://aibrix.github.io/posts/2026-10-12-v0.8.0-release/)) | Router + vLLM engines + Prometheus/Grafana via the `vllm/vllm-stack` Helm chart; LMCache offload tutorial ([README](https://cdn.jsdelivr.net/gh/vllm-project/production-stack@main/README.md)) | K8s control plane CRD built on llm-d ([KServe 0.17](https://kserve.github.io/website/blog/kserve-0.17-release)) | Ray Serve primitives; OpenAI API; P/D; prefix-aware routing ([docs](https://docs.ray.io/en/latest/serve/llm/index.html)) |
| Engines | vLLM and SGLang; first-class SGLang post 2026-11-23 ([llm-d blog](https://llm-d.ai/blog)) | SGLang, TensorRT-LLM, vLLM ([NVIDIA news](https://nvidianews.nvidia.com/news/dynamo-1-0)) | vLLM, SGLang endpoints, TensorRT-LLM integration, vLLM-Omni ([v0.8.0](https://aibrix.github.io/posts/2026-10-12-v0.8.0-release/)) | vLLM ([README](https://cdn.jsdelivr.net/gh/vllm-project/production-stack@main/README.md)) | vLLM (via llm-d) ([KServe 0.18](https://kserve.github.io/website/blog/kserve-0.18-release)) | vLLM, SGLang ([docs](https://docs.ray.io/en/latest/serve/llm/index.html)) |
| Accelerators | Validated on NVIDIA, AMD and Intel GPUs and Google TPUs ([README](https://cdn.jsdelivr.net/gh/llm-d/llm-d@main/README.md)) | NVIDIA-focused (Blackwell claims) ([NVIDIA blog](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready)) | n/a in fetched sources | GPUs on any K8s ([README](https://cdn.jsdelivr.net/gh/vllm-project/production-stack@main/README.md)) | Inherits llm-d | GPUs |
| Gateway relation | Built on GIE; recommended production setup puts a general Gateway API implementation at the edge with llm-d's inference gateway beneath ([The New Stack](https://thenewstack.io/six-frameworks-for-efficient-llm-inferencing/)) | Standalone KV router by default; "gateway mode" runs the Dynamo EPP behind a GIE gateway (documented from v1.2) ([architecture](https://docs.nvidia.com/dynamo/latest/design-docs/overall-architecture)) | Envoy-based gateway; Envoy sidecar mode since v0.6 ([AIBrix v0.6.0](https://aibrix.github.io/posts/2026-03-03-v0.6.0-release)) | Own router; KV-aware routing, P/D and metric-based autoscaling listed as roadmap in README ([README](https://cdn.jsdelivr.net/gh/vllm-project/production-stack@main/README.md)) | HTTPRoute/Gateway API; canary splitting ([KServe blog](https://kserve.github.io/website/blog)) | Ray Serve ingress |
| GitHub stars (approx.) | ~5.3k ([repo](https://github.com/llm-d/llm-d)) | ~8.6k ([repo](https://github.com/ai-dynamo/dynamo)) | ~5.2k ([repo](https://github.com/vllm-project/aibrix)) | ~2.7k ([repo](https://github.com/vllm-project/production-stack)) | ~6.3k ([repo](https://github.com/kserve/kserve)) | ~44.4k for all of Ray ([repo](https://github.com/ray-project/ray)) |

#### llm-d details and learning materials
- Release cadence:

| Version | Date | Highlights |
|---|---|---|
| v0.4 | Dec 2025 | Intel XPU and Google TPU disaggregation support |
| v0.5 | Feb 2026 | Hierarchical KV offloading, active-active HA, reproducible benchmark workflows |
| v0.6.0 | 2026-04-03 | — |
| v0.7 | May 2026 | Predicted-latency scheduling GA |
| v0.8 | — | "From Platform to Control Plane" |
| v0.9 | 2026-08-28 | "Hardened for Scale" |
| v0.10 | Sep 2026 | Operational hardening |
| v1.0 | 2026-10-20 | — |

  Sources: [llm-d README](https://cdn.jsdelivr.net/gh/llm-d/llm-d@main/README.md); [llm-d latest-release doc](https://llm-d.ai/docs/architecture/latest-release); [llm-d blog](https://llm-d.ai/blog)
- llm-d 1.0 was announced at PyTorch Conference North America (San Jose). One report says 1.0 aligned components at v1.0.0 and added a peer-to-peer model-weight-loading integration focused on Day-2 operations. — [Unite.AI](https://unite.ai/llm-d-1-0-debuts-at-pytorch-conference-with-production-inference-stack/); [llm-d GitHub](https://github.com/llm-d/llm-d)
- Current well-lit paths in the README: Optimized Baseline (the default start), Tiered Prefix Cache (KV offload for multi-turn), Wide Expert Parallelism (large MoE over fast interconnects), Agentic workloads, and Batch serving. The original v0.2 paths were intelligent inference scheduling, P/D disaggregation and wide EP. Quickstart: https://llm-d.ai/docs/getting-started/quickstart. — [README](https://cdn.jsdelivr.net/gh/llm-d/llm-d@main/README.md); [CNCF sandbox application](https://github.com/cncf/sandbox/issues/462)
- 2026 blog posts double as tutorials:

| Date | Post |
|---|---|
| 2026-11-23 | First-class SGLang |
| 2026-11-16 | Multi-tenant/multi-priority serving |
| 2026-11-09 | Agent trace replay |
| 2026-10-29 | P/D on GKE topology |
| 2026-09-14 | RL post-training time-slicing |
| 2026-08-31 | Multimodal |
| undated | KV offload to any filesystem |
| undated | "No Kubernetes? No Problem: llm-d Now Runs Anywhere" |
| undated | BLIS simulation |
| undated | Heterogeneous serving across three GPU vendors |

  Source: [llm-d blog](https://llm-d.ai/blog)
- Red Hat Developer llm-d material (English, free):

| Date | Resource | Notes |
|---|---|---|
| 2025-08-19 | "Getting started with llm-d" | Hands-on; needs OpenShift 4.17+ and NVIDIA GPU Operator 25.3 |
| 2025-09-08 | Wide-EP for DeepSeek-style MoEs | — |
| 2025-11-21 | "Introduction to distributed inference with llm-d" | Concepts and well-lit paths |
| May 2026 | Browser-based interactive demo | — |
| 2026-06-11 | "Intelligent inference scheduling" | — |
| 2026-06-26 | Troubleshooting distributed inference blueprints | — |

  Sources: [Getting started](https://developers.redhat.com/articles/2025/08/19/getting-started-llm-d-distributed-ai-inference); [Wide-EP](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep); [Intro](https://developers.redhat.com/articles/2025/11/21/introduction-distributed-inference-llm-d); [Interactive demo](https://www.redhat.com/en/interactive-demo/introduction-llm-d); [Scheduling](https://developers.redhat.com/articles/2026/06/11/intelligent-inference-scheduling-llm-d-red-hat-ai); [Troubleshooting](https://developers.redhat.com/index%2Ephp/articles/2026/06/26/deploying-distributed-ai-inference-blueprints-troubleshooting)
- Productized docs: Red Hat OpenShift AI 3.4/3.5 "Deploying models by using Distributed Inference with llm-d". — [RHOAI 3.5 docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploy_models_using_distributed_inference_with_llm-d/deploying-models-using-distributed-inference_distributed-inference)
- KServe and llm-d material: Yuan Tang's blog posts (2026-03-05, "Best of Both Worlds"; 2026-04-21, "Production-Grade LLM Inference at Scale with KServe, llm-d, and vLLM") and the Red Hat article (2026-04-21). — [terrytangyuan 03-05](https://terrytangyuan.github.io/2026/03/05/cloud-native-ai-inference-kserve-llm-d); [terrytangyuan 04-21](https://terrytangyuan.github.io/2026/04/21/production-grade-llm-inference-at-scale-kserve-llm-d-vllm/); [Red Hat](https://developers.redhat.com/articles/2026/04/21/kserve-llm-d-optimized-gen-ai-inference)
- Third-party explainers: Ubuntu's "Understanding disaggregated GenAI model serving with llm-d", AWS's "Introducing disaggregated inference on AWS powered by llm-d" (SageMaker HyperPod EKS), and IBM Research's CNCF-donation post. — [Ubuntu](https://ubuntu.com/blog/understanding-disaggregated-genai-model-serving-with-llm-d); [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d); [IBM Research](https://research.ibm.com/blog/donating-llm-d-to-the-cloud-native-computing-foundation)
- llm-d uses NVIDIA NIXL for KV-cache transfer between prefill and decode workers. — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference)

#### NVIDIA Dynamo details and learning materials
- Dynamo 1.0 was announced at GTC as production-ready. NVIDIA claims "up to 7x" Blackwell inference gains in industry benchmarks (vendor-reported). Named users include Cursor, Perplexity and Together AI, and AWS, Azure, Google Cloud and OCI have integrations. — [NVIDIA newsroom](https://nvidianews.nvidia.com/news/dynamo-1-0); [NVIDIA dev blog](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready)
- Kubernetes quickstart prerequisites: a K8s cluster (v1.24+ in v1.2 docs, v1.30+ in newer dev docs) with GPU nodes, kubectl, Helm and the GPU Operator. There are two paths. A DynamoGraphDeploymentRequest (DGDR) auto-profiles the model and hardware and generates a DGD; alternatively you apply a DGD manifest directly. The quickstart example model is Qwen/Qwen3-0.6B, and "recipes" are tuned DGD manifests for vLLM, SGLang and TRT-LLM. Docs are versioned (v1.0–v1.6) and also available in zh-CN. — [Dynamo K8s quickstart](https://docs.nvidia.com/dynamo/kubernetes-deployment/start-here/kubernetes-quickstart); [Deploy with DGD (zh-CN v1.5)](https://docs.nvidia.com/dynamo/zh-CN/v1.5.0/kubernetes/model-deployment/deploy-with-dgd)
- AIConfigurator ("Offline optimization of your disaggregated Dynamo graph") is at v0.12.0 (2026-09-18) with ~470 stars, and serves as a capacity-planning helper. — [AIConfigurator repo](https://github.com/ai-dynamo/aiconfigurator)
- Cloud blueprints: the AWS "AI on EKS" Dynamo blueprint covers disaggregated vLLM/SGLang/TRT-LLM and KV-aware routing, but one of its pages is flagged "This blueprint is currently outdated". AWS also has a Dynamo+EKS ML blog post. — [AI on EKS Dynamo (outdated flag)](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/framework-guides/GPUs/nvidia-dynamo); [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-with-nvidia-dynamo-and-amazon-eks)

#### AIBrix details and learning materials
- Release posts:

| Version | Date | Highlights |
|---|---|---|
| v0.4.0 | 2025-08-04 | P/D, EP, KVCache connector, multi-engine |
| v0.5.0 | 2025-11-10 | OpenAI-compatible Batch API, KVCache v1 connector, P/D orchestration |
| v0.6.0 | 2026-03-03 | Envoy sidecar, routing profiles, LoRA delivery |
| v0.7.0 | 2026-06-16 | Management console, self-hosted batch, KV-centric disaggregation, HA gateway |
| v0.8.0 | 2026-10-12 | See below |

  v0.8.0 details: per-replica RPS and in-flight limits; a least-request-top-k strategy; Redis session pinning for agentic traffic; experimental external routing; experimental ModelClaim GPU sharing and model sleep/relocation; drain-aware StormService rollouts; Volcano gang scheduling; AIMD batch concurrency.

  Sources: [AIBrix posts](https://aibrix.github.io/posts/); [v0.5.0](https://aibrix.github.io/posts/2025-11-10-v0.5.0-release); [v0.7.0](https://aibrix.github.io/posts/2026-06-16-v0.7.0-release/); [v0.8.0](https://aibrix.github.io/posts/2026-10-12-v0.8.0-release/)
- Practitioner posts: "Scaling LLM Inference in Production: Lessons Learned with AIBrix Gateway" (2026-10-19) and "Should You Self-Host Batch Inference? An Honest Cost Breakdown" (2026-06-22). The design paper is arXiv 2504.03648. — [AIBrix posts](https://aibrix.github.io/posts/); [arXiv](https://arxiv.org/html/2504.03648v1)
- The AWS AI on EKS inference blueprints include AIBrix, alongside Ray Serve, Triton, vLLM and Dynamo. — [AI on EKS inference](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference)

#### vLLM production-stack details and learning materials
- The README tutorials run from 0 (install kubectl/helm/minikube) through cloud deployment (AWS, GCP, Lambda Labs, Azure), a minimal stack, custom configs, loading weights, multiple models, and 6 (KV-cache offloading with LMCache). The router supports round-robin and session-ID routing; prefix-aware routing is marked WIP, while KV-aware routing, P/D and vLLM-metric autoscaling are roadmap items. The README may lag the code. — [production-stack README](https://cdn.jsdelivr.net/gh/vllm-project/production-stack@main/README.md)
- The 2026 roadmap prioritizes router work and a release bot for Helm, controller and Docker artifacts. — [Issue #855](https://github.com/vllm-project/production-stack/issues/855)

#### Ray Serve LLM learning materials
- Anyscale Academy "LLM Serving Foundations" is free, with 24 lessons and about 2 hours of video. It covers LLM-serving concepts, Ray Serve LLM setup, local deployment, Anyscale Services and tool calling. A free "Model Serving Foundations" (Ray Serve) course has 7 lessons. Open notebooks are on ray-project.github.io/enablement-content. — [LLM Serving Foundations](https://courses.anyscale.com/courses/llm-serving-foundations); [Model serving collection](https://courses.anyscale.com/collections/model-serving); [enablement notebook](https://ray-project.github.io/enablement-content/courses/foundations/LLM_Serving/00_intro_serve_llm/output/notebook_01.html)
- Legacy Ray training repos are stale. ray-project/ray-educational-materials was last pushed 2024-02-13. anyscale/academy's latest release tag dates from 2020, though the repo was pushed 2026-06-25. — [ray-educational-materials](https://github.com/ray-project/ray-educational-materials); [anyscale/academy](https://github.com/anyscale/academy)

#### Other Kubernetes operators and cloud labs
- **KAITO (Azure).** You declare a "Workspace" CR naming the model and GPU instance type. KAITO provisions GPU nodes through Karpenter NodePools, estimates memory, tunes vLLM and exposes an OpenAI-compatible Service. GPU auto-provisioning is most mature on AKS and limited to Ampere-or-newer NVIDIA GPUs (Thoughtworks Radar, Nov 2026). The AKS add-on went GA in September 2025. Microsoft docs still cite KAITO 0.4.x while GitHub is at v0.13.0, so the docs may be stale. — [Thoughtworks KAITO](https://www.thoughtworks.com/radar/platforms/kaito); [AKS docs](https://learn.microsoft.com/en-us/azure/aks/ai-toolchain-operator); [GA notice](https://mc.merill.net/message/AZ503263); [KAITO repo](https://github.com/kaito-project/kaito)
- **GKE.** Tutorials cover Gemma (now including Gemma 4) with vLLM on B200/H100/RTX Pro 6000/L4 on Autopilot or Standard; Llama 4 with vLLM (needs H100/H200 quota); Qwen3 and Gemma 3 27B (TP across 8 GPUs); a Terraform reference architecture; and GKE Inference Gateway. The GKE Inference Quickstart tool compares serving performance and cost. — [Gemma vLLM on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm); [Llama vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llama-gpus-vllm); [Qwen3](https://docs.cloud.google.com/ai-hypercomputer/docs/tutorials/gpu/qwen3-vllm-inference); [Terraform architecture](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-open-models-terraform); [Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-with-gke-inference-gateway)
- Google's sample repos: GoogleCloudPlatform/kubernetes-engine-samples (~1.4k stars, pushed 2026-11-10) and accelerated-platforms (~110 stars, pushed 2026-11-24). — [kubernetes-engine-samples](https://github.com/GoogleCloudPlatform/kubernetes-engine-samples); [accelerated-platforms](https://github.com/GoogleCloudPlatform/accelerated-platforms)
- **AWS.** "AI on EKS" (awslabs, ~230 stars, pushed 2026-11-11) provides Terraform templates and blueprints, and the EKS User Guide has "Quickstart: High-throughput LLM inference with vLLM on Amazon EKS". — [AI on EKS intro](https://aws.amazon.com/blogs/containers/introducing-ai-on-eks-powering-scalable-ai-workloads-with-amazon-eks); [EKS vLLM quickstart](https://docs.aws.amazon.com/eks/latest/userguide/ml-realtime-inference-llm-inference-vllm.html); [repo](https://github.com/awslabs/ai-on-eks)

#### Talks, workshops and community series
- **vLLM Office Hours** run live every other Thursday, with an archive on vllm.ai. 2026 sessions include #48 tool calling (Apr 30), #51 vLLM v0.22 and speculators (Jun 11), #52 semantic router (Jun 25), #55 Mooncake + vLLM/llm-d (Aug 6), #56 GuideLLM v0.7 benchmarking (Aug 20), #57 vLLM-Omni (Sep 3) and #58 llm-d semantic classifier (Oct 1). Many feature Red Hat speakers. — [vLLM office hours](https://vllm.ai/office-hours)
- **PyTorch Conference NA 2026** (Oct 20) was the llm-d 1.0 launch. **KubeCon NA 2026** (Nov 9–12) had the AI Inference + Agentic track. — [Unite.AI](https://unite.ai/llm-d-1-0-debuts-at-pytorch-conference-with-production-inference-stack/); [CNCF](https://www.cncf.io/announcements/2026/08/10/cncf-reveals-kubecon-cloudnativecon-north-america-2026-schedule-adds-new-ai-inference-agentic-track/)
- The **first SGLang Summit** (Nov 12–13, 2026, Fort Mason, San Francisco) drew 2,000+ attendees. — [LMSYS recap](https://www.lmsys.org/blog/2026-11-17-sglang-summit-recap)
- **NVIDIA GTC.** An instructor-led workshop "Deploying and Optimizing AI Inference at Scale" is listed on NVIDIA's workshop pages. A GTC 2026 session, S82474 "Continuous Inference Benchmarking of the AI Frontier", covers InferenceMAX/X. — [NVIDIA instructor-led workshops](https://www.nvidia.com/en-in/training/instructor-led-workshops/); [GTC26 S82474](https://www.nvidia.com/assets/nvod/md/gtc26-s82474.md)
- **"State of the Model Serving Communities"** is a monthly Substack (InferenceOps). The newest issue found is February 2026. — [Feb 2026 issue](https://inferenceops.substack.com/p/state-of-the-model-serving-communities-f81)

### Inferences
- Maturity ranking for a learner in late 2026, by stable API and cadence:
  1. KServe (CNCF incubating, LLMInferenceService GA) ≈ Ray Serve LLM (part of Ray core).
  2. Dynamo 1.x (GA since March, six minor releases).
  3. llm-d 1.0 (GA in October, but Sandbox-stage governance).
  4. AIBrix 0.8 (pre-1.0, feature-rich).
  5. production-stack (0.1.x chart, reference quality).
- On architecture, llm-d and KServe bet on upstream Kubernetes APIs (GIE, LWS/DisaggregatedSet), while Dynamo brings its own operator and graph CRDs (DGD/DGDR, Grove) and can optionally plug into GIE. The convergence points are vLLM/SGLang engines, NIXL for KV transfer and GIE for routing, so concepts learned in one project transfer well.
- Suggested hands-on progression:
  1. production-stack on minikube.
  2. A GKE/EKS vLLM tutorial.
  3. KServe LLMInferenceService or the llm-d Optimized Baseline.
  4. P/D disaggregation via llm-d or Dynamo recipes.
  5. Wide-EP and tiered KV cache.

### Gaps
- I found no neutral, peer-reviewed head-to-head benchmark of llm-d vs Dynamo vs AIBrix vs production-stack. A particula.tech comparison exists but I did not verify it. InferenceX includes "Dynamo (SGLang/TRT-LLM/vLLM)" and "llm-d vLLM" configurations, which may be the closest public comparison. — [particula.tech](https://particula.tech/blog/llm-d-vs-dynamo-vs-vllm-production-stack); [InferenceX about](https://inferencex.semianalysis.com/about)
- I could not fetch the full llm-d 1.0 "Day 2" post (only its excerpt) or llm-d.ai/docs/guide (HTTP 404), so the exact list of 1.0 well-lit paths comes from the README.
- No verified schedule or recordings for 2026 Ray Summit, llm-d community meetings, or KubeCon tutorial titles. Dynamo-specific NVIDIA DLI self-paced courses were not found.

## 3. How do practitioners benchmark and capacity-plan LLM serving (metrics, tools, methodology, public benchmark efforts), and what resources teach this?

### Takeaway
Practice in 2026 revolves around:
- **Latency:** TTFT and ITL/TPOT at p50/p99.
- **Throughput:** tokens/s/GPU and tokens/s/user, plotted as a Pareto frontier.
- **SLO-based goodput.**
- **Cost per million tokens** and joules per token.

The standard open tools are `vllm bench serve/sweep`, GuideLLM (v0.8 added SLO/goodput search), NVIDIA AIPerf (which replaced GenAI-Perf), Kubernetes' inference-perf and llm-d-benchmark. Ray's LLMPerf is archived. Public reference points are MLPerf Inference v6.0 (Apr 2026) and v6.1 (Sep 2026), and SemiAnalysis's continuous, open-source InferenceX (formerly InferenceMAX), which added an agentic "AgentX" workload. For sizing and TCO the best teaching resources are NVIDIA's blogs and the DLI "Sizing LLM Inference Systems" course, GKE's autoscaling guide, and the DeepLearning.AI/Red Hat vLLM course labs.

### Cited Findings

#### Metric definitions and pitfalls
- TTFT includes queueing, prefill and network latency, and grows with prompt length. ITL/TPOT excludes the first token, so the denominator subtracts 1. Tools differ: GenAI-Perf excludes TTFT from its average while LLMPerf includes it, so numbers are not comparable across tools without checking definitions. — [NVIDIA "LLM Inference Benchmarking: Fundamental Concepts"](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts); [NIM benchmarking metrics](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics)
- AIPerf docs warn that TTFT and output-token metrics are defined differently for reasoning models, so comparisons with GenAI-Perf need the equivalent AIPerf metrics. — [AIPerf migration guide](https://docs.nvidia.com/aiperf/getting-started/migrating-from-gen-ai-perf)
- Goodput counts requests that meet all SLO thresholds at once. In GuideLLM v0.8.0, SLOs are passed as `--metrics` (e.g., `ttft_ms`, `tpot_ms`) and `--profile kind=goodput,target_attainment=0.99` doubles concurrency until the SLO fails, then binary-searches. The source write-up notes that the code default is 0.95 while the CLI example uses 0.99. (Secondary Korean-language source; v0.8.0 dated 2026-09-30; v0.8.1 added knee detection.) — [PyTorch Korea write-up](https://discuss.pytorch.kr/t/guidellm-v0-8-0-slo-feat-vllm/12149)
- The InferenceX metrics set:
  - interactivity (tok/s/user ≈ 1/ITL)
  - E2E-normalized interactivity
  - tok/s per chip (input/output)
  - tok/s/MW
  - P99 TTFT
  - cost per million tokens (total/input/output) using hyperscaler-volume ownership and rental pricing
  - joules per token from chip power telemetry

  — [InferenceX about](https://inferencex.semianalysis.com/about)

#### Benchmark tools (GitHub metadata 2026-11-24)

| Tool (URL) | Maintainer | Latest version (date) | Stars / last push | What it does | Status |
|---|---|---|---|---|---|
| `vllm bench serve` / `vllm bench sweep` ([docs](https://docs.vllm.ai/en/stable/cli/bench/sweep/); [sweeps guide](https://docs.vllm.ai/en/stable/benchmarking/sweeps/)) | vLLM | ships with vLLM v0.34.0 | n/a | Sweep subcommands: `serve`, `serve_workload`, `startup`, `plot`, `plot_pareto` (tokens/s/user vs tokens/s/GPU). Cartesian product of serve × bench params; `--num-runs` default 3; `--resume`. Docs say `--workload-var max_concurrency` generally gives more reliable results than the default `request_rate` | Active, versioned docs |
| [GuideLLM](https://github.com/vllm-project/guidellm) | vLLM project (Red Hat / ex-Neural Magic) | v0.8.1 (2026-10-08) | ~1.8k / 2026-11-24 | Realistic traffic simulation, SLO and goodput profiles; used in the DLAI vLLM course | Active |
| [AIPerf](https://github.com/ai-dynamo/aiperf) | NVIDIA (ai-dynamo) | v0.14.0 (2026-11-20) | ~900 / 2026-11-20 | Official GenAI-Perf successor. Multiprocess rewrite (ZMQ) to avoid client GIL limits, not built on Perf Analyzer. 15+ endpoint types, GPU telemetry via DCGM/pynvml; `pip install aiperf` or `uv tool install aiperf` | Active; drop-in replacement for supported GenAI-Perf features |
| GenAI-Perf / [perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) | NVIDIA | — | ~160 / 2026-11-17 | Legacy Triton-era tool | Superseded by AIPerf |
| [inference-perf](https://github.com/kubernetes-sigs/inference-perf) | Kubernetes SIG Scalability (founded under WG Serving) | v0.7.0 (2026-09-15) per GitHub metadata | ~280 / 2026-11-24 | Model-server-agnostic. Constant, Poisson and concurrent-user load; multi-stage saturation sweeps. Datasets: ShareGPT, CNN-DM, BillSum, VisionArena, synthetic, shared-prefix, multi-turn; trace replay (Azure, OTel, agentic). Metrics: TTFT/TPOT/ITL/normalized TPOT, goodput. Claims 10k+ QPS; K8s deploy guide; converts `vllm bench` and `aiperf` outputs; Apache-2.0; JOSS paper | Active |
| [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) | llm-d | v1.0.0 (2026-10-16) | ~80 / 2026-11-23 | llm-d's reproducible benchmark workflows; WG Serving benchmarking moved here | Active |
| [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) | llm-d | v1.2.0 (2026-11-10) | ~240 / 2026-11-19 | Lightweight simulator mimicking vLLM, for testing routers and autoscalers without GPUs | Active |
| [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) | NVIDIA | v0.12.0 (2026-09-18) | ~470 / 2026-10-29 | Offline config search for disaggregated Dynamo deployments | Active |
| [Vidur](https://github.com/microsoft/vidur) | Microsoft Research | no release | ~730 / 2026-08-24 | Large-scale LLM inference simulator | Research tool |
| [LLMPerf](https://github.com/ray-project/llmperf) | Ray project | v2.0 (2023-12) | ~1.1k / 2024-12-09 | Load and correctness tests | **Archived 2025-12-17 (read-only)** |
| [HF inference-benchmarker](https://github.com/huggingface/inference-benchmarker) | Hugging Face | v1.1.0 (2025-03-28) | ~180 / 2026-09-23 | Inference server benchmarking | Low activity |
| SGLang `bench_serving` | SGLang | ships with SGLang | — | Engine-bundled benchmark | (not separately verified) |

Sources (in addition to links in the table): [NVIDIA AIPerf blog](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf); [GenAI-Perf vs AIPerf matrix](https://docs.nvidia.com/aiperf/getting-started/gen-ai-perf-vs-ai-perf-cli-feature-comparison-matrix); [inference-perf README](https://github.com/kubernetes-sigs/inference-perf); [LLMPerf repo archive notice](https://github.com/ray-project/llmperf); [CNCF WG Serving wrap-up](https://www.cncf.io/blog/2026/02/26/kubernetes-wg-serving-concludes-following-successful-advancement-of-ai-inference-support/)

#### Public benchmark efforts
- **MLPerf Inference v6.0** (results around 2026-04-01) added GPT-OSS-120B (Offline, Server and Interactive scenarios), DLRMv3, a vision-language model, text-to-video, and a DeepSeek-R1 Interactive scenario with tighter latency. Vendor highlights include NVIDIA's 288 Blackwell Ultra GPUs across four GB300 NVL72 systems (largest submission to date); Red Hat's gpt-oss-120b at 93,070.70 offline and 71,588.13 server tok/s on 8×B200 with vLLM; and AMD's MI355X-only submission with >1M tok/s multi-node. — [MLCommons group post](https://groups.google.com/a/mlcommons.org/g/public/c/TFU4lsU-e0A); [NVIDIA](https://developer.nvidia.com/blog/nvidia-extreme-co-design-delivers-new-mlperf-inference-records/); [Red Hat](https://www.redhat.com/en/blog/red-hat-ai-tops-mlperf-inference-v60-vllm-qwen3-vl-whisper-and-gpt-oss-120b); [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html)
- **MLPerf Inference v6.1** (2026-09-16) drew a record 30 organizations and 486 results. It added an end-to-end RAG test and an Edge Agentic Inference test, and included the largest system ever (512 accelerators). NVIDIA claims its Vera Rubin NVL72 preview delivers up to 3.7× GB300 NVL72 throughput (vendor claim; some post-submission numbers are unverified by MLCommons). — [HPCwire/MLCommons](https://www.hpcwire.com/off-the-wire/mlcommons-sets-participation-record-with-new-mlperf-inference-v6-1-benchmark-results/); [StorageReview](https://www.storagereview.com/news/mlperf-inference-v6-1-5-7x-per-accelerator-gains-a-512-gpu-run-and-vera-rubins-first-peer-reviewed-numbers); [NVIDIA via HPCwire](https://www.hpcwire.com/off-the-wire/nvidia-vera-rubin-nvl72-delivers-leading-performance-in-mlperf-inference-v6-1-debut/); [Lambda](https://lambda.ai/blog/mlperf-inference-v6.1); [AMD](https://www.amd.com/en/blogs/2026/amd-delivers-its-broadest-mlperf-inference-6-1-submission.html)
- The MLPerf reference implementations repo (mlcommons/inference, ~1.7k stars) shows latest release tag v5.1.1 (2025-10-28) but was pushed 2026-11-06. — [mlcommons/inference](https://github.com/mlcommons/inference)
- **SemiAnalysis InferenceMAX → InferenceX.** The benchmark was renamed InferenceX. InferenceXv2 (2026-02-16) was the first to benchmark GB300 NVL72 and B300 across the full Pareto frontier, using close to 1,000 GPUs in a full run, with a free public dashboard. The original repo redirects to SemiAnalysisAI/InferenceX (~2.1k stars, pushed 2026-11-22). — [SemiAnalysis InferenceXv2](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs); [Original InferenceMAX launch post](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference); [repo](https://github.com/SemiAnalysisAI/InferenceX)
- InferenceX methodology:
  - Hardware: NVIDIA (H100, H200, B200, B300, GB200/GB300 NVL72 and others), AMD (MI300X, MI325X, MI355X) and Google TPU.
  - Engines: vLLM, SGLang, TRT-LLM, Dynamo and llm-d configurations.
  - Workloads: fixed-sequence 1k/1k, 1k/8k and 8k/1k at multiple concurrencies, plus "AgentX". AgentX is a long-context multi-turn agentic coding workload built on AIPerf, replaying Weka-format traces with shared prefixes and subagent fan-out, minimum 900 s runs.
  - Runs: originally nightly, now continuous and triggered by configuration changes. Every data point links to a public GitHub Actions run, with recipes in `/benchmarks`.
  - Caveat: the fetched About page carried a banner "This is an unofficial deployment, not affiliated with SemiAnalysis" (possibly template text from the open-source dashboard app) and listed some unfamiliar engines and hardware. Treat the detailed lists cautiously.

  — [InferenceX about](https://inferencex.semianalysis.com/about)
- NVIDIA promotes its InferenceMAX results (Oct 2025 blog), a reminder that vendor commentary on these benchmarks is promotional. — [NVIDIA blog](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results)

#### Capacity planning and methodology resources (free unless noted)
- **NVIDIA, "How to Size GPUs for AI Inference and TCO Without Overspending"** (2026-09-01, Prerana Gambhir and Manasa Manohara). The method:
  1. Map the use case (chatbot/copilot, agents, content generation, translation).
  2. Gather inputs: model, DAUs and concurrency, ISL/OSL, KV-cache hit rate, latency targets, requests per user per day, contract length.
  3. Use a "core" (on-prem/reserved) plus "flex" (spot/on-demand) capacity model.
  4. Right-size GPUs to memory footprint, latency and concurrency.

  It recommends FP8 as the starting quantization; in its example FP8 cut Llama-3.1-8B weight memory from 16.06 GB to 9.08 GB (−43.5%). Intermediate level. — [NVIDIA blog](https://developer.nvidia.com/blog/how-to-size-gpus-for-ai-inference-and-tco-without-overspending)
- **Earlier NVIDIA sizing material**: "Practical Strategies for Optimizing LLM Inference Sizing and Performance" (2024), the GTC24 session S62797 "LLM Inference Sizing: Benchmarking End-to-End Inference Systems" (Dmitry Mironov, Sergio Perez; slides public), and "LLM Inference Benchmarking: How Much Does Your LLM Inference Cost?" — [NVIDIA 2024 blog](https://developer.nvidia.com/blog/practical-strategies-for-optimizing-llm-inference-sizing-and-performance); [S62797 slides](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/S62797-LLM-Inference-Sizing_-Benchmarking-End-to-End-Inference-Systems.pdf); [NVIDIA cost blog](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost)
- **NVIDIA DLI "Sizing LLM Inference Systems"** is a self-paced notebook course with a 6-hour compute allowance (forum thread dated 2026-04-01) and appears in the DLI catalog. Its price was not verified. — [NVIDIA forum](https://forums.developer.nvidia.com/t/nvidia-sizing-llm-inference-systems-request-for-additional-lab-hours/365324); [DLI catalog PDF](https://dam-cdn.nvd.orangelogic.com/AssetLink/f14exk056qm0g3x3550ibl0xi88880r6.pdf)
- **Red Hat Developer GuideLLM tutorials**: "GuideLLM: Evaluate LLM deployments for real-world inference" (2025-06-20), "How to deploy and benchmark vLLM with GuideLLM on Kubernetes" (2025-12-24), and the learning-path resource "Measure what matters: benchmarking and evaluation". — [GuideLLM article](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference); [vLLM+GuideLLM on K8s](https://developers.redhat.com/index%2Ephp/articles/2025/12/24/how-deploy-and-benchmark-vllm-guidellm-kubernetes); [Measure what matters](https://developers.redhat.com/learning/learn:openshift-ai:get-started-vllm/resource/resources:measure-what-matters-benchmarking-and-evaluation)
- **Luca Berton's guide** to using GuideLLM for SLO sign-off of vLLM on Kubernetes/OpenShift is a practitioner resource. — [lucaberton.com](https://lucaberton.com/blog/guidellm-vllm-benchmark-slo-kubernetes/)
- **Modal's "LLM Engineer's Almanac"** (benchmarks and methodology; "three types of LLM workloads" separating prefill from decode) and Modal's high-performance LLM inference guide. — [LLM Almanac workloads](https://modal.com/llm-almanac/workloads); [Modal guide](https://modal.com/docs/guide/high-performance-llm-inference)
- **GKE Inference Quickstart** provides data-driven performance and cost analysis to pick hardware and model before deploying. — [GKE Terraform tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-open-models-terraform)
- **DeepLearning.AI × Red Hat "Fast & Efficient LLM Inference with vLLM"** includes a lab running GuideLLM against a vLLM deployment, plus lm-eval accuracy checks (see Q4). — [DLAI course page](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/)

### Inferences
- A sound self-learner methodology from these sources:
  1. Fix the workload shape (ISL/OSL distribution, prefix sharing, multi-turn/agentic traces).
  2. Sweep concurrency rather than request rate.
  3. Plot the tok/s/user vs tok/s/GPU Pareto curve.
  4. Find the knee and the goodput-maximizing concurrency under explicit TTFT/TPOT SLOs.
  5. Convert to cost per million tokens using the GPU-hour price.
  6. Repeat for each quantization level and parallelism configuration.

  vLLM's sweep tooling, GuideLLM's goodput profile, inference-perf and AIPerf each implement parts of this.
- Agentic, multi-turn, prefix-heavy workloads (InferenceX AgentX, inference-perf trace replay, llm-d agent trace replay, MLPerf v6.1 RAG/agentic tests) are becoming the 2026 benchmarking frontier. Fixed 1k/1k-style tests remain useful but no longer represent production traffic.
- Cross-tool numbers are not directly comparable (TTFT inclusion, reasoning-token handling), so a learner should pin one tool per study and report its version.

### Gaps
- Official MLCommons v6.0/v6.1 results tables were not fetched; the per-system numbers above are vendor-reported.
- I found no NVIDIA-documented definition of "goodput" in the fetched sources, and no community-standard goodput definition across tools.
- Not verified: SGLang `bench_serving` specifics, Artificial Analysis (hosted-API benchmarks), and the full InferenceX license statement. The About page says open source; the earlier newsletter says Apache 2.0.

## 4. Which structured courses, workshops and books (free or mostly free) teach LLM deployment and MLOps hands-on, and how current are they?

### Takeaway
The most current, on-target structured resources in late 2026 are:
- DeepLearning.AI's **"Fast & Efficient LLM Inference with vLLM"** (with Red Hat; released Jun 3, 2026, updated Nov 11, 2026).
- The older but refreshed **"Efficiently Serving LLMs"** (Predibase) and the Hugging Face **quantization** pair.
- Anyscale's free **"LLM Serving Foundations"** for Ray Serve LLM.
- KodeKloud's beginner **"AI Infrastructure Explained (GPUs, vLLM, and LLM-D)"** on Coursera (Oct 2026, paid via Coursera Plus).
- **NVIDIA DLI** sizing/NIM/Triton courses.
- Two strong 2026 books: Philip Kiely's **"Inference Engineering"** (Baseten, free online, finished Jan 2026) and Huß & Zonca's **"Generative AI on Kubernetes"** (O'Reilly, Feb 2026).

Classic MLOps courses (DataTalksClub MLOps Zoomcamp, Made With ML, Full Stack Deep Learning) remain useful foundations but do not teach LLM serving, and FSDL is explicitly dated. LLM Zoomcamp 2026 teaches LLM-application ops (RAG, evaluation, monitoring), not self-hosted serving.

### Cited Findings

#### Short courses and MOOCs (English unless noted)

- **Fast & Efficient LLM Inference with vLLM** — https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/
  - Provider/author: DeepLearning.AI × Red Hat; instructor Cedric Clyburn (Senior Developer Advocate, Red Hat).
  - Dates: released 2026-06-03; page shows "Updated Nov 11, 2026".
  - Format: 1h38m video course, 9 video lessons, 3 code labs, graded quiz.
  - Cost: free to enroll; graded assignment and certificate require PRO.
  - Level and prerequisites: intermediate; Python plus basic LLM concepts (tokens, inference, GPU usage).
  - Topics: weight and activation quantization with LLM Compressor; vLLM OpenAI-compatible serving; PagedAttention, continuous batching and prefix caching; KV cache and GPU memory hierarchy; perplexity; GuideLLM benchmarking and lm-eval.
  - Why it stands out: the only 2026 short course covering the full quantize → serve → benchmark loop on vLLM.
  - Sources: [DLAI page](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/); [vLLM blog](https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course); [Red Hat blog](https://developers.redhat.com/blog/2026/06/03/learn-optimize-deploy-and-benchmark-llms-vllm-new-free-course)
- **Efficiently Serving LLMs** — https://www.deeplearning.ai/short-courses/efficiently-serving-llms/
  - Provider/author: DeepLearning.AI × Predibase; Travis Addair (Predibase co-founder and CTO).
  - Dates: page "last updated" 2026-08-20; the course launched in 2024.
  - Format: 2h09m, 8 video lessons, 6 code examples, 10-minute graded quiz.
  - Cost: free "for a limited time during the platform beta"; certificate requires PRO.
  - Level and prerequisites: intermediate Python.
  - Topics: autoregressive generation, KV caching, batching and continuous batching, quantization, LoRA and multi-LoRA, LoRAX.
  - Why it stands out: builds engine-agnostic mental models from first principles.
  - Caveat: its LoRAX server (predibase/lorax, ~3.8k stars) last released lorax-0.4.0 in January 2025, so the tooling is dated.
  - Sources: [DLAI page](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/); [LoRAX repo](https://github.com/predibase/lorax)
- **Quantization in Depth** (and its prerequisite **Quantization Fundamentals with Hugging Face**) — https://www.deeplearning.ai/short-courses/quantization-in-depth/
  - Provider/author: DeepLearning.AI × Hugging Face; Marc Sun and Younes Belkada.
  - Dates: last updated 2025-10-07.
  - Format: 2h20m, 18 video lessons, 13 code examples.
  - Cost: free during beta; graded quiz/certificate requires PRO.
  - Level: intermediate.
  - Topics: linear quantization (symmetric/asymmetric); per-tensor, per-channel and per-group granularity; building a PyTorch quantizer; weight packing (e.g., four 2-bit weights per int8); loading quantized weights from the Hub.
  - Why it stands out: explains the "why" behind quantization, which is the main lever for serving cost.
  - Source: [DLAI page](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
- **AI Infrastructure Explained (GPUs, vLLM, and LLM-D)** — https://www.coursera.org/learn/ai-infrastructure-gpus-vllm-llmd
  - Provider/author: KodeKloud on Coursera; instructor Mumshad Mannambeth.
  - Dates: last updated October 2026.
  - Format: about 6 hours across 4 modules (about 51 minutes of video), 8 graded items, optional hands-on vLLM lab; self-paced.
  - Cost: included in Coursera Plus; no clear free audit, so treat as **paid**.
  - Level and prerequisites: beginner; no ML or coding background required.
  - Modules: from GPU to model server; token-by-token generation and the KV cache; why serving gets hard (batching, TP, why classic load balancing fails); llm-d (cache-aware routing, P/D separation, Kubernetes).
  - Why it stands out: the gentlest structured on-ramp to llm-d concepts.
  - Source: [Coursera page](https://www.coursera.org/learn/ai-infrastructure-gpus-vllm-llmd)
- **Anyscale Academy, LLM Serving Foundations** — https://courses.anyscale.com/courses/llm-serving-foundations
  - Provider: Anyscale.
  - Dates: not stated.
  - Format: free; 24 lessons, about 2 hours of video.
  - Topics: LLM-serving concepts, Ray Serve LLM setup, local deployment, Anyscale Services, tool calling.
  - Companion: "Model Serving Foundations" (7 lessons).
  - Level: beginner–intermediate (assessed); prerequisite Python.
  - Sources: [course](https://courses.anyscale.com/courses/llm-serving-foundations); [collection](https://courses.anyscale.com/collections/model-serving)
- **NVIDIA Deep Learning Institute (DLI) self-paced catalog** (inference-relevant items)
  - "Introduction to NVIDIA NIM Microservices": free, about 2 hours.
  - "Sizing LLM Inference Systems": notebook labs with a 6-hour compute allowance.
  - "Introduction to Deploying RAG Pipelines for Production at Scale".
  - "Deploying a Model for Inference at Production Scale": Triton, 4 hours, $30.
  - Instructor-led workshops include "Deploying and Optimizing AI Inference at Scale".
  - No Dynamo-specific self-paced DLI course was found.
  - Free DLI courses are described as beginner-level and completable in a day.
  - Sources: [NVIDIA self-paced page](https://www.nvidia.com/en-us/training/self-paced-courses/); [AI Learning Essentials](https://www.nvidia.com/ai-essentials); [DLI online listing](https://www.nvidia.com/en-eu/training/online.md); [DLI catalog PDF](https://dam-cdn.nvd.orangelogic.com/AssetLink/f14exk056qm0g3x3550ibl0xi88880r6.pdf); [Sizing forum](https://forums.developer.nvidia.com/t/nvidia-sizing-llm-inference-systems-request-for-additional-lab-hours/365324); [ILT workshops](https://www.nvidia.com/en-in/training/instructor-led-workshops/)
- **Hugging Face LLM Course, Chapter 2.8 "Optimized Inference Deployment"** — https://huggingface.co/learn/llm-course/en/chapter2/8
  - Provider: Hugging Face.
  - Format: free text plus code; repo huggingface/course (~4.4k stars, pushed 2026-11-17).
  - Topics: TGI vs vLLM vs llama.cpp.
  - Caveat: **partly outdated** because TGI is in maintenance mode; one TGI passage reads like a llama.cpp description and may be a page error.
  - Sources: [HF chapter](https://huggingface.co/learn/llm-course/en/chapter2/8); [course repo](https://github.com/huggingface/course); [TGI maintenance notice](https://huggingface.co/docs/text-generation-inference/index)
- **DataTalksClub LLM Zoomcamp (2026)** — https://github.com/DataTalksClub/llm-zoomcamp
  - Provider/author: DataTalks.Club; Alexey Grigorev, Will Russell (Kestra), Timur Kamaliev.
  - Format: free; pre-recorded videos, homework, capstone; certificate only in the live cohort.
  - Modules: Agentic RAG, vector search (minsearch, sqlitesearch, PGVector), Kestra orchestration, a dlt workshop on LLM traces, evaluation, monitoring with dashboards, best practices (hybrid search, reranking), project.
  - Prerequisites: Python, CLI, basic Docker; no GPU (about $1–5 of OpenAI credits).
  - Repo: ~7.7k stars, pushed 2026-11-16.
  - Caveat: **does not cover self-hosted serving or deployment**.
  - Source: [repo README](https://github.com/DataTalksClub/llm-zoomcamp)
- **DataTalksClub MLOps Zoomcamp** — https://github.com/DataTalksClub/mlops-zoomcamp
  - Provider/author: DataTalks.Club; Cristian Martinez, Alexey Grigorev, Emeli Dral.
  - Format: free, self-paced; **no live cohort planned in 2026**.
  - Topics: MLflow tracking and registry; orchestration; deployment (Flask, AWS Kinesis/Lambda, batch); monitoring (Prometheus, Evidently, Grafana); testing, CI/CD (GitHub Actions) and Terraform.
  - Prerequisites: Python, Docker, ML basics, about a year of programming.
  - Repo: ~15.6k stars, pushed 2026-11-16; banner art from 2025.
  - Caveat: classic MLOps, not LLM serving.
  - Source: [repo README](https://github.com/DataTalksClub/mlops-zoomcamp)
- **Made With ML** — https://madewithml.com/
  - Provider/author: Goku Mohandas; now part of Anyscale.
  - Format: free lessons plus repo; GitHub v1.1.0 (2026-03-04), ~50.1k stars.
  - Topics: design → data → model (training, tracking, tuning, evaluation, serving) → testing → reproducibility → production (jobs/services, CI/CD, monitoring), with an LLM fine-tuning project.
  - Caveat: site footer © 2025; general MLOps rather than LLM-serving infrastructure.
  - Sources: [site](https://madewithml.com/); [repo](https://github.com/GokuMohandas/Made-With-ML)
- **Full Stack Deep Learning, LLM Bootcamp (Spring 2023) and 2022 course labs** — https://www.fullstackdeeplearning.com/llm-bootcamp
  - Format: free.
  - Caveat: **OUTDATED/archival**. The site itself warns that tools and model capabilities have changed since the lectures were recorded, and the 2022 labs repo was last pushed 2024-01-17.
  - Sources: [LLM Bootcamp page](https://www.fullstackdeeplearning.com/llm-bootcamp); [labs repo](https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs)
- **mlabonne/llm-course** — https://github.com/mlabonne/llm-course
  - Format: free roadmap plus Colab notebooks; ~85.2k stars; last pushed 2026-02-05.
  - Not verified: depth of its deployment content (see Gaps).
  - Source: [repo](https://github.com/mlabonne/llm-course)
- **Paid or other platforms, not verified in depth**
  - Maven "vLLM & SGLang Engineering: End-to-End LLM Serving" (Rajat Dandekar; paid cohort).
  - O'Reilly live event "Inference Optimization for LLMs with vLLM" (O'Reilly subscription).
  - Coursera "Deploying Deep Learning: Quantization, Serving and Edge AI" (Board Infinity).
  - Sources: [Maven](https://maven.com/rajat-dandekar/vllm-sglang-engineering); [O'Reilly live event](https://www.oreilly.com/live-events/inference-optimization-for-llms-with-vllm/0642572193737/); [Coursera Board Infinity](https://www.coursera.org/learn/deploying-deep-learning-quantization-serving-and-edge-ai)

#### Vendor learning paths and cloud labs (free docs; cloud GPU costs apply)
- **Red Hat Developer "Get started with vLLM" learning path** (OpenShift AI), with a "Measure what matters: benchmarking and evaluation" resource, plus the llm-d articles listed in Q2. The learning-path page returned HTTP 500 at fetch time, so its full contents were not verified. — [Red Hat resource](https://developers.redhat.com/learning/learn:openshift-ai:get-started-vllm/resource/resources:measure-what-matters-benchmarking-and-evaluation)
- **Google Cloud GKE tutorials** (Gemma 4/Llama 4/Qwen3 with vLLM, Terraform reference architecture, Inference Gateway, autoscaling best practices). Intermediate; needs a GCP project and GPU quota. — [GKE Gemma vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm); [GKE autoscaling](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)
- **AWS AI on EKS blueprints** (vLLM, Ray Serve, Triton, AIBrix, Dynamo) and the EKS vLLM quickstart. Some Dynamo pages are flagged outdated. — [AI on EKS inference](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference); [EKS vLLM quickstart](https://docs.aws.amazon.com/eks/latest/userguide/ml-realtime-inference-llm-inference-vllm.html)
- **Azure AKS KAITO docs.** Microsoft pages cite KAITO 0.4.x while upstream is at v0.13.0, so verify versions. — [AKS KAITO](https://learn.microsoft.com/en-us/azure/aks/ai-toolchain-operator)
- **Project quickstarts**: llm-d quickstart, Dynamo Kubernetes quickstart, production-stack tutorials, KServe LLMInferenceService docs, Ray Serve LLM quickstart (see Q2 for URLs). — [llm-d README](https://cdn.jsdelivr.net/gh/llm-d/llm-d@main/README.md); [Dynamo quickstart](https://docs.nvidia.com/dynamo/kubernetes-deployment/start-here/kubernetes-quickstart)

#### Books and long-form guides

- **Inference Engineering** (Philip Kiely, Head of AI Education at Baseten) — https://www.baseten.co/inference-engineering/book/
  - Publisher and date: Baseten Books, 2026; "finished in January 2026".
  - Format: full text free online, including Markdown. The e-book is reportedly free to download; print copies were sold out as of Pragmatic Engineer's article.
  - Chapters: 0 Inference, 1 Prerequisites, 2 Models, 3 Hardware, 4 Software (CUDA → PyTorch → vLLM/SGLang/TRT-LLM/Dynamo), 5 Techniques (quantization, speculative decoding, caching, parallelism, disaggregation), 6 Modalities, 7 Production; Appendix A glossary, Appendix B reading list.
  - Level: beginner–intermediate. A reviewer called it "a pretty good book to start with" that is "not an academic text book".
  - Caveat: vendor-authored.
  - Sources: [book TOC](https://www.baseten.co/inference-engineering/book/); [Chapter 0](https://www.baseten.co/inference-engineering/book/00-inference/); [Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/what-is-inference-engineering); [book landing](https://www.baseten.co/inference-engineering/)
- **Generative AI on Kubernetes: Operationalizing Large Language Models** (Roland Huß and Daniele Zonca, Red Hat) — https://www.oreilly.com/library/view/generative-ai-on/9781098171919/
  - Publisher and date: O'Reilly, February 2026, 406 pp.
  - Editions: English, plus German, Italian and Chinese.
  - Topics: training, fine-tuning, deploying and scaling GenAI on Kubernetes; model data; Kubernetes and GPUs; observability; customization; agentic apps in production.
  - Audience: MLOps engineers, developers and K8s admins (intermediate).
  - Cost: paywalled. A Red Hat-sponsored download listing appeared in Oct 2026, which may be a free sponsored copy.
  - Sources: [O'Reilly](https://www.oreilly.com/library/view/generative-ai-on/9781098171919/); [Virtualization Review listing](https://virtualizationreview.com/whitepapers/2026/10/red-hat-generative-ai-on-kubernetes-operationalizing-large-language-models.aspx); [O'Reilly Radar](https://oreilly.com/radar/kubernetes-in-the-age-of-ai/)
- **AI Engineering** (Chip Huyen) — https://www.oreilly.com/library/view/ai-engineering/9781098166298/
  - Publisher and date: O'Reilly; Open Library lists a 2024 edition.
  - Relevant content: Chapter 9 "Inference Optimization" covers performance metrics, AI accelerators, model optimization and inference-service optimization.
  - Companion: free resources repo chiphuyen/aie-book (~18.4k stars, pushed 2026-07-03).
  - Cost: paywalled.
  - Level: intermediate.
  - Related: Huyen's P99 CONF keynote warns that many inference optimizations change model quality, so evaluate cost, latency and quality together.
  - Sources: [Ch. 9](https://oreilly.com/library/view/ai-engineering/9781098166298/ch09.html); [Open Library](https://openlibrary.org/isbn/1098166302); [aie-book repo](https://github.com/chiphuyen/aie-book); [The New Stack on P99 talk](https://thenewstack.io/pg-99-conf-2026-inference-costs/)
- **Designing Machine Learning Systems** (Chip Huyen; O'Reilly, 2022)
  - Pre-LLM but foundational for MLOps (data, deployment, monitoring).
  - Companion summaries repo chiphuyen/dmls-book (~5.5k stars, pushed 2026-06-09).
  - Source: [dmls-book repo](https://github.com/chiphuyen/dmls-book)
- **AI Systems Performance Engineering** (Chris Fregly) — https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/
  - Publisher and date: O'Reilly, November 2025, about 1,062 pp.
  - Content: training and inference performance across GPUs, CUDA, PyTorch and multinode inference. Chapter 19 covers dynamic and adaptive inference (adaptive parallelism, precision, cache management, RL-based tuning).
  - Extras: a 175+ (book) or 200+ (repo) item checklist; companion repo cfregly/ai-performance-engineering.
  - Level: advanced. Cost: paywalled.
  - Sources: [O'Reilly](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/); [Ch. 19](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/ch19.html); [repo](https://github.com/cfregly/ai-performance-engineering)
- **LLMs in Production** (Christopher Brousseau and Matt Sharp) — https://www.manning.com/books/llms-in-production
  - Publisher and date: Manning; Manning lists December 2024, retailers February 2025; about 456 pp.
  - Topics: an LLMOps plan from design to delivery, prompt engineering, retraining and load testing, cost handling, security.
  - Audience: data scientists and ML engineers who know Python and cloud basics.
  - Cost: paywalled.
  - Sources: [Manning](https://www.manning.com/books/llms-in-production); [O'Reilly listing](https://www.oreilly.com/library/view/llms-in-production/9781633437203/)
- **LLM Engineer's Handbook** (Paul Iusztin and Maxime Labonne)
  - Publisher and date: Packt, October 2024, 522 pp, 1st edition; no 2nd edition found.
  - Content: an end-to-end "LLM Twin" project with LLMOps.
  - Code repo: ~5.4k stars, pushed 2026-04-22.
  - Cost: paywalled.
  - Sources: [Packt](https://www.packtpub.com/product/llm-engineers-handbook/9781836200079); [repo](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- **Kubernetes for Generative AI Solutions** (Ashok Srirama and Sukirti Gupta; Packt), recommended by O'Reilly Radar alongside Huß/Zonca. Details not verified. — [O'Reilly Radar](https://oreilly.com/radar/kubernetes-in-the-age-of-ai/)
- **Machine Learning Engineering Open Book** (Stas Bekman)
  - Format: free GitHub book, ~19.7k stars, pushed 2026-11-24, actively maintained.
  - Topics: ML infrastructure (hardware, networking, storage, debugging).
  - Not verified: depth of its inference chapter.
  - Source: [repo](https://github.com/stas00/ml-engineering)
- **LLM Inference Handbook** (originally BentoML; repo now modular/llm-inference-handbook)
  - Repo: ~490 stars, pushed 2026-11-24.
  - The bentoml.com site could not be reached from this session, so content was not verified.
  - Source: [repo](https://github.com/modular/llm-inference-handbook)
- **Modal GPU Glossary** (Charles Frye et al.) — https://modal.com/gpu-glossary/readme
  - Format: free hyperlinked reference; text CC BY 4.0, code MIT; about 2–3 hours to read.
  - Level: intermediate; for readers with some deep-learning exposure and no CUDA background.
  - Why it stands out: hardware foundations for serving.
  - Sources: [glossary](https://modal.com/gpu-glossary/readme); [contributors](https://modal.com/gpu-glossary/contributors); [Beri summary](https://www.beri.net/learning/modal-gpu-glossary); [license listing](https://awesome.ecosyste.ms/projects/github.com%2Fmodal-labs%2Fgpu-glossary)

#### CNCF and Linux Foundation resources
- **CNCF Cloud Native AI whitepaper** (March 2024; WG under TAG Runtime). It surveys AI/ML techniques, what cloud native offers and open challenges. It is **dated** (pre-GIE/llm-d), and no 2026 successor was found. The WG charter lists further whitepapers and project mapping; a TOC slide proposes restructuring WGs into initiatives, including a "Cloud Native AI Scheduling Challenges" whitepaper. — [CNAI whitepaper](https://tag-runtime.cncf.io/wgs/cnaiwg/whitepapers/cloudnativeai/); [CNCF 2024 announcement](https://www.cncf.io/blog/2024/03/19/announcing-the-ai-working-groups-new-cloud-native-artificial-intelligence-whitepaper/); [WG charter](https://tag-runtime.cncf.io/wgs/cnaiwg/charter)
- **Kubernetes AI Conformance**: requirements live in the versioned files of cncf/k8s-ai-conformance, a good map of "what a K8s AI platform must provide". — [repo](https://github.com/cncf/k8s-ai-conformance)
- **CNCF/LF Education and Udemy** announced a partnership in June 2026 for a unified cloud-native training and certification path. No specific Linux Foundation LLM-inference course was found. — [CNCF announcement](https://www.cncf.io/announcements/2026/06/17/cncf-and-linux-foundation-education-partner-with-udemy-to-provide-a-unified-cloud-native-training-certification-opportunity/)

#### Outdated, archived or superseded items (flag list)
- **LLMPerf** (ray-project/llmperf): archived read-only 2025-12-17. — [repo](https://github.com/ray-project/llmperf)
- **GenAI-Perf**: superseded by AIPerf. — [AIPerf migration](https://docs.nvidia.com/aiperf/getting-started/migrating-from-gen-ai-perf)
- **TGI**: in maintenance mode, so TGI-centric tutorials (including HF course ch. 2.8) are dated. — [HF docs](https://huggingface.co/docs/text-generation-inference/index)
- **Full Stack Deep Learning LLM Bootcamp (2023) and 2022 labs**: dated by the site's own warning; repo last pushed Jan 2024. — [FSDL](https://www.fullstackdeeplearning.com/llm-bootcamp); [labs](https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs)
- **CNCF Cloud Native AI whitepaper (2024)**: pre-dates the 2025–26 inference stack. — [CNAI whitepaper](https://tag-runtime.cncf.io/wgs/cnaiwg/whitepapers/cloudnativeai/)
- **MLOps Zoomcamp**: no 2026 live cohort; classic MLOps only. — [repo](https://github.com/DataTalksClub/mlops-zoomcamp)
- **"Efficiently Serving LLMs"**: concepts still valid, but its LoRAX tooling's last release was Jan 2025. — [LoRAX](https://github.com/predibase/lorax)
- **Stale or low-activity repos**:
  - OpenLLM: last release v0.6.30, Apr 2025.
  - llmaz: last release Jun 2025.
  - HF inference-benchmarker: last release Mar 2025.
  - ray-educational-materials: last push Feb 2024.
  - NVIDIA GenerativeAIExamples: last release v0.8.0, Aug 2024, though recently pushed.
  - Sources: [OpenLLM](https://github.com/bentoml/OpenLLM); [llmaz](https://github.com/InftyAI/llmaz); [inference-benchmarker](https://github.com/huggingface/inference-benchmarker); [ray-educational-materials](https://github.com/ray-project/ray-educational-materials); [GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples)
- **Renamed (old names still appear in tutorials)**:
  - InferenceMAX → InferenceX.
  - Envoy AI Gateway → Agent Router.
  - Triton → Dynamo-Triton.
  - llm-d-inference-scheduler → llm-d-router.
  - GIE InferenceModel → InferenceObjective.
  - Sources: [InferenceMAX repo](https://github.com/InferenceMAX/InferenceMAX); [AAIF](https://aaif.io/blog/agent-router-joins-aaif); [Dynamo-Triton](https://developer.nvidia.com/dynamo-triton); [GKE GIE migration](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/migrate-gke-inference-gateway)
- **AI on EKS Dynamo blueprint page**: flagged "currently outdated". — [page](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/framework-guides/GPUs/nvidia-dynamo)

### Inferences
- Suggested sequence for a self-learner aiming at vLLM/SGLang deployment, using only the verified resources above:
  1. **Foundations.** Read the Modal GPU Glossary and Inference Engineering ch. 0–3, then AI Engineering ch. 9.
  2. **Concepts to hands-on.** Take DLAI "Efficiently Serving LLMs", then DLAI "Fast & Efficient LLM Inference with vLLM", then Quantization Fundamentals and Quantization in Depth.
  3. **Benchmarking.** Use `vllm bench sweep` and GuideLLM on a single GPU, then the Red Hat GuideLLM-on-K8s article.
  4. **Kubernetes serving.** Take the KodeKloud course (concepts), deploy production-stack on minikube, then a GKE/EKS vLLM tutorial, then KServe LLMInferenceService or the llm-d Optimized Baseline.
  5. **Distributed serving.** Try llm-d well-lit paths (P/D, wide EP, tiered prefix cache) or Dynamo recipes. Add autoscaling via the GKE guide and KEDA, and benchmarking via inference-perf or AIPerf.
  6. **MLOps breadth.** Made With ML, MLOps Zoomcamp, DMLS; for LLM-app ops, LLM Zoomcamp and LLMs in Production; Generative AI on Kubernetes as the capstone reference.
- Most structured courses are short (1.5–6 hours). Depth comes from project docs, blogs and the 2025–26 books, so a learner should budget for doc-driven labs rather than expect a single comprehensive LLM-serving MOOC.

### Gaps
- No free, comprehensive, multi-week MOOC dedicated to Kubernetes-scale LLM serving was found. The closest are KodeKloud (paid) and vendor learning paths.
- Not verified: Red Hat learning-path contents (HTTP 500), DLI course prices other than the $30 Triton course, the deployment sections of mlabonne/llm-course and Stas Bekman's book, "Kubernetes for Generative AI Solutions" details, and whether the Red Hat-sponsored "Generative AI on Kubernetes" download is free.
- Could not reach bentoml.com, so the LLM Inference Handbook's current state is unverified beyond repo activity.
- Not checked: Udemy, edX or university-affiliated Coursera specializations (e.g., Duke LLMOps), the latter partly out of scope.

## 5. What are the key cost and hardware-selection considerations for serving (GPU types, cloud vs on-prem, quantization for cost, local deployment with Ollama/llama.cpp), and which resources explain them?

### Takeaway
Cost per token is set by:
- **Right-sizing the GPU** to model memory (weights plus KV cache), concurrency and latency targets.
- **Quantization** (FP8 is the recommended first step; 4-bit for local).
- **Prefix/KV-cache reuse.**
- **The capacity model**: reserved/on-prem "core" for steady load plus cloud "flex" for bursts.
- **Autoscaling on queue/batch signals** rather than GPU utilization.

Public cost data now comes from InferenceX (cost per million tokens, J/token, a TCO calculator), MLPerf (per-system throughput) and SemiAnalysis ClusterMAX (GPU-cloud quality ratings, mostly paywalled). For local deployment the consensus is Ollama for getting started, llama.cpp for control/edge/CPU and single-user speed, and vLLM for multi-user GPU serving.

### Cited Findings

#### Sizing and TCO
- NVIDIA's 2026 guidance:
  - Oversized GPUs lower utilization and raise cost per token, while undersized GPUs constrain throughput and latency.
  - Concurrency strains memory and latency more than raw DAUs.
  - A higher KV-cache hit rate cuts prefill work, TTFT and cost.
  - Stable traffic may justify long-term contracts or on-prem; volatile or experimental workloads fit flexible cloud; colocation may be needed for expansion or data residency.
  - Levers: quantization (FP16 → FP8/INT8 cuts memory about 25–50% without retraining; FP8 is the recommended start), pruning and distillation via NVIDIA Model Optimizer and NeMo.

  — [NVIDIA sizing/TCO blog, 2026-09-01](https://developer.nvidia.com/blog/how-to-size-gpus-for-ai-inference-and-tco-without-overspending)
- InferenceX publishes cost per million tokens computed with hyperscaler-volume ownership and rental pricing, plus tok/s/MW and J/token from chip power telemetry, and links a TCO calculator whose methodology is not described. Hardware spans NVIDIA H100/H200/B200/B300/GB200/GB300 NVL72, AMD MI300X/MI325X/MI355X and Google TPU. — [InferenceX about](https://inferencex.semianalysis.com/about)
- Baseten's book preface asserts that open models are often at least 80% cheaper at scale (vendor claim). — [Inference Engineering](https://www.baseten.co/inference-engineering/book/)
- AIBrix published "Should You Self-Host Batch Inference? An Honest Cost Breakdown" (2026-06-22). — [AIBrix posts](https://aibrix.github.io/posts/)
- Chip Huyen warns that some providers' advertised cost and latency gains come from techniques that degrade model quality, so evaluate quality alongside cost and latency. — [The New Stack](https://thenewstack.io/pg-99-conf-2026-inference-costs/)

#### Cloud and GPU-provider selection
- SemiAnalysis ClusterMAX rates GPU clouds on about ten criteria (performance, security, reliability, ease of use, price, availability). A SemiAnalysis post describes ClusterMAX 3.0 (September 2026): only 19 neoclouds achieve a "Medallion" rating, and a new "Participation Ribbon" tier was added. The official ClusterMAX site still calls 3.0 "Coming Soon" and lists 2.1 (Apr 2026) or 2.0 as current. The full report is paywalled. — [SemiAnalysis ClusterMAX 3.0](https://newsletter.semianalysis.com/p/clustermax-30-the-industry-standard); [ClusterMAX site](https://clustermax.semianalysis.com/); [ClusterMAX 2.0](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard); [original rating-system post](https://newsletter.semianalysis.com/p/the-gpu-cloud-clustermax-rating-system-how-to-rent-gpus); [Latent Space writeup](https://latent.space/reading/reading-youtube-9b19b40623e62467/)
- GKE tutorials list current serving accelerators (B200, H100, RTX Pro 6000, L4 for Gemma 4; H100/H200 for Llama 4). GKE Inference Quickstart provides per-model performance and cost analysis. — [GKE Gemma vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm); [GKE Llama vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llama-gpus-vllm); [GKE Terraform/Inference Quickstart](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-open-models-terraform)
- KAITO's GPU auto-provisioning supports only Ampere-or-newer NVIDIA GPUs, and AMD VM sizes are unsupported in workspaces. — [Thoughtworks KAITO](https://www.thoughtworks.com/radar/platforms/kaito); [AKS docs](https://learn.microsoft.com/en-us/azure/aks/ai-toolchain-operator)
- SkyPilot can mix candidate GPU types (L4, A10g, A100, etc.) and spot instances for serving replicas. — [vLLM SkyPilot docs](https://docs.vllm.ai/en/stable/deployment/frameworks/skypilot/)
- MLPerf v6.1 included first results for AMD MI350P, Intel Xeon 6 (five SKUs, CPU inference) and NVIDIA Vera Rubin NVL72 (preview). These are useful for tracking hardware generations, but the figures are vendor-highlighted. — [AMD](https://www.amd.com/en/blogs/2026/amd-delivers-its-broadest-mlperf-inference-6-1-submission.html); [Intel](https://www.intel.com/content/www/us/en/newsroom/news/data-center/intel-software-optimizations-boost-ai-inference-in-mlperf-v6-1.html); [HPCwire](https://www.hpcwire.com/off-the-wire/nvidia-vera-rubin-nvl72-delivers-leading-performance-in-mlperf-inference-v6-1-debut/)

#### Autoscaling and utilization economics
- In GKE's benchmarks, autoscaling on GPU utilization overprovisions and is cost-inefficient. Queue size is recommended for throughput/cost and batch size for latency. — [GKE autoscaling best practices](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling)
- Scale-to-zero for LLM services is now native in KServe 0.21 via KEDA. AIBrix v0.8 ModelClaim can put idle models to sleep and pack long-tail models onto shared GPUs. llm-d publishes time-slicing for RL post-training ("40% more experiments on the same GPUs"). — [KServe blog](https://kserve.github.io/website/blog); [AIBrix v0.8.0](https://aibrix.github.io/posts/2026-10-12-v0.8.0-release/); [llm-d blog](https://llm-d.ai/blog)

#### Quantization for cost
- LLM Compressor (vLLM project, 0.15.0, 2026-10-29, ~4.0k stars) is the compression tool used in the DeepLearning.AI/Red Hat course. That course teaches weight/activation quantization, then perplexity and lm-eval to measure the accuracy trade-off. — [LLM Compressor](https://github.com/vllm-project/llm-compressor); [DLAI course](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/)
- Hugging Face's quantization short courses explain the underlying techniques: linear/symmetric/asymmetric, per-channel/per-group, and packing. — [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

#### Local deployment (Ollama, llama.cpp, vLLM)
- Red Hat published "llama.cpp vs vLLM: Choosing the right local LLM inference engine" (2026-06-15). — [Red Hat Developer](https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine)
- Community 2026 guides agree:
  - Ollama is best for solo development and prototyping. It wraps llama.cpp on x86/non-Apple hardware and MLX on Apple Silicon (as of Ollama 0.19, March 2026), adding roughly 10–30% overhead versus raw llama.cpp in one test.
  - llama.cpp suits embedded and edge work, custom quantization, multi-GPU tensor-split control and single-user speed.
  - vLLM suits concurrent multi-user GPU serving; its AMD support is described as incomplete.
  - Ollama and llama.cpp do not scale horizontally.

  These are secondary blog sources and their benchmark figures vary. — [dev.to comparison](https://dev.to/thurmon_demich/ollama-vs-llamacpp-vs-vllm-which-should-you-use-in-2026-10gp); [Codersera 2026](https://codersera.com/blog/ollama-vs-lm-studio-vs-vllm-vs-llama-cpp-vs-mlx-2026/); [SitePoint](https://www.sitepoint.com/local-llm-deployment-ollama-vs-vllm-vs-lm-studio-compared/)
- llama.cpp (~136k stars, latest release tag v1.0.0 on 2026-11-13) and Ollama (~188k stars, v0.40.18 on 2026-11-24) are both very active. — [llama.cpp](https://github.com/ggml-org/llama.cpp); [Ollama](https://github.com/ollama/ollama)

#### Explanatory resources on hardware and cost
- **Modal GPU Glossary**: free; hardware concepts. — [GPU Glossary](https://modal.com/gpu-glossary/readme)
- **Inference Engineering Ch. 3 "Hardware" and Ch. 7 "Production"**: free; chapter scope is inferred from titles only. — [book TOC](https://www.baseten.co/inference-engineering/book/)
- **AI Engineering Ch. 9**: AI accelerators, inference metrics. — [Ch. 9](https://oreilly.com/library/view/ai-engineering/9781098166298/ch09.html)
- **AI Systems Performance Engineering**: advanced; GPU, CUDA and cluster co-optimization. — [O'Reilly](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/)
- **NVIDIA sizing blogs and the DLI sizing course.** — [NVIDIA 2026 blog](https://developer.nvidia.com/blog/how-to-size-gpus-for-ai-inference-and-tco-without-overspending); [DLI forum](https://forums.developer.nvidia.com/t/nvidia-sizing-llm-inference-systems-request-for-additional-lab-hours/365324)

### Inferences
- For a self-learner on a budget:
  1. Start locally (Ollama, then llama.cpp with GGUF 4-bit) to learn memory and quantization trade-offs.
  2. Move to a single rented GPU (L4/A10-class, then H100-class) running vLLM/SGLang with FP8 and `vllm bench sweep`/GuideLLM to compute cost per million tokens.
  3. Only then try multi-GPU or multi-node, where disaggregation and KV-cache tiers pay off.
- Cost-per-token comparisons are only meaningful at a stated SLO (TTFT/TPOT), quality level (quantization accuracy checks) and workload shape (prefix-hit rate). InferenceX's Pareto/cost views and NVIDIA's sizing inputs converge on this.

### Gaps
- No current GPU-hour price sheets or neutral cloud price comparisons were collected, and ClusterMAX ratings are paywalled with conflicting version status.
- I found no independent, methodology-documented open TCO calculator. The InferenceX calculator's methodology is undocumented on the About page.
- A weak source (usagepricing.com) claims Qualcomm acquired Modular, BentoML's new parent, in July 2026. This is unverified and should not be stated as fact. — [usagepricing.com](https://www.usagepricing.com/blueprint/activity/bentoml-2026-07-29-qualcomm-modular-acquisition)
- No neutral, controlled benchmark comparing Ollama, llama.cpp and vLLM at different concurrency levels was found. The available figures come from individual bloggers.
