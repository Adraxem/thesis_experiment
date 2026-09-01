# NOVELTY.md — literature audit & where this thesis is actually novel

*Honest assessment for the thesis "Managing Power Behavior of ML Inference: From Edge
Device to the Data Centers." The point of this file is to tell the student — and any
committee member — what has already been done, so the novelty claim survives contact with
a reviewer. Every claim below has a citable source.*

## TL;DR verdict

Each individual move in the thesis (characterize → predict → optimize → scale) has
substantial prior art. **The defensible novelty is narrow and lives in the seams**, not in
any single RQ:

1. **Full-waveform / transient features as the prediction target.** Prior predictors output
   *scalar* watts or joules. Learning config → *waveform-shape* descriptors (peak-to-average
   crest factor, dP/dt, rise/settle, spikiness) on an edge device is materially
   under-explored. **This is the strongest wedge.**
2. **Per-primitive full-waveform signatures + compositional waveform reconstruction**
   (the proposed RQ2.5). Operator-level *energy* exists; operator-level *transient waveform*
   composition is thin. Most defensible micro-level contribution.
3. **One cheap real device threading the whole pipeline** (characterize→predict→optimize→
   scale) as a single reproducible artifact — a $250 Orin Nano as a datacenter-dynamics
   proxy/teaching platform. Integration + accessibility is a legitimate systems contribution
   even though each stage is individually derivative.

**Do NOT claim as novel:** that power distinguishes layer types / residual connections
(known — side-channel literature), that config predicts average power on Jetson (known),
per-layer scalar energy prediction (known — NeuralPower/Eyeriss), or that synchronized
training causes amplified datacenter power spikes (known — Google/Microsoft/SemiAnalysis).

**Two 2026 papers you must cite and differentiate** — they now sit closest to RQ2.5+RQ4
(see Thread 4). A committee will find them; get ahead of it.

---

## Thread 1 — Power/energy prediction of DNNs from config (RQ2 / RQ2.5)

The most well-trodden thread. Predicting DNN energy/latency from architecture is mature.

- **NeuralPower** — Cai, Juan, Stamoulis, Marculescu, ACML/PMLR 2017. Layer-wise polynomial
  regression predicting **power, runtime, energy** of CNNs on GPUs, summed to whole-network.
  This is essentially RQ2 (config→power) + RQ2.5 (per-layer composition) done nine years
  ago — but it predicts *scalar* average power/energy, **not the transient waveform.**
  https://arxiv.org/abs/1710.05420
- **Designing Energy-Efficient CNNs (Eyeriss energy tool)** — Yang, Chen, Sze, CVPR 2017.
  Per-layer analytic energy model from data movement; the canonical "energy differs by layer
  type" reference. Scalar energy. https://arxiv.org/abs/1611.05128 · https://energyestimation.mit.edu/
- **nn-Meter** — Zhang et al., MobiSys 2021. Kernel-level *latency* prediction on edge via
  kernel detection + per-kernel regression — the exact compositional paradigm RQ2.5 proposes,
  but for latency not power. https://github.com/microsoft/nn-Meter
- **DeepEn2023 — Unveiling Energy Efficiency in Deep Learning** — ACM/IEEE SEC 2023. Measures
  *and predicts* DNN energy across edge devices + an efficiency score. Very close to RQ2, but
  aggregate energy not waveform. https://arxiv.org/abs/2310.18329
- **One Proxy Device Is Enough for Hardware-Aware NAS** — Lu et al., SIGMETRICS 2022.
  Representative of the large NAS latency/energy-predictor literature. https://dl.acm.org/doi/10.1145/3491046

**Takeaway:** NeuralPower + nn-Meter together already demonstrate "config→power/latency" and
"per-primitive composition." The differentiator MUST be the *target variable* (transient
waveform features, not a scalar) and the edge focus.

---

## Thread 2 — Jetson/edge power characterization (RQ1)

Sweeping precision/batch/power-mode on Jetson with the internal INA3221 is **standard
practice, not novel on its own.**

- **Profiling Energy Consumption of DNNs on NVIDIA Jetson Nano** — Holly, Wendt, Lechner,
  2020. Power-mode/frequency sweep of DNN inference energy on a Nano. Overlaps the RQ1 core.
- **Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson
  Devices** — arXiv 2306.13107, 2023. Characterizes INA3221 accuracy / sampling / latency.
  **Cite this to defend the instrument.** https://arxiv.org/html/2306.13107
- **PolyThrottle** — arXiv 2310.19991, 2023. Energy-efficient edge inference by co-tuning
  hardware knobs + batch. Overlaps RQ3. https://arxiv.org/html/2310.19991v2
- Plus several 2023–2025 Jetson benchmarking papers (arXiv 2307.16834, 2508.08430, 2509.20160).

**Takeaway:** the FP16/FP32 × batch × nvpmodel sweep is table stakes. Novelty here is ~zero
unless the *waveform-level* observation (not average watts) is foregrounded. The
inference-vs-training comparison on the same edge device is slightly fresher.

---

## Thread 3 — Power/EM side-channel fingerprinting (the "layers are distinguishable" idea)

**Your hypothesis that power distinguishes layer types / residual connections is already
established science** — arguably with *stronger* evidence than this thesis will produce,
because these papers *recover architecture* from traces. Cite them as foundation; do not
claim to discover this.

- **CSI NN** — Batina, Bhasin, Jap, Picek, USENIX Security 2019. Recovers layer types,
  activations, neuron/filter counts, and weights from EM/power side channels. The foundational
  "power/EM encodes architecture" result. https://www.usenix.org/conference/usenixsecurity19/presentation/batina
- **Open DNN Box by Power Side-Channel Attack** — Xiang et al., 2020. Recovers DNN structural
  attributes from power side-channel on an embedded accelerator.
- **DeepEM** — Yu et al., HOST 2020. Recovers DNN details via EM leakage on an edge FPGA.
- **CNN Architecture Extraction on Edge GPU** — COSADE 2024 / arXiv 2401.13575. Architecture
  extraction on an **edge GPU (Jetson-class)** — closest platform match. https://arxiv.org/pdf/2401.13575

**Takeaway:** the thesis uses this distinguishability **constructively** (for prediction /
optimization) rather than adversarially (for extraction). That reframing is legitimate and
worth stating — but the *premise* is prior art. Frame it as "power is known to be
architecture-discriminative (CSI NN et al.); we exploit that for forward prediction."

---

## Thread 4 — Waveform/transient power & datacenter-scale synchronized spikes (RQ4)

This thread exploded in 2024–2026 and holds the thesis's most dangerous overlaps.

**Phenomenon is well documented (RQ4's premise is NOT novel):**
- **Google — Mitigating Power and Thermal Fluctuations in ML Infrastructure.** Synchronized
  power oscillations from lockstep ML training at scale. https://cloud.google.com/blog/topics/systems/mitigating-power-and-thermal-fluctuations-in-ml-infrastructure
- **Microsoft — Power Stabilization for AI Training Datacenters** — arXiv 2508.14318, 2025.
- **SemiAnalysis — AI Training Load Fluctuations at Gigawatt-scale.** The grid-risk articulation.
- **EasyRider: Mitigating Power Transients in Datacenter-Scale Training** — arXiv 2604.15522, 2026.

**⚠️ Two papers that directly occupy RQ2.5 + RQ4 territory — CITE AND DIFFERENTIATE:**
- **From Servers to Sites: Compositional Power Trace Generation of LLM Inference for
  Infrastructure Planning** — Microsoft Research, arXiv 2603.18383, 2026. **The single most
  threatening prior work:** it *compositionally generates site-level power traces by combining
  per-server traces* — exactly the "superpose per-device traces to model the aggregate" idea
  and the "compositional trace generation" framing. Differs in scale (server→site vs
  edge→datacenter), workload (LLM inference vs vision training), platform (real DC vs one
  Jetson), and — importantly — the thesis targets *transient features*. Must be addressed head
  on. https://arxiv.org/abs/2603.18383
- **Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center
  Infrastructure Planning** — arXiv 2604.07345, 2026. Waveform-level workload power profiles →
  facility planning. https://arxiv.org/html/2604.07345v1
- (Also relevant: **Smoothing the Ramp, Not the Peak** — arXiv 2608.01250, 2026 — scheduling →
  grid-scale power dynamics, the micro→macro link.)

**Takeaway:** RQ4's genuine distinction is doing it *bottom-up from a single cheap edge
device's measured traces* as an accessible proxy — a pedagogical angle, not a new phenomenon.
Coherent superposition of synchronized traces is real physics but assumed, not a finding.
**This is the most contestable claim in the thesis** (see below).

---

## Thread 5 — Per-operator power micro-benchmarking (RQ2.5)

- **nn-Meter** (kernel-level latency composition) — same paradigm, latency. https://github.com/microsoft/nn-Meter
- **Fine-Grained Energy and Performance Profiling for GPUs** — arXiv 1803.11151, 2018. Per-kernel
  energy attribution. https://arxiv.org/pdf/1803.11151
- **MIPP: microbenchmark suite for power/energy of GPU architectures** — IEEE.

**Takeaway:** operator/kernel-level *energy* attribution exists on GPUs. What is thin:
**per-primitive full-waveform** signatures on an edge device (the transient *shape* of a
depthwise-separable block vs a residual block, not just its joules) used to *reconstruct* a
full waveform. Freshest sub-area for the thesis — this is where RQ2.5 earns its keep.

---

## Strongest defensible novelty claim (use this framing)

> *A per-primitive, full-waveform (transient) power characterization on a single commodity
> edge device that (a) learns a config→waveform-feature predictor and (b) compositionally
> reconstructs and superposes waveforms to model datacenter-scale synchronized power
> transients — i.e. transient-shape prediction and compositional transient synthesis, as
> opposed to the scalar-energy prediction and mean-power scaling that dominate prior work.*

Anchor novelty on **waveform/transient features** and **compositionality of the transient**,
not on any single RQ.

## Weakest / most-contested claim (defend it or reframe it)

> The **edge-to-datacenter scale-up via coherent superposition of synchronized traces (RQ4).**

The phenomenon is documented industrially, and **Microsoft's "From Servers to Sites"
(arXiv 2603.18383, 2026)** already formalizes compositional power-trace generation for
scale-up. A reviewer will say the single-Jetson superposition is a toy proxy for effects
already measured on real fleets, and that coherent superposition is *assumed*, not validated.
**Reframe RQ4 as an accessible/illustrative proxy model, and validate the superposition
assumption against even 2–4 real synchronized devices** rather than presenting it as a
predictive datacenter model.

## Blindside flags — surface these in Related Work before a committee does

1. **arXiv 2603.18383 (Microsoft, 2026)** — compositional power-trace generation, server→site.
   Nearest collision with RQ2.5+RQ4.
2. **NeuralPower (2017)** — already did per-layer config→power prediction + composition;
   differentiate on **waveform vs scalar**.
3. **CSI NN / DeepEM / edge-GPU extraction** — "layers distinguishable in power" is prior art;
   cite as foundation, don't claim discovery.
4. **arXiv 2604.07345 & 2608.01250 (2026)** — waveform-level profiles + scheduling→grid; overlap
   the "waveform matters" and micro→macro claims.

---

*Caveat: this audit rests on abstracts/titles + prior knowledge; individual PDFs were not all
opened. Before finalizing Related Work, read in full: arXiv 2603.18383, arXiv 2604.07345,
NeuralPower (1710.05420), and CSI NN — those four most directly shape the novelty argument.*
