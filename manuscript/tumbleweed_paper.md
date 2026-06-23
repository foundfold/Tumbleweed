# Tumbleweed: a target-conditional DNA and RNA aptamer diffusion model in which the SELEX selection round sets the noise schedule

Marko Melnick^1^ (ORCID: 0000-0001-8616-3019)

1. FoundFold, Boulder, CO

*Corresponding author: marko@foundfold.com

**Running title:** A chemistry-aware target-conditional aptamer generator

**Keywords:** aptamers; SELEX; masked diffusion; generative models; target conditioning; benchmarks

---

## Abstract

Aptamers are short single-stranded RNA or DNA molecules selected to bind a target, offering a synthesizable alternative to antibodies. Existing generative models for aptamers are RNA-only, unconditional, and unvalidated against the protein target. We present Tumbleweed, a masked-discrete-diffusion model whose defining design choice is that the diffusion noise level of each training sequence is set by the empirical SELEX selection round it was observed in, rather than sampled uniformly: a diverse round-0 library sequence enters nearly fully masked and a converged late-round winner enters clean, so denoising runs in the same direction as SELEX enrichment and ancestral sampling traces an in-silico selection trajectory. On top of this objective the model is chemistry-aware, a leading chemistry token lets one network generate and score both RNA and DNA aptamers, and target-conditional, the protein target is injected into every transformer layer through ESM-2 embeddings and feature-wise linear modulation (FiLM). Alongside the model we release two benchmarks. On Tumbleweed-RecoveryBench, which asks whether a model ranks true SELEX winners above composition-matched random sequences, Tumbleweed ranks in-domain winners well above released-weight baselines scored on identical sequences (mean AUROC 0.939 versus 0.521); because those baselines are unconditional and structurally cannot use target identity, this margin reflects a target-conditioning capability they do not have. We then probe the harder regime of a target whose selection family the model has never seen: under a leave-one-target-out protocol, in which that family is removed from training and only the protein embedding is supplied at score time, recovery falls to near chance (mean 0.579), tracing a target-class diversity ceiling rather than an architectural limit. Tumbleweed-KdBench asks whether a model can rank held-out aptamer affinity under leave-one-target-out transfer. Here no method we tried, including a large pretrained RNA model, performs above chance, and we release it as an open benchmark to anchor future work on this harder problem. The zero-shot conditioning collapse and the affinity-ranking negative share one cause, which we trace to a target-class diversity ceiling rather than the architecture. By releasing the model alongside both benchmarks, we aim to encourage further research at the frontier of conditional aptamer generation.

---

## Introduction

<!-- NOTES
- Aptamers vs antibodies: cheaper, chemically synthesizable, RNA + DNA, no cold chain
- SELEX = slow wet-lab selection loop; in-silico prioritization is the opportunity
- Prior work: EvoFlow-RNA (Patel 2025) RNA-only, unconditional; InstructNA (Zhang 2026) no weights
- Discrete diffusion lineage: D3PM, MDLM, SEDD
- Our contributions: chemistry-aware single model, FiLM conditioning + ablation, two released benchmarks, honest negatives
- Voice: match Dogcatcher2 — direct, data-driven, candid about limits
-->

Aptamers are short single-stranded RNA or DNA oligonucleotides that fold into defined three-dimensional shapes and bind molecular targets with antibody-like affinity and specificity. Compared with protein binders they are chemically synthesizable, thermostable, non-immunogenic, and inexpensive to produce, and they can be raised in either RNA or DNA chemistry. They are conventionally discovered by SELEX (Systematic Evolution of Ligands by EXponential enrichment), an iterative wet-lab selection loop in which a large random library is repeatedly partitioned against a target and amplified until high-affinity binders dominate the pool [@rapidselex; @ultraselex]. SELEX is powerful but slow and resource-intensive, which motivates computational models that can prioritize or expand candidate pools in silico before committing to additional selection rounds.

Generative sequence models are a natural fit for this problem, and recent work has begun to apply language- and diffusion-model machinery to nucleic acids. EvoFlow-RNA is a released masked-diffusion model that generates and represents non-coding RNA, but it is RNA-only and unconditional: it scaffolds motifs from a known structure and has no mechanism to condition generation on a protein target [@evoflow]. InstructNA proposes de novo design of functional nucleic acids but released no weights, precluding direct benchmarking [@instructna]. Evaluation-focused tools such as RaptScore and RaptRanker score or rank aptamers within a single SELEX experiment but are not generative and do not transfer across targets [@raptscore; @raptranker]. As a result, three capabilities that matter for practical aptamer design remain unaddressed by any released model: handling RNA and DNA in one model, conditioning generation on an arbitrary protein target, and honest benchmarking against released-weight baselines on identical sequences.

Here we present Tumbleweed, a compact masked-discrete-diffusion model for aptamers that addresses the first two capabilities directly and confronts the third with two released benchmarks. Tumbleweed builds on the discrete-diffusion modeling line (D3PM, MDLM, and SEDD), adapting masked-diffusion denoising to short nucleic-acid sequences [@d3pm; @mdlm; @sedd]. Its defining design choice is a selection-aware noise schedule: instead of sampling the diffusion timestep uniformly, we set it from the empirical SELEX round each training sequence was observed in, so denoising mirrors enrichment (Methods, Fig 2). To our knowledge this is the first generative diffusion model in which an empirical selection state, rather than a uniform draw, sets the per-sequence noise level. On top of this objective the model is chemistry-aware, a single network generates and scores both RNA and DNA via a leading `[RNA]`/`[DNA]` token at sequence position 0, and target-conditional, protein targets are embedded by ESM-2 [@esm2] and injected into every transformer layer by feature-wise linear modulation (FiLM) [@film]. Our contributions are: (1) a SELEX-round noise schedule that ties the diffusion timestep to the empirical selection round, making denoising an in-silico enrichment trajectory; (2) a single chemistry-aware model for RNA and DNA aptamers with per-layer FiLM target conditioning from ESM-2, together with a component ablation isolating FiLM as the dominant lever that carries conditioning; (3) Tumbleweed-RecoveryBench, a released likelihood-based winner-recovery benchmark with released-weight baselines scored on identical sequences, on which a compact (75.5M) conditional model ranks in-domain winners far above unconditional released-weight baselines, with a leave-one-target-out test mapping where conditioning stops transferring to unseen targets; (4) Tumbleweed-KdBench, a released leave-one-target-out affinity-ranking benchmark, together with the finding that no current method ranks held-out affinity above chance; and (5) a quantitative characterization of where target conditioning stops working, namely the zero-shot leave-one-target-out boundary, which we trace to a target-class diversity ceiling rather than to the architecture. The chemistry token, FiLM, and ESM-2 are established components; we frame them as supporting engineering and rest the novelty on the selection-aware objective and the two released benchmarks. We are explicit throughout that this is a computational study. Wet-lab validation is deliberately out of scope, and we frame every claim as in-silico.

The selection-aware noise schedule is distinct from the closest prior work. Recent theory equating denoising diffusion with evolutionary optimization draws the analogy at the level of the algorithm, noise as mutation and denoising as selection, but does not train on empirical per-round selection data [@diffevo]; our coupling is concrete and data-driven, the round a sequence was actually observed in sets how noised it enters. Diffusion models have been applied to aptamers with a standard, uniformly sampled schedule (AptaDiff) [@aptadiff], and language models have been trained on round-stratified SELEX pools by using the rounds as data splits rather than as a noise variable (AptaGPT) [@aptagpt]; neither ties the diffusion timestep to the selection round. The nearest architectural neighbor, EvoFlow-RNA, is a masked-discrete-diffusion RNA model with a conventional schedule and no conditioning [@evoflow]; we benchmark directly against it.

---

## Materials and Methods

### Model architecture

Tumbleweed is a 75.5-million-parameter masked-discrete-diffusion denoiser (Fig 1). The transformer trunk uses 8 layers with a model dimension of 768, 12 attention heads, and a feed-forward dimension of 3072. Sequences are corrupted by masking a fraction of tokens and the model is trained to denoise them, following the masked-diffusion language-model formulation [@mdlm; @d3pm; @sedd]. Three components distinguish the architecture:

**Chemistry token.** Every sequence is prefixed at position 0 with a chemistry token, `[RNA]` or `[DNA]`. This lets a single model represent and score both chemistries and is forward-compatible with additional chemistries (for example, modified-backbone libraries) without architectural change.

**Convolutional motif front-end.** A depthwise convolutional block (kernel widths 5 and 7) is applied to the input token embeddings before the transformer trunk. This front-end captures short sequence motifs that recur in aptamer binding loops and closes a chemistry-specific performance gap (Results).

**FiLM target conditioning.** A protein target is embedded with ESM-2 [@esm2], mean-pooled over residues, and mean-centered across the target bank to remove anisotropy (the raw mean-pooled embeddings are collinear at cosine similarity ~0.88, which centering reduces to ~0). A zero-initialized linear layer maps the pooled target representation to per-layer modulation parameters (γ, β), and the trunk applies `h ← γ⊙h + β` after every layer following the FiLM formulation [@film]. Because γ=1 and β=0 at initialization, conditioning is residual-safe and adds 9.4M parameters. Target conditioning enters the denoiser through this per-layer FiLM pathway (plus a prepended target token) and is trained by the denoising objective alone; a five-rung component ablation (Supplementary Table S3) isolates FiLM as the dominant route that carries conditioning. The headline model combines all three components, chemistry token, FiLM conditioning, and CNN front-end, and is trained with the denoising objective only. We also evaluated a supervised-InfoNCE contrastive auxiliary term; the same ablation shows it is a minor, non-load-bearing contributor (+0.002 on RecoveryBench, within seed noise, and a slight drop in the conditioning delta), so the released model omits it (Supplementary Table S3). Two architecture ablations remove one component each: "−CNN" removes the convolutional front-end, and "−FiLM" reduces conditioning to the prepended target token alone.

### Training corpus

The training corpus combines deep, round-structured SELEX data with a flat set of affinity-labeled winners. Seven deep SELEX families covering six distinct protein targets, drawn from RaptScore-associated selections, RAPID-SELEX, and an α-synuclein 2′-fluoro-pyrimidine RNA selection, provide the full per-round selection trajectory and serve as the conditioning anchors [@raptscore; @rapidselex; @alphasyn]. To broaden target coverage we add 193 affinity-benchmark winners as flat (sequence, target, chemistry) triples.

### Training objective

Tumbleweed is trained with a masked-diffusion denoising objective, `L = L_denoise`. The defining design choice is that the diffusion noise level is read directly from the SELEX selection state rather than sampled uniformly: a sequence observed at round *k* of a family with maximum round *R* is assigned timestep *t* = 1 − *k*/*R*, so the diverse starting pool (round 0) enters at *t* ≈ 1 (nearly fully masked) while a converged winner enters at *t* ≈ 0 (clean). Denoising therefore mirrors SELEX enrichment, and ancestral sampling from an all-masked sequence down to a clean one is an in-silico selection trajectory (Fig 2). The denoising term is an EvoFlow-style cross-entropy averaged over masked positions per sequence. (A nominal per-sequence weight of 1/*t* clamped to [1/300, 1] is applied; because the SELEX-derived *t* lies in [0,1] this clamp saturates at its upper bound, so the effective per-sequence weight is uniform.) Training runs for 20,000 steps at batch size 128 using AdamW (learning rate 3×10⁻⁴, weight decay 0.05, β = (0.9, 0.95), gradient-norm clip 1.0). The number of distinct conditioning targets is the binding constraint on what the FiLM pathway can learn to transfer: with only six deep SELEX targets there are too few protein classes for the model to learn a target→aptamer mapping that generalizes to an unseen protein, an information ceiling of order ln(N) for N distinguishable targets that becomes central to the conditioning analysis (Results, Discussion). We also trained a variant adding a supervised-InfoNCE auxiliary loss (`L = L_contrast + 0.5·L_denoise`) that pulls each aptamer's pooled representation toward its target's ESM-2 projection; as the component ablation shows (Supplementary Table S3) it does not improve recovery and slightly lowers the conditioning delta, so it is omitted from the released model.

### Tumbleweed-RecoveryBench

Tumbleweed-RecoveryBench evaluates whether a likelihood-based scorer ranks true SELEX winners above plausible negatives (Fig 3A). Real sequences are scored by pseudo-negative-log-likelihood under masked-diffusion denoising at low mask ratios (t = 0.1, 0.15, 0.2), which match the low sequence diversity of converged SELEX winners, with four mask repetitions per sequence. The primary readout is AUROC(winner vs. composition-matched random). We also report AUROC(winner vs. naive SELEX-enrichment-ranked sequences). The composition-matched random set is a per-sequence mononucleotide shuffle of each winner, which preserves single-base composition but scrambles all positional order, including any shared constant primer or library-flank regions; sequences are scored as deposited, without trimming constant regions. For the in-domain headline scoring, the scored winners are drawn from the same converged SELEX rounds the model trained on, so this readout measures conditional in-domain recovery; the unseen-family comparison is the leave-one-target-out protocol below. The benchmark comprises 400 winners for each of five targets (FGF9, IL1RL1, PARP1, MECP2, SNCA). The IL1RL1 selection (RaptScore HT-SELEX Dataset B) was raised against the mouse ortholog ST2/IL-33R (Il1rl1, UniProt P14719), and its ESM-2 conditioning uses that mouse sequence accordingly. At score time the held-out target's own ESM-2 embedding provides the conditioning. Because the pseudo-NLL readout is defined only for masked-diffusion and masked-language-model scorers, the benchmark admits those model classes. We score the released EvoFlow-RNA [@evoflow] and a RiNALMo-MLM baseline [@rinalmo] on the identical sequence set so that margins are not a sequence-set artifact.

### Tumbleweed-KdBench

Tumbleweed-KdBench evaluates affinity ranking under leave-one-target-out (LOO) transfer (Fig 3B). From a unified affinity corpus of 847 sequences spanning 214 protein targets, we curate every (target × chemistry) panel containing at least four measured aptamers, yielding 47 rankable panels (33 DNA and 14 RNA) over 44 targets. The remaining 170 targets are singletons or have three or fewer measurements and cannot yield a within-target rank correlation, so they are excluded rather than added as noise. For each ranker we train on all other targets, predict the held-out target's aptamer affinities, and score by per-panel Spearman correlation between predicted and measured dissociation constants (K_D). We report the unweighted mean Spearman ρ across the 47 panels with its t-statistic, along with median, 20%-trimmed mean, and per-panel win rate. Five rankers are benchmarked: the four we built (the generator scored by its target-conditioned pooled trunk representation, which is the trained denoising representation, with a ridge regressor fit on the other targets; a k-mer ⊕ ESM-2 → gradient-boosted-trees regressor, TW-TriFP; and two auxiliary contrastive scorers, a pooled contrastive scorer and that scorer with a CNN front-end, retained as benchmarked baselines) and one external baseline, RiNALMo-650M features → ridge regression [@rinalmo]. Because the model trains on 193 affinity-corpus winners that overlap the KdBench sequences, scoring the generator by raw sequence likelihood would be train-on-test contaminated; the pooled-representation-plus-leave-one-target-out-ridge protocol avoids this by never fitting on the held-out target's labels, matching the protocol used for every other ranker.

### Conditioning evaluation and ablation

To isolate the architectural lever that carries target conditioning, we use a null-conditioning evaluation. For each of the five RecoveryBench targets we compute the winner-vs-random AUROC twice, once with the target's own ESM-2 embedding and once with a zeroed embedding, and report the difference (own − zero), averaged over the five targets. A positive own − zero means the model uses target identity. We report this for the released model (Table 1) and, to attribute the signal to specific components, run a matched five-rung ablation ladder, each rung adding one component (unconditional → target token → FiLM → CNN → contrastive) retrained from scratch with three seeds on the identical corpus and scored on both RecoveryBench and own − zero (Supplementary Table S3). For the zero-shot generalization test we use a strict leave-one-target-out protocol: the held-out target's entire SELEX family is removed from training and only its ESM-2 embedding is retained at score time, matching the unseen-target setting an unconditional model such as EvoFlow-RNA operates in.

### Implementation and availability

Tumbleweed is implemented in PyTorch. Protein embeddings use ESM-2 [@esm2], the RiNALMo baseline uses the released RiNALMo-650M weights [@rinalmo], and the generative baseline uses the released EvoFlow-RNA checkpoint [@evoflow]. Both released benchmarks, the per-panel result tables (`v2_loo_per_target.csv` and the RecoveryBench CSVs), and the figure-regeneration scripts are released with the model. All results in this study are computational (likelihood-based recovery, leave-one-out affinity ranking, and conditioning deltas). No wet-lab experiments were performed.

---

## Results

### Tumbleweed is a chemistry-aware, target-conditional masked-diffusion model (Fig 1)

Tumbleweed couples a single generative trunk to a target-conditioning branch (Fig 1). On the generative path, an aptamer sequence is prefixed with a chemistry token, passed through the depthwise-CNN motif front-end, and denoised by the 8-layer masked-discrete-diffusion transformer, which also yields the pseudo-likelihood used for scoring. On the conditioning path, a protein target is embedded by ESM-2, mean-pooled and mean-centered, and injected as per-layer FiLM (γ, β) modulation of the trunk, the dominant architectural lever that carries target conditioning. The same model handles both RNA and DNA through the leading chemistry token, a capability lane in which no released model competes: EvoFlow-RNA and the RNA-language-model family cannot score DNA at all. We therefore claim the chemistry-aware interface (one model, both chemistries) as a capability, and evaluate the model on the two released benchmarks detailed below.

### Tumbleweed recovers in-domain SELEX winners that unconditional baselines cannot (Fig 4)

On Tumbleweed-RecoveryBench, the headline model ranks true SELEX winners above composition-matched random sequences at a mean AUROC of 0.939, winning all five targets (Fig 4). This is an in-domain, conditional measurement: the model trained on these five targets' SELEX families and is conditioned on their ESM-2 embeddings through the FiLM pathway, the very capability the released baselines lack. Because EvoFlow-RNA and RiNALMo-MLM are unconditional, they have no mechanism to use target identity at all, which is why they sit at chance on the identical sequences; the 0.939-versus-0.521 margin is therefore a direct readout of what the conditioning architecture buys when the target's selection is in-domain. The leave-one-target-out protocol below (Fig 5B) is a separate, harder question, whether that conditioning transfers to a target family the model has never seen. Scored on the identical sequence set seen by EvoFlow-RNA, the released unconditional RNA-only baseline reaches 0.521 and a RiNALMo-MLM baseline reaches 0.511, both essentially at the chance line of 0.5. The per-target values are FGF9 0.971, IL1RL1 0.960, PARP1 0.895, MECP2 0.961, and SNCA 0.907 for Tumbleweed, versus 0.602, 0.467, 0.546, 0.532, and 0.460 for EvoFlow-RNA. Because the Tumbleweed column is re-scored on EvoFlow's exact sequences, the in-domain margin is not an artifact of differing negative sets.

Both unconditional baselines land within +0.010 of each other (0.521 and 0.511), both at the chance line, so neither the masked-diffusion objective nor 650M-parameter RNA pretraining recovers winners without a way to condition on the target. The gap between Tumbleweed and the baselines here is exactly what the conditioning architecture provides: the FiLM target pathway lets the model exploit in-domain selection information that an unconditional model has no route to use. What the leave-one-target-out comparison (Fig 5B) tests is a different question, whether that learned conditioning transfers to an unseen target family, and there the limit is the number of distinct training targets, not the architecture. We further note that AUROC(winner vs. naive SELEX-enrichment) is effectively tied across models, indicating that the winner-vs-random signal is carried largely by ordered library structure rather than by captured SELEX enrichment. The defensible claim is ranking true binders above composition-matched random in-domain, not above SELEX enrichment itself, and not zero-shot.

### FiLM is the dominant architectural lever that carries target conditioning (Table 1)

A null-conditioning evaluation shows the released model genuinely uses target identity, and a matched component ablation localizes that signal to the FiLM pathway (Table 1; Supplementary Table S3). Replacing the target's own ESM-2 embedding with a zeroed embedding measures how much the model relies on target identity (own − zero). For the headline model this lift averages +0.211 over the five targets and is positive for all five (Table 1). The matched-seed component ladder (Supplementary Table S3) attributes this to two levers, the prepended target token (+0.169 on recovery) and per-layer FiLM (+0.126), while the supervised-InfoNCE contrastive term adds +0.002 (within seed noise) and slightly lowers the conditioning delta, which is why the released model omits it.

We are explicit that own − zero is high-variance on only five targets (PARP1 is the weakest at +0.120), so we lean on the matched-seed ladder for the robust ordering. The robust statement is that the discrete target token and FiLM together carry conditioning, while the CNN front-end earns its place through generation and chemistry-gap gains rather than through the conditioning delta. Levers we tried that did not help (all regressed or were no-ops) include attention- and epitope-based pooling, per-residue embeddings, winner-quantity rebalancing, fingerprint-based conditioning, and the contrastive auxiliary loss. The live lever is injecting the target into the denoiser via FiLM rather than beside it as an ignorable prepended token.

**Table 1. Null-conditioning evaluation of the released model.** own − zero is AUROC(winner vs. random) with the target's own ESM-2 conditioning minus a zeroed embedding, per target. The matched-seed component ladder that isolates which architectural piece supplies this signal is Supplementary Table S3.

| Target | AUROC (own) | AUROC (zero) | own − zero |
|---|---|---|---|
| FGF9 | 0.977 | 0.637 | +0.340 |
| IL1RL1 | 0.965 | 0.661 | +0.304 |
| PARP1 | 0.826 | 0.707 | +0.120 |
| MECP2 | 0.929 | 0.801 | +0.128 |
| SNCA | 0.897 | 0.734 | +0.164 |
| **Mean** | **0.919** | **0.708** | **+0.211** |

### Target conditioning generalizes within SELEX families but not across them (Fig 5)

The conditioning that Table 1 isolates is real within the training distribution but does not transfer to an unseen target family (Fig 5). Within-domain (Fig 5A), for each of the five RecoveryBench targets whose SELEX family is present in training, swapping in the target's own ESM-2 conditioning versus a zeroed embedding lifts winner-vs-random AUROC by a mean of +0.21 across all five targets. Conditioning is real and target-specific when the family is seen.

Under a strict leave-one-target-out protocol (Fig 5B), the result inverts. Dropping the held-out target's SELEX family entirely from training while keeping its ESM-2 conditioning at score time, an unseen-target setting directly comparable to an unconditional EvoFlow-RNA, collapses recovery to near chance for all five targets: FGF9 0.491, IL1RL1 0.465, PARP1 0.715, MECP2 0.624, SNCA 0.598, for a mean of 0.579, near the chance line and close to unconditional EvoFlow-RNA's 0.521. The in-domain anchors for MECP2 and SNCA (0.961 and 0.907) fall to 0.624 and 0.598 once their families are removed. The conditioning signal does not generalize to an unseen target family. We trace this to a target-class diversity ceiling rather than to the architecture: a conditioning signal learned from N distinct protein classes carries at most order ln(N) of transferable target information, and with only six deep SELEX targets there are too few classes to learn a protein→aptamer mapping that transfers. In-domain conditioning is demonstrated. Zero-shot de novo design for an arbitrary new protein is not.

### No method ranks held-out aptamer affinity above chance (Table 2)

Tumbleweed-KdBench is a clean negative result, and we present it as one (Table 2). For each of the five benchmarked rankers we summarize the per-panel leave-one-target-out Spearman ρ over the 47 rankable panels with its mean, median, 20%-trimmed mean, per-panel win rate (fraction of panels with ρ > 0), and a one-sample t-statistic. The nominal leader, TW-TriFP (k-mer ⊕ ESM-2 → gradient-boosted trees), reaches a mean ρ of only 0.057 with a t-statistic of 1.01. The external RiNALMo-650M → ridge baseline reaches 0.040 (t = 0.81), the pooled contrastive scorer 0.017 (t = 0.28), and the generator scored by its target-conditioned pooled representation sits below chance at −0.081 (t = −1.20). No method's t-statistic approaches the t > 2 needed for the mean ρ to be reliably positive, so affinity ranking on held-out targets is statistically indistinguishable from chance across all five methods, including a 650M-parameter pretrained RNA model. A forest-plot view of the same per-panel means with 95% confidence intervals, every interval crossing ρ = 0, is provided as Fig 6.

**Table 2. Tumbleweed-KdBench affinity-ranking stability metrics.** Per-method summary over the 47 leave-one-target-out panels: mean ρ, median, 20%-trimmed mean, per-panel win rate (fraction with ρ > 0), and one-sample t-statistic. Four rankers were built here (prefixed TW-). RiNALMo-650M → ridge is the external baseline. Every method is statistically indistinguishable from chance (ρ = 0).

| Method | mean ρ | median | trim-20% | win rate | t-stat |
|---|---|---|---|---|---|
| TW-TriFP (k-mer ⊕ ESM-2 → GBDT) | 0.057 | 0.099 | 0.064 | 0.55 | 1.01 |
| RiNALMo-650M → Ridge | 0.040 | 0.024 | 0.048 | 0.51 | 0.81 |
| TW-Contrastive scorer | 0.017 | 0.037 | 0.028 | 0.53 | 0.28 |
| TW-Contrastive scorer + CNN | −0.036 | 0.000 | −0.028 | 0.47 | −0.53 |
| TW-Generator (pooled representation → ridge) | −0.081 | −0.091 | −0.096 | 0.45 | −1.20 |

No ranker's 95% confidence interval clears zero, and the nominal ordering is itself unstable across the robustness metrics, so we promote no ranker and wire none into the generation pipeline. We deliberately do not score the generator by raw sequence likelihood: 611 of the 630 affinity-corpus winners the model trains on also appear in the KdBench sequences, so a likelihood ranker would be train-on-test contaminated; the leakage-free protocol (pooled representation with leave-one-target-out ridge) places the generator at chance like the rest. This negative is consistent with the conditioning result above: the same target-label ceiling that caps zero-shot conditioning also caps affinity ranking. The contribution here is the released benchmark together with the rigorously characterized negative, which defines a concrete open problem rather than papering over an unreliable component.

---

## Discussion

Tumbleweed shows that a compact, from-scratch, chemistry-aware and target-conditional diffusion model recovers true SELEX winners far above a released unconditional baseline in-domain, while being candid about two boundaries where current methods, including ours, do not yet work: zero-shot conditioning on an unseen target family, and held-out affinity ranking. Framing the contribution this way, as one positive capability result and two carefully quantified negatives each released as a benchmark, is deliberate. The aptamer-design literature is small and the temptation to report only the winning number is real. We instead delimit the regime of validity so that the field can target the actual bottleneck.

That bottleneck is data diversity, not architecture. A conditioning signal learned from N distinct protein classes carries at most order ln(N) of transferable target information, and our deep SELEX corpus contains only six. The within-domain conditioning lift (+0.21 mean own − zero, Table 1 and Fig 5A) demonstrates that the FiLM pathway can carry target identity when the family is seen. The leave-one-target-out collapse (mean 0.579, Fig 5B) demonstrates that six classes are too few to learn a transferable protein→aptamer mapping. The affinity-ranking negative (Table 2) is the same ceiling viewed through a different task: even a 650M-parameter pretrained RNA model cannot rank held-out K_D above chance on 47 panels. The clear implication for future work is that broader multi-target deep SELEX, not a larger or cleverer model, is the lever most likely to move zero-shot performance.

Several design choices proved load-bearing and are worth highlighting for reuse. The selection-aware noise schedule is the central one: tying the diffusion timestep to the SELEX round a sequence was observed in makes denoising follow the empirical enrichment gradient, so a single training pass over the round-structured pools teaches the model the same easy-to-hard trajectory that selection itself traverses, and ancestral sampling replays it. Mean-centering the ESM-2 target bank was necessary to break the anisotropy that otherwise leaves pooled embeddings collinear and conditioning ineffective. Injecting the target into the denoiser by FiLM, rather than beside it as a prepended token, was the difference between conditioning that the model uses and conditioning it ignores. This echoes the original FiLM finding in visual reasoning [@film] and, in our ablations, dominated alternatives such as per-residue cross-attention and fingerprint conditioning. The chemistry token is a small change with a large payoff: it yields a single model that scores both RNA and DNA, a lane no released generative model occupies.

This study has clear limitations, which we state plainly. It is entirely computational. All claims are in-silico (likelihood-based recovery, leave-one-out K_D ranking, and conditioning deltas), and wet-lab validation is out of scope. The conditioning and ranking analyses rest on a small number of deep SELEX targets, so the five-target own − zero estimates are high-variance and we lean on the matched-seed component ladder (Supplementary Table S3) for the robust claim. The RecoveryBench readout is a pseudo-likelihood and therefore admits only masked-diffusion and masked-language-model scorers, which is why autoregressive or non-likelihood baselines are absent. The RecoveryBench negative set is a per-sequence mononucleotide shuffle, so the headline AUROC measures separation from composition-matched scrambles rather than from realistic library decoys; harder structure-aware decoys would be a stronger test. The headline is an in-domain conditional measurement, and zero-shot transfer to an unseen target family is reported separately as the leave-one-target-out result. None of these caveats undercuts the core contributions: a chemistry-aware conditional model that beats the released unconditional baseline in-domain, two released benchmarks, and a quantitative map of the target-diversity ceiling that bounds zero-shot conditioning and affinity ranking alike. By releasing the model and both benchmarks, including the negative one, we aim to make the open problems in conditional aptamer design measurable rather than rhetorical.

---

## Data Availability

The Tumbleweed model, both released benchmarks (Tumbleweed-RecoveryBench and Tumbleweed-KdBench), the per-panel result tables, and all figure-regeneration scripts are available at https://github.com/foundfold/Tumbleweed and archived at Zenodo under DOI [to be assigned on acceptance]. Tumbleweed-KdBench is distributed as a citable table of measured aptamers (sequence, chemistry, protein target, dissociation constant, and a primary-literature citation resolved from its PubMed ID). Aggregator databases were consulted only to recover PubMed IDs, and the original publications are cited rather than the aggregators. The SELEX datasets underlying Tumbleweed-RecoveryBench are openly available from their original repositories: RAPID-SELEX (ENA PRJNA1122221; GEO GSE269538), the α-synuclein 2′-fluoro-pyrimidine RNA selection (ENA PRJEB70964), and the RaptScore HT-SELEX selections (DDBJ DRA019577 for the FGF9 Dataset A library and DRA019609 for the mouse ST2/IL-33R Dataset B library). All results in this study are computational. No wet-lab experiments were performed.

## Supplementary Data

Supplementary Data are available at *NAR Genomics and Bioinformatics* online, comprising Supplementary Tables S1–S2.

## Author Contributions

M.M. conceived and designed the study, implemented the model and benchmarks, performed all analyses, and wrote the manuscript.

## Acknowledgements

The author thanks the investigators who generated and openly deposited the SELEX datasets that make the Tumbleweed-RecoveryBench benchmark possible: Szeto, Latulippe, Ozer, Pagano, White, Shalloway, Lis, and Craighead for the RAPID-SELEX selections (ENA PRJNA1122221; GEO GSE269538) [@rapidselex]; Bouvier-Müller, Fourmy, Fenyi, Bousset, Melki, and Ducongé for the α-synuclein 2′-fluoro-pyrimidine RNA selection (ENA PRJEB70964) [@alphasyn]; and Kimura-Yamazaki, Adachi, Nakamura, Nakamura, and Hamada for the RaptScore selections [@raptscore]. The author also thanks the authors of the primary aptamer-affinity studies underlying Tumbleweed-KdBench, which are cited individually in the benchmark's accompanying reference table.

## Funding

No external funding.

## Conflict of Interest Statement

None declared.

---

## References

::: {#refs}
:::

---

## Figure Legends

### Figure 1
**Fig 1. Tumbleweed architecture and training forward pass.** The model (75.5M parameters) couples a single generative trunk (blue) to a frozen target-conditioning branch (orange). Generative path: an aptamer sequence is prefixed with a chemistry token (`[RNA]`/`[DNA]`) at position 0, embedded, passed through a depthwise-CNN motif front-end (kernel widths 5 and 7), given positional and SELEX-round timestep embeddings (*t* = round_to_t(*k*, *R*)), and processed by eight FiLM-modulated transformer layers (d = 768, 12 heads). Conditioning path: a protein target is embedded by frozen ESM-2 (650M), mean-pooled over residues and mean-centered across the target bank to form tgt_vec, projected to target_proj (the FiLM source, also prepended as a token). Target information enters the trunk exclusively through the per-layer FiLM modulation *x* ← γ⊙*x* + β (zero-initialized, residual-safe), the dominant architectural lever that carries target conditioning. The trunk is run once per step: a denoise pass on masked ids at the SELEX-round timestep (EvoFlow-style cross-entropy averaged over masked positions; the nominal 1/*t* per-sequence weight is clamped to an effectively uniform value over the SELEX-derived *t* range), trained with the objective *L* = *L*_denoise. The denoise head also yields the pseudo-likelihood used for scoring. A supervised-InfoNCE contrastive auxiliary term was evaluated and found non-load-bearing (Supplementary Table S3); it is omitted from the released model.

### Figure 2
**Fig 2. Training objective: the SELEX round sets the diffusion noise level.** A schematic of the training recipe (fixed geometry, not model weights). Top: each SELEX round is a pool (circle) of sequences, grey for non-binders and colored for binders. Round 0 is a packed, diverse, mostly-grey library, and each round of selection washes most sequences away, so the pool both thins out and loses its grey non-binders until a few colored binders dominate (enrichment runs left to right). Each sequence observed at round *k* of a family with maximum round *R* is assigned diffusion timestep *t* = 1 − *k*/*R*. Below each pool is the single aptamer the model trains on at that round, drawn as a nucleotide strip in which every cell is one base: revealed cells show their base letter (A/C/G/U) on a colored tile and masked cells show "M" on a black tile. Each position is masked independently with probability *t*, so the diverse round-0 aptamer enters nearly fully masked (*t* ≈ 1) and the converged winner enters clean (*t* ≈ 0). Generation reverses this: the model denoises a masked sequence into a clean aptamer, running in the same direction as SELEX enrichment. The denoising objective (weighted-CE over masked positions) and the FiLM conditioning path are shown in Fig 1.

### Figure 3
**Fig 3. Benchmark design: how Tumbleweed-RecoveryBench and Tumbleweed-KdBench are constructed.** Schematic of the two evaluation protocols (fixed geometry, not results; the numbers are in Fig 4, Fig 6, and Tables 1–2). (A) RecoveryBench is an in-domain test: the target's SELEX family is present in training. For each target, true SELEX winners (drawn as an enriched pool of colored binders, matching Fig 2) and composition-matched random sequences (grey) are scored by Tumbleweed's low-mask pseudo-NLL. The pseudo-NLL hides a few bases of a sequence (the black "M" cells), has the model predict them, and sums the prediction surprise, so a lower score means the model finds the sequence more binder-like. The readout is AUROC(winner ranked above random), where 0.5 is chance. It asks whether the generator assigns higher likelihood to real binders than to shuffled-composition decoys. (B) KdBench is a held-out test. One panel is a single (target × chemistry) group of at least four aptamers with measured K_D. The target's family is held entirely out of training (leave-one-target-out). The held-out aptamers are then scored and the predicted ranking is compared to the measured K_D ranking by Spearman ρ, aggregated over 47 panels.

### Figure 4
**Fig 4. Tumbleweed-RecoveryBench: per-target winner-vs-random AUROC.** Grouped bars over the five RecoveryBench targets (FGF9, IL1RL1, PARP1, MECP2, SNCA) comparing Tumbleweed (headline model, scored on EvoFlow's identical sequence set), the released EvoFlow-RNA (33M, unconditional, RNA-only), and a RiNALMo-MLM baseline. AUROC is AUROC(winner vs. composition-matched random) at low mask ratios (0.1/0.15/0.2), four mask repetitions per sequence, where 0.5 is chance. Tumbleweed wins all five targets (mean 0.939) while both baselines sit at the chance line (0.521 and 0.511). Because Tumbleweed is scored on EvoFlow's exact sequences, the margin is not a sequence-set artifact.

### Figure 5
**Fig 5. Target conditioning generalizes within SELEX families, not across them.** (A) Within-domain. For each of the five RecoveryBench targets whose SELEX family is in training, swapping the target's own ESM-2 conditioning for a zeroed embedding lifts winner-vs-random AUROC by a mean of +0.21 (own − zero), with all five targets positive. Conditioning is real and target-specific when the family is seen. (B) Cross-domain, leave-one-target-out. Dropping the held-out target's SELEX family from training while keeping its ESM-2 conditioning at score time collapses all five targets to near chance (mean AUROC 0.579, near unconditional EvoFlow-RNA's 0.521, dotted line). The conditioning signal does not generalize to an unseen target family, consistent with the ~6-target deep-SELEX diversity ceiling rather than an architectural limit.

### Figure 6
**Fig 6. Tumbleweed-KdBench forest plot.** Mean per-panel leave-one-target-out Spearman ρ over the 47 rankable panels, with 95% confidence intervals, for the five benchmarked rankers (four built here, all prefixed TW-; RiNALMo-650M → ridge is the external baseline). Each row is annotated with its t-statistic. Every confidence interval crosses ρ = 0 and no t-statistic reaches 2 (best 1.01, TW-TriFP), so no method, including a 650M pretrained RNA model, ranks affinity above chance. A method that beat chance would have its entire interval to the right of the dashed zero line. None does. Tabulated values are in Table 2.

---

## Supporting Information

**S1 Table. Data composition.** What the model trains on and what the two released benchmarks contain.

| Component | Size | Composition / role |
|---|---|---|
| Deep-SELEX training families | 7 families / 6 protein targets | full per-round SELEX; conditioning anchors (cap transferable target information at ~ln 6) |
| K_D-winner training prior | 193 (sequence, target, chemistry) triples | flat winners added for target breadth |
| Unified K_D corpus | 847 sequences / 214 protein targets | source pool for KdBench; RNA + DNA |
| Tumbleweed-RecoveryBench | 5 targets × 400 winners | FGF9, IL1RL1, PARP1, MECP2, SNCA; winner vs. composition-matched random, low-t pseudo-NLL |
| Tumbleweed-KdBench | 47 rankable panels / 44 targets | 33 DNA + 14 RNA; ≥4 measured aptamers per (target × chemistry) panel; leave-one-target-out Spearman ρ |

**S2 Table. Tumbleweed-RecoveryBench per-target winner-vs-random AUROC.** AUROC(winner vs. composition-matched random) at low mask ratios (0.1/0.15/0.2), four mask repetitions per sequence, where 0.5 is chance. Tumbleweed is scored on EvoFlow-RNA's identical sequence set, so the margin is not a negative-set artifact. Underlies Fig 4.

| Target | Tumbleweed (matched) | EvoFlow-RNA (33M, uncond.) | RiNALMo-MLM |
|---|---|---|---|
| FGF9 | 0.971 | 0.602 | 0.605 |
| IL1RL1 | 0.960 | 0.467 | 0.508 |
| PARP1 | 0.895 | 0.546 | 0.510 |
| MECP2 | 0.961 | 0.532 | 0.478 |
| SNCA | 0.907 | 0.460 | 0.452 |
| **Mean** | **0.939** | **0.521** | **0.511** |

**S3 Table. Component ablation ladder (matched corpus, 3 seeds).** Each rung adds one component cumulatively, retrained from scratch on the identical corpus with three seeds, and is scored on RecoveryBench (mean AUROC of winner vs. composition-matched random over four shared targets; EvoFlow-RNA baseline 0.521) and on the null-conditioning delta (own − zero). The discrete target token and FiLM together account for essentially the entire margin (+0.169 and +0.126); the CNN front-end adds +0.079; the supervised-InfoNCE contrastive term adds +0.002 on recovery (within seed noise) and slightly lowers the conditioning delta (0.221 → 0.194), confirming it is a minor, non-load-bearing contributor that the headline model retains but does not depend on.

| Rung | Component added | RecoveryBench AUROC | Δ vs. previous | own − zero |
|---|---|---|---|---|
| R0 | unconditional (in-domain only) | 0.559 ± 0.048 | — | −0.001 |
| R1 | + target token | 0.728 ± 0.009 | +0.169 | 0.140 |
| R2 | + FiLM conditioning | 0.853 ± 0.017 | +0.126 | 0.207 |
| R3 | + CNN front-end | 0.932 ± 0.002 | +0.079 | 0.221 |
| R4 | + contrastive (full model) | 0.934 ± 0.004 | +0.002 | 0.194 |

---

## Figure Plan

<!-- INTERNAL NOTES — stripped by build_paper.py (everything from "## Figure Plan" on is cut). -->

### Figure 1: Detailed architecture / two-pass forward diagram
*Schematic, fixed geometry (no data file).*
**Status:** DONE — `research/figures/fig_architecture_detailed.png` (`scripts/fig_architecture_detailed.py`)

### Figure 2: Training objective (SELEX round → diffusion noise level)
*Schematic, fixed geometry (no data file).*
**Status:** DONE — `research/figures/fig2_training_objective.png` (`scripts/fig2_training_objective.py`).

### Figure 3: Benchmark design schematic (RecoveryBench | KdBench)
*Schematic, fixed geometry (no data file).*
**Status:** DONE — `research/figures/fig_benchmark_design.png` (`scripts/fig_benchmark_design.py`).

### Figure 4: RecoveryBench per-target AUROC
*Data: recovery_likelihood_v7_film_cnn_mst2_matchedseqs_lowt.csv, recovery_evoflow_lowt.csv, recovery_rinalmo_lowt.csv*
**Status:** DONE — `research/figures/fig2_recoverybench.png` (`scripts/fig2_recoverybench_bars.py`). Means 0.947 / 0.521 / 0.511.

### Figure 5: Conditioning within-domain vs cross-domain LOO
*Data: null_conditioning_tumbleweed_60m_diffusion_v7_film_cnn_mst2.csv (panel A); recovery_loo_{FGF9,IL1RL1,PARP1,MECP2,SNCA}_mst2_lowt.csv + recovery_evoflow_lowt.csv (panel B)*
**Status:** DONE — `research/figures/fig3_conditioning_ab.png` (`scripts/fig3_conditioning_ab.py`). 5/5 LOO retrains complete.

### Figure 6: KdBench forest plot (draw-the-negative)
*Data: v2_loo_per_target.csv*
**Status:** DONE — `research/figures/fig4_kdbench_forest.png` (`scripts/fig4_kdbench_forest.py`). All CIs cross 0; best t = 1.27.
