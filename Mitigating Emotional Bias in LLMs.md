
**Author:** Yihao Zhang 
**Date:** October 17, 2025
**case id:** yxz3114

**Scope:** Integrates provided papers across concepts, datasets, measurement, and mitigation; then proposes a practical, end‑to‑end implementation plan  to mitigate emotional bias in LLMs.

---

## Executive Summary

Emotional bias refers to systematic, unwanted shifts in an LLM’s emotional tone or moral sentiment—e.g., defaulting to optimism, minimizing anger/disgust, or expressing emotions differently across demographic or ideological groups—independent of task-relevant factors. From the existing literature, three recurring drivers emerge: (1) **Topic confounds** in emotion datasets; (2) **Prompt and template fragility** that makes fairness metrics unstable; and (3) **Model priors** shaped by pretraining and RLHF that skew affect and moral sentiments. Effective mitigation requires: robust evaluation that withstands prompt phrasing; causal or adversarial controls for topics/personas; controllable editing/unlearning at the representation level; and steering or decoding policies that preserve task quality while constraining affect.

---

## Paper‑by‑Paper Mini‑Reviews

### 1) **Bias and Fairness in Large Language Models: A Survey** (Gallegos et al., 2024; arXiv:2309.00770) — *Concepts & Taxonomy*

This survey consolidates bias definitions for LLMs and organizes **evaluation metrics**, **datasets**, and **mitigation methods** into clean taxonomies spanning **pre‑**, **in‑**, **intra‑**, and **post‑processing** interventions. Metrics are grouped by the level of operation—embeddings, token probabilities, and generated text—clarifying how dataset structure (counterfactual templates vs. prompts) interacts with metric choice. On mitigation, the article surfaces dominant families: **data balancing/augmentation** (e.g., counterfactuals), **loss design** (fairness‑aware, adversarial, causal), **inference‑time controls** (decoding constraints, calibration), and **representation updates** (editing/unlearning). Two high‑level insights are especially relevant to emotional bias. First, **evaluation practice** strongly shapes conclusions; the same model may appear “fair” or “unfair” depending on template and metric. Second, **task‑agnostic adaptations**—instruction tuning, preference optimization, safety layers—can introduce new biases even as they remove others. The paper recommends diversified, multi‑view evaluation (cross metrics, datasets, and prompt regimes), reporting of variance/sensitivity analyses, and careful alignment between harms targeted by a dataset and the fairness objective optimized. For emotional bias specifically, the survey’s framing supports a **pipeline** where affect distributions (emotion/moral tone) are treated as quantities to be **measured, compared across groups**, and **constrained** via either training‑time objectives or post‑hoc controls, always validated under prompt and topic perturbations.

### 2) **Topic Bias in Emotion Classification** (Wegge & Klinger, 2023/24; arXiv:2312.09043) — *Causal Confounds in Emotion Data*

Wegge and Klinger identify **topic bias** as a structural confound in emotion corpora: emotions like *sadness* or *joy* are correlated with overrepresented topics (e.g., funerals), leading models to rely on topical cues rather than true affect. The paper automatically labels topics across multiple emotion datasets, showing strong **topic–emotion correlations**, and then demonstrates that standard emotion classifiers are **confounded** by topic. Crucially, it evaluates **adversarial correction via gradient reversal** to reduce topic reliance, improving generalization when topics shift. For emotional bias mitigation, the takeaway is twofold. First, **data curation** should explicitly monitor and balance topic distributions within each emotion class; second, **modeling** should include **topic‑invariant representations**, e.g., via multi‑task setups (emotion + topic with gradient reversal) or **causal regularization** that penalizes spuriously predictive topics. The work also underscores that emotional bias is not only about group stereotypes; it can be a **semantic shortcut** tied to content domains. Therefore, any robust mitigation plan must: (a) estimate and visualize topic–emotion dependency; (b) include topic‑perturbed evaluation (train on a subset of topics, test on held‑out topics); and (c) integrate adversarial or invariant learning to remove topic pathways. When combined with prompt‑robust testing (see Paper 3) and affective alignment measurement (Paper 4), the methods yield a fuller picture of **when** and **why** an LLM’s emotional tone drifts.

### 3) **On the Fragility of Template‑Based Fairness Evaluation** (Gould et al., 2022; arXiv:2210.04337) — *Measurement & Prompt Sensitivity*

This study shows that bias metrics computed on **templated counterfactual datasets** are **highly sensitive** to subtle, meaning‑preserving wording changes. Across tasks (sentiment, toxicity, NLI, MLM), small template edits flip the direction or magnitude of measured bias—e.g., a template that suggests male>female positivity in one phrasing can show the opposite after light paraphrasing. For emotional bias, this means that **“fairness by template” is brittle**: conclusions can hinge on lexical and syntactic variants rather than actual model properties. The authors recommend moving beyond single template sets toward **handcrafted diversity or large paraphrase banks**, and, when templates are used, reporting **statistical sensitivity** (variance across paraphrases) alongside mean bias scores. Practically, mitigation *evaluation* must therefore include **prompt‑robust suites**: vary wording, role descriptors, and discourse markers while keeping the same protected attribute or emotion target. A strong implementation will operationalize **confidence intervals** over many paraphrases and prefer **distributional tests** (e.g., overlap or EMD) rather than single‑point estimates. This paper’s findings justify several steps in the plan below: prompt ensembling during eval, adversarial prompt sets for red‑teaming, and registering **template uncertainty** to avoid over‑claiming debiasing success.

### 4) **Equity Evaluation Corpus (EEC)** (Kiritchenko & Mohammad, 2018; arXiv:1805.04508) — *Counterfactual Dataset for Sentiment/Emotion Bias*

EEC provides ~8.6k short sentences from **11 templates** (some with explicit emotion words) where **protected attributes** are counterfactually swapped (e.g., gendered names, “he”↔“she”). It was initially built to assess **sentiment/affect intensity** differences across groups. EEC’s value for emotional bias work is twofold. First, it offers a **controlled counterfactual setup** where group terms change while content is fixed, enabling paired tests of **score deltas** (e.g., positivity differences male vs female). Second, because templates include both **emotion‑bearing** and **neutral** sentences, one can probe whether models over‑index on **identity terms** to infer affect even when no emotion is present. However, EEC’s template nature means results are **prompt‑sensitive** (Paper 3), and coverage is limited to certain identities and English. In implementation, EEC should be used as one **component** of a broader evaluation: compute paired deltas and effect sizes; then replicate on **paraphrase‑augmented** variants; and finally triangulate with **open‑domain prompts** that elicit emotional language without fixed templates. Combining EEC with **topic‑confound splits** and **affective‑alignment** metrics creates a multi‑angle, more reliable lens on emotional bias.

### 5) **Whose Emotions and Moral Sentiments Do Language Models Reflect?** (He, Guo, Rao, Lerman, 2024; arXiv:2402.11114) — *Affective Alignment & Ideological Skew*

This paper defines **affective alignment**: how closely an LM’s **emotion and moral sentiment distributions** match those of human groups (e.g., liberals vs conservatives) on sociopolitical topics. Using human Twitter corpora and **emotion/moral‑foundation classifiers**, the authors compute **1 – Jensen–Shannon distance** between LM‑generated and human distributions. They find that most LMs, including instruction‑tuned systems, **misalign** with both groups by default, often showing **liberal‑leaning affect** on COVID‑19 topics (e.g., more *care/loyalty*, less *anger/disgust*). **Persona steering** via prompts improves alignment but **does not eliminate** systemic skew. For emotional bias mitigation, this suggests three design requirements. First, include **affect distribution matching** (ADM) as a target: if a product requires parity across communities, constrain the model so its emotion/moral distributions **do not deviate** beyond thresholds from reference corpora. Second, **persona‑aware testing** should check whether controls (steering, decoding) simply **shift** bias rather than remove it. Third, prefer **topic‑specific** alignment analysis (mask mandates vs abortion) because priors vary across domains. In practice, ADM can be implemented via **decoding penalties** (KL/JSD to target distributions), **reinforcement learning** with affect rewards, or **post‑hoc calibration**—all evaluated under prompt paraphrases and topic shifts.

### 6) **Mitigating Social Biases in Language Models through Unlearning** (2024; arXiv:2406.13551) — *Unlearning & Task‑Vector Editing*

This paper explores **parameter‑space interventions** to remove unwanted behaviors using **unlearning** and **task vectors**. The idea: learn a small **bias task** (e.g., gender‑or identity‑linked toxicity) and construct a **task vector**; then **negate/scale** it and add to the base model to cancel that behavior, with limited impact on general ability. Variants include **LoRA adapters** for surgical edits, **gradient negation** on bias exemplars, and **pruning** of bias‑salient neurons. The study emphasizes **trade‑offs**: edits can blunt bias but may reduce helpful expressivity if over‑applied; **safety alignment** layers can be stacked with vector edits. For emotional bias, a parallel approach is natural: define **emotion‑skew tasks** (e.g., excessive optimism, suppression of anger/disgust toward specific targets), collect **contrastive pairs** (desired vs over‑emotional outputs), and compute a **debias vector**. Apply **selective scaling** per domain (e.g., healthcare vs politics) and verify with **topic‑held‑out** and **prompt‑robust** tests. Combining vector edits with **affect‑aware decoding** provides a defense‑in‑depth: vectors shift the model prior; decoding policies keep generation on‑rails.

### 7) **Template‑Robust Bias Testing Across Tasks** (Gould et al., 2022; arXiv:2210.04337) — *Replicated Here for Emotional Settings*

Beyond the headline fragility, Gould et al. detail **practical template suites** across sentiment, toxicity, NLI, and MLM. For emotional bias work, these patterns inspire **evaluation harnesses**: (i) **Minimal pairs** with identity swaps across many paraphrases; (ii) **neutral vs emotion‑laden** templates to test leakage from identity → emotion; (iii) statistical tests at **template‑group** level (paired t‑tests or non‑parametrics), reporting **% of templates** that flip conclusions. Implementation should include **prompt ensembling** (5–10 paraphrases per test) with uncertainty quantification and **selection‑bias controls** (avoid cherry‑picking templates that “look good”).

### 8) **Bias and Fairness on Multimodal Emotion Detection Algorithms** (Schmitz et al., 2021/22; arXiv:2205.08383) — *Dataset Modalities & Bias Patterns*

Although focused on multimodal emotion recognition, this work is informative for text‑only LLMs. It shows that **modalities contribute unequally** to both accuracy and bias; notably, **text‑only** systems often drive most performance while showing **lower group disparities** than audio/vision. Two lessons carry over. First, **textual emotion datasets** used to probe LLMs should be checked for **demographic balance** in any accompanying non‑text signals (e.g., speaker metadata) if such fields are used at all. Second, when building **affect‑alignment references** (human corpora), **modality‑mix effects** should be considered: if the product will interface with multimodal user inputs, it is safer to derive text references that are **robust to modality‑induced skew**. The paper strengthens the case for **clear operational definitions** (what counts as emotional bias?) and **clean evaluation partitions** (by gender, age, etc.) with parity metrics like statistical parity differences over emotion labels.

### 9) **Balancing the Scales: Enhancing Fairness in Facial Expression Recognition with Latent Alignment** (2024; arXiv:2410.19444) — *Cross‑Modal Fairness Insight*

This paper targets **facial expression recognition** (FER) and proposes **latent alignment** to reduce group disparities without sacrificing performance. While the domain is visual, the **principle** generalizes: align latent representations across groups via an auxiliary objective, thereby reducing **downstream affect classification gaps**. For LLMs and emotional bias, the analogues are: **(a)** adversarial training for **group‑invariant** affect representations, **(b)** **distribution matching** of affect logits across groups under matched topics/prompts, and **(c)** **regularizers** that penalize between‑group **Wasserstein/JSD** distances for emotion distributions conditioned on content. The key is to avoid over‑smoothing (removing legitimate, content‑driven affect) by conditioning on **topic and stance**; alignment should operate on the **residual** once these are controlled.

---

## Synthesis: What “Emotional Bias” Looks Like in LLMs

1. **Skewed affect priors:** default generations over‑use some emotions (e.g., optimism), under‑use others (anger/disgust), or prefer particular moral foundations.
2. **Group‑conditioned disparities:** affect shifts with **demographic terms** (gender/race), **ideology/persona**, or **dialect**, independent of content.
3. **Topic confounds:** emotions predicted from topical cues rather than genuine affect.
4. **Prompt fragility:** fairness conclusions change under paraphrase; mitigation successes can vanish with rewording.

---

### A. Measurement & Guardrails 

* **A1. Affective Alignment (ADM):** For selected domains (e.g., health, civic), collect reference human corpora with **emotion** and **moral‑foundation** distributions. Compute **S = 1 – JSD** between LM‑generated and human distributions per topic and persona. Track **ΔS** before/after interventions.
* **A2. Counterfactual Group Parity:** Use **EEC‑style** minimal pairs (identity swaps) for emotion/sentiment prompts. Report **paired deltas** (e.g., Δ positivity) with **CIs** across **10 paraphrases** per template.
* **A3. Topic‑Invariance Probes:** Train/test splits that **hold out topics** within each emotion. Evaluate macro‑F1 and **residual topic predictability** from hidden states.
* **A4. Prompt Robustness:** For every metric, compute **mean ± CI over paraphrase ensembles**; include a **flip‑rate** (% of templates changing verdicts).

### B. Training‑Time Mitigations

* **B1. Topic‑Adversarial Learning:** Multi‑task: primary loss = emotion (or task); auxiliary **topic classifier** with **gradient reversal** to enforce topic‑invariant features.
* **B2. Group‑Invariant Regularization (optional):** Add **distribution matching** (e.g., JSD or MMD) over **emotion logits** across demographic slices **conditioned on topic/stance**.
* **B3. Preference Optimization with Affect Constraints:** During instruction tuning (SFT → DPO/RLHF), add **penalties** if sampled outputs deviate from **target affect distributions** or show **group deltas** beyond thresholds. Use **dual‑objective**: helpfulness + affect‑fairness.

### C. Representation‑Level Edits (surgical, low‑risk)

* **C1. Task‑Vector Debiasing:** Learn a small “over‑emotionality” task (e.g., prompts that induce excessive *optimism* or suppress *anger* toward protected targets). Compute the **task vector** and **subtract/scale** it (possibly via **LoRA**) from the base model. Tune scaling on a validation set that monitors ADM and utility.
* **C2. Unlearning on Contrastive Pairs:** Build pairs of (biased, desired) generations; perform **gradient negation** on biased pairs with **KL‑stability** regularizers to preserve fluency/knowledge.

### D. Inference‑Time Controls (cheap, reversible)

* **D1. Affect‑Aware Decoding:** Add a **running penalty** to the log‑prob if predicted next tokens shift the rolling **emotion distribution** away from a topic‑specific target band. Periodically **re‑normalize** to avoid degenerate repetition.
* **D2. Persona‑Steered Guardrails:** Permit explicit persona steering **only** when user‑requested; otherwise apply **neutral‑affect priors**. Log persona requests for audit (privacy‑safe).
* **D3. Calibration & Post‑Edit:** After generation, apply a **calibration layer** that rewrites overly emotional spans (e.g., replaces intensifiers, tones down moralizing phrasing) using a constrained rewriter, then re‑check ADM and EEC deltas.

### E. Evaluation Protocol 

1. **Pre‑Mitigation Baseline:** Report ADM per topic/persona; EEC deltas (mean±CI); topic‑held‑out F1; prompt flip‑rates; utility metrics (BLEU/ROUGE or task success).
2. **Ablations:** (i) B1 only; (ii) B1+B3; (iii) B1+B3+C1; (iv) +D1. For each, plot **Pareto curves** (utility vs fairness) and **regression to mean** checks.
3. **Stress Tests:** Red‑team prompts for sarcasm, negation, and subtle identity mentions; **out‑of‑domain topics**; multi‑turn persona shifts.
4. **Safety Checkbacks:** Ensure interventions do **not** suppress legitimate expressions of harm/anger in safety‑critical contexts (e.g., abuse disclosures). Use content‑policy tests with **manual QA**.

### F. Data & Tooling

* **Reference corpora:** Curate balanced topic sets per domain. Annotate with emotion + moral foundations (weak supervision acceptable but calibrate).
* **Templates & paraphrases:** Maintain a prompt bank; auto‑paraphrase then **human filter** for semantics.
* **Open‑source components:** Emotion classifiers (e.g., SpanEmo), moral foundation taggers, JSD/MMD utilities, gradient‑reversal heads.

### G. Governance & Monitoring

* **Continuous telemetry:** Track ADM and EEC deltas over time; alert on drift.
* **Disclosure:** Document residual biases, evaluation variance, and known limitations.
* **Opt‑outs:** Provide user‑side controls to **request neutral tone** or **preferred tone** transparently.

---

## Expected Trade‑offs & Limitations

* **Utility vs neutrality:** Excessive debiasing can flatten affect and harm user experience; tune to application needs.
* **Label noise in affect classifiers:** Emotion/moral labels are imperfect; use multiple taggers and bootstrap CIs.
* **Prompt variance remains:** Even with robust suites, some fragility persists; always report uncertainty and flip‑rates.

---

## References

* Gallegos et al. *Bias and Fairness in Large Language Models: A Survey.* arXiv:2309.00770.
* Wegge & Klinger. *Topic Bias in Emotion Classification.* arXiv:2312.09043.
* Gould et al. *On the Fragility of Template‑Based Fairness Evaluation.* arXiv:2210.04337.
* Kiritchenko & Mohammad. *The Equity Evaluation Corpus (EEC).* arXiv:1805.04508.
* He, Guo, Rao, Lerman. *Whose Emotions and Moral Sentiments Do Language Models Reflect?* arXiv:2402.11114.
* *Mitigating Social Biases in Language Models through Unlearning.* arXiv:2406.13551.
* Schmitz et al. *Bias and Fairness on Multimodal Emotion Detection Algorithms.* arXiv:2205.08383.
* *Balancing the Scales: Enhancing Fairness in Facial Expression Recognition with Latent Alignment.* arXiv:2410.19444.


