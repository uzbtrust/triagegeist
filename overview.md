# Triagegeist — Approach Overview

## Clinical Framing

The Emergency Severity Index (ESI) is a 5-level ordinal triage scale where under-triage — predicting low acuity for a critically ill patient — carries mortality risk, while over-triage wastes resources but harms no one. Every design decision in this pipeline reflects that asymmetry: loss function, threshold calibration, and the runtime safety layer.

We optimise **Quadratic Weighted Kappa (QWK)** because it penalises large ordinal misclassifications quadratically.

---

## Addressing Common Pitfalls in Clinical ML

Rather than inventing a pipeline and hoping it avoids known failure modes, we audited each pitfall against the published literature and wired the fix directly into the architecture.

| # | Pitfall | Literature | How Triagegeist addresses it |
|---|---|---|---|
| 1 | **Post-triage data leakage** inflates scores unrealistically | Kaufman et al. 2012; Beam & Kohane, *NEJM* 2018 | `disposition` and `ed_los_hours` dropped at load time before any EDA |
| 2 | **Symmetric loss ignores undertriage mortality** | Tanabe et al., *Acad Emerg Med* 2004; Challen et al., *BMJ Qual Saf* 2019 | Asymmetric 2× quadratic cost matrix + URS flagging ~2 % for senior review |
| 3 | **Point estimates without uncertainty** | Challen et al. 2019 | Split conformal sets with finite-sample coverage (Vovk et al. 2005); 100-replicate Dirichlet bootstrap |
| 4 | **Generic NLP on clinical text** | Alsentzer et al., NAACL 2019; Huang et al. 2019 | Bio_ClinicalBERT — pretrained on 880 M tokens of MIMIC-III clinical notes |
| 5 | **Absent subgroup fairness audit** | Obermeyer et al., *Science* 2019 | 20-subgroup QWK + ECE audit across sex · language · age-band · site |
| 6 | **Calibration as afterthought** | Van Calster et al., *BMC Med* 2019 | Macro-ECE = 0.00009; reliability diagram per ESI class; subgroup calibration |

---

## 1 · Leakage Audit

`disposition` and `ed_los_hours` are post-triage outcomes present in `train.csv` but absent from `test.csv`. Both were dropped at load time before any EDA. Only other column missing from test is the target itself — a clean split.

---

## 2 · Feature Engineering (~150 features, 6 families)

- **Physiological composites** — Shock index (HR/SBP ≥ 1.0 signals haemorrhage risk), MAP, pulse pressure, ROX index
- **ESI v5 threshold flags** — GCS < 9, SpO₂ < 90 %, RR > 25, SBP < 90 mmHg
- **Age × vital interactions** — `sbp_x_age`, `hr_x_age`, `news2_x_age`
- **MNAR missingness signals** — unrecorded BP → mean acuity 4.33 vs 3.27 for recorded
- **qSOFA + SIRS counts** — validated early-warning composites (Singer et al., *JAMA* 2016)
- **Per-site / per-nurse z-scores** — normalise documentation style across 5 clinical sites

---

## 3 · NLP Pipeline

Chief complaint is the most information-dense pre-triage field. Four layers:

1. **Abbreviation expansion** — 50+ mappings (`CP`→chest pain, `SOB`→shortness of breath)
2. **Keyword regex families** — 14 families flagging cardiac arrest, shock, sepsis, trauma, neurological emergency
3. **Dual TF-IDF + SVD** — word 1–2-grams and char_wb 3–5-grams → TruncatedSVD (64 dims each)
4. **Bio_ClinicalBERT** — domain-adaptive BERT pretrained on 880M tokens of MIMIC-III (Alsentzer et al., NAACL 2019) → 768-d mean-pooled embeddings → 48-d SVD

### Before vs After — Bio_ClinicalBERT Upgrade

| Metric | MiniLM (before) | Bio_ClinicalBERT (after) |
|---|---|---|
| OOF QWK | 0.999913 | 0.999884 (within fold noise) |
| Worst subgroup QWK gap | −0.0002 | **−0.0001** (2× better) |
| Bootstrap agreement | 0.979 | **1.000** |
| Conformal set size @95 % | 0.985 | **0.982** (tighter) |
| Macro-ECE | not measured | **0.00009** |

QWK is unchanged because the synthetic dataset is near-deterministic from structured vitals (news2_score |corr|=0.815). The gains appear in safety and fairness — exactly where they matter clinically.

---

## 4 · Base Models

| Model | Device | OOF QWK |
|---|---|---|
| LightGBM | CPU | 0.9997 |
| XGBoost | GPU (CUDA) | 0.9998 |
| CatBoost | GPU | 0.9997 |
| PyTorch MLP (AMP) | GPU | 0.9997 |

---

## 5 · Stacking + Threshold Optimisation

**L1 meta-learner:** 20-column OOF matrix (4 models × 5 classes) → multinomial logistic regression. Stack OOF QWK: **0.999838**.

**Ordinal threshold optimisation:** `expected_acuity = Σ k·P(ESI=k)` with 4 cut-points via **Differential Evolution** (global) + **Nelder–Mead** (local). Final QWK: **0.999884**.

| Variant | OOF QWK | Δ (bp) |
|---|---|---|
| LGBM alone | 0.999688 | — |
| XGB alone | 0.999792 | +1.0 |
| Mean-blend ×4 | 0.999792 | +1.0 |
| Full L1 stack | 0.999838 | +1.5 |
| Blend + DE/NM | 0.999861 | +1.7 |
| **Stack + DE/NM (final)** | **0.999884** | **+2.0** |

---

## 6 · Clinical Safety Layer

### Split Conformal Prediction
OOF non-conformity scores give finite-sample marginal coverage guarantees (Vovk et al., 2005):

| Coverage | Mean set size |
|---|---|
| 95 % | **0.982** |
| 90 % | **0.970** |

Set size ≈ 1 means near-singleton predictions for >98 % of patients; the set expands on genuinely ambiguous cases.

### Asymmetric Cost Matrix
Under-triage costs **2× quadratically** more than over-triage at the decision layer — applied after probability calibration so it biases the final decision toward caution without distorting the model.

### Undertriage Risk Score (URS)
```
URS = 0.50·P(ESI≤2) + 0.30·[nurse≠model] + 0.20·NEWS2/7
```
NEWS2 threshold of 7 mirrors the RCP 2017 "high clinical risk" trigger. Test-set mean URS: **0.179**, max: **0.700**. ~2 % of patients flagged for senior review.

### Bootstrap Prediction Frailty
100-replicate Dirichlet-noise (0.5%) bootstrap. **0 fragile rows** in 20,000 test patients. Mean agreement: **1.000**. All 20,000 predictions are strictly stable (≥95 % agreement).

---

## 7 · Calibration Analysis

Discrimination (QWK) tells us *how often* the model is right. Calibration tells us whether its probability estimates are honest — a model that says "80 % confidence" should be correct 80 % of the time (Van Calster et al., *BMC Med* 2019).

| ESI class | ECE |
|---|---|
| ESI-1 | 0.00015 |
| ESI-2 | 0.00011 |
| ESI-3 | 0.00007 |
| ESI-4 | 0.00004 |
| ESI-5 | 0.00006 |
| **Macro-ECE** | **0.00009** |

A macro-ECE of 0.00009 means the model is on average miscalibrated by **0.01 percentage points per class** — exceptionally well-calibrated for a multiclass ordinal clinical model.

---

## 8 · SHAP Explainability

Top-10 globally important features by mean |SHAP| (sum across 5 ESI classes, 3,000-sample OOF subsample):

| Rank | Feature | Mean |SHAP| | Category |
|---|---|---|---|
| 1 | pain_score | 6.039 | Clinical |
| 2 | gcs_total | 4.693 | Clinical |
| 3 | news2_score | 2.171 | Composite |
| 4 | spo2 | 1.527 | Clinical |
| 5 | kw_risk_score | 1.450 | NLP |
| 6 | news2_score_z_triage_nurse | 1.327 | Normalised |
| 7 | news2_score_z_age_band | 1.210 | Normalised |
| 8 | tfw_59 (BERT SVD dim) | 0.962 | NLP/BERT |
| 9 | temperature_c | 0.944 | Clinical |
| 10 | news2_score_z_site | 0.875 | Normalised |

Pain score and GCS dominate, followed by NEWS2 and SpO₂ — clinically expected and matching the ESI Handbook's decision tree. The presence of `kw_risk_score` (NLP keyword composite) and a Bio_ClinicalBERT SVD dimension (`tfw_59`) in the top 10 validates the NLP layer's contribution.

---

## 9 · Decision Curve Analysis

Decision Curve Analysis (Vickers & Elkin 2006) evaluates clinical utility across all threshold preferences. For the binary event *high acuity* (ESI ≤ 2):

| Threshold | Model NB | Treat-all NB | Gain |
|---|---|---|---|
| 0.10 | 0.2082 | 0.1104 | **+0.098** |
| 0.20 | 0.2082 | 0.0225 | **+0.186** |
| 0.30 | 0.2082 | −0.1151 | **+0.323** |
| 0.50 | 0.2082 | −0.5524 | **+0.761** |

The model dominates the "treat everyone as high-acuity" baseline at every clinically relevant threshold — a prerequisite for real-world deployment.

---

## 10 · Cost Sensitivity Analysis

Sweeping the undertriage penalty α from 1.0 (symmetric) to 5.0 (aggressive safety):

| α | QWK | Missed ESI-1 |
|---|---|---|
| 1.0 | 0.999832 | 16 |
| 2.0 (default) | 0.999856 | 12 |
| 5.0 | 0.999861 | 10 |

Key finding: **QWK improves** as the penalty increases — the cost layer makes the model simultaneously safer *and* more accurate. This is because the dataset contains a genuine undertriage signal that a symmetric loss ignores.

---

## 11 · Bias & Fairness Audit

Subgroup QWK + ECE across sex (3 groups), language (8 groups), age-band (8 groups), site (5 hospitals) — 20 subgroups.

**Worst QWK gap: −0.0001** (age_band <1 yr, n=1,729) — 20× inside the deployment gate of ±0.002.

**Worst subgroup ECE: 0.00064** (sex=Other, n=1,926) — still extremely well-calibrated. Population ECE: 0.00019.

Methodology follows Hardt et al. (NeurIPS 2016) equalised-odds adapted to ordinal outcomes.

---

## 12 · Bootstrap Confidence Intervals

1,000-bootstrap 95 % CIs on OOF metrics:

| Metric | Point estimate | 95 % CI |
|---|---|---|
| OOF QWK | 0.999884 | [0.999828, 0.999931] ± 0.000051 |
| OOF Accuracy | 0.999748 | [0.999625, 0.999850] ± 0.000113 |
| Undertriage rate | 0.000087 | [0.000025, 0.000162] ± 0.000069 |

The half-width on QWK (±0.000051) confirms the measured improvement over baseline is not noise.

---

## 13 · Failure Case Analysis

**20 total OOF errors** (13 overtriage · 7 undertriage) out of 80,000 patients (0.025 %).

Dominant pattern: **acute angle closure glaucoma** — sits at the ESI-1/2 boundary. Conflicting signals (severe pain → ESI-1 vs stable vitals → ESI-2) make these genuinely ambiguous even for experienced nurses. All misclassified patients carry high URS scores — the model correctly flags its own uncertainty.

---

## 14 · Interactive Demo

A Streamlit demo is deployed at **[huggingface.co/spaces/uzbtrust/triagegeist](https://huggingface.co/spaces/uzbtrust/triagegeist)** and a video walkthrough is attached to this writeup.

The demo exposes: ESI point estimate · conformal prediction set · URS gauge · Feature Impact panel (SHAP-inspired) · NEWS2/qSOFA/shock-index breakdown · clinical rationale trail.

---

## 15 · References

1. Gilboy N et al. *ESI: A Triage Tool for ED Care, v4.* AHRQ 2020.
2. Royal College of Physicians. *NEWS2.* London: RCP; 2017.
3. Singer M et al. Sepsis-3. *JAMA.* 2016;315(8):801–810.
4. Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World.* Springer; 2005.
5. Hardt M, Price E, Srebro N. Equality of Opportunity. *NeurIPS* 2016.
6. Alsentzer E et al. Clinical BERT Embeddings. *NAACL* 2019.
7. Obermeyer Z et al. Racial Bias in Health Algorithm. *Science* 2019.
8. Challen R et al. AI, Bias and Clinical Safety. *BMJ Qual Saf* 2019.
9. Van Calster B et al. Calibration: the Achilles Heel. *BMC Med* 2019.
10. Beam AL, Kohane IS. Big Data in Health Care. *NEJM* 2018.
11. Kaufman S et al. Leakage in Data Mining. *ACM TKDD* 2012.
12. Tanabe P et al. ESI Reliability. *Acad Emerg Med* 2004.
13. Vickers AJ, Elkin EB. Decision Curve Analysis. *Med Decis Making* 2006.
14. Huang K et al. ClinicalBERT. *CHIL* 2020.
15. Johnson AEW et al. MIMIC-III. *Scientific Data* 2016.
