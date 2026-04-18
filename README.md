# Triagegeist — Clinically-Safe ESI Prediction with 4-Model Stack & Conformal Sets

> **Competition:** Triagegeist · Laitinen–Fredriksson Foundation · Community Hackathon 2026
> **Metric:** Quadratic Weighted Kappa (QWK)
> **Hardware:** Kaggle T4 × 2 (32 GB VRAM)

![Thumbnail](visuals/thumbnail.png)

---

## Results

| Metric | Value |
|---|---|
| OOF QWK (5-fold) | **0.999913** |
| OOF Accuracy | **0.999812** |
| Total errors | **15 / 80,000** (0.019 %) |
| Undertriage errors | 6 |
| Overtriage errors | 9 |
| Worst subgroup QWK gap | −0.0002 |
| Fragile rows (bootstrap) | 0 |

---

## Overview

A clinically-safe 4-model ensemble for Emergency Severity Index (ESI 1–5) prediction, built around the principle that a triage assistant must expose its uncertainty — not just its point estimate.

The Emergency Severity Index is a 5-level ordinal triage scale. Under-triage — predicting low acuity when the patient is critically ill — is associated with preventable mortality. Our pipeline accounts for this asymmetry at every stage: loss function, threshold calibration, and the runtime safety layer.

---

## Pipeline

### 1 · Leakage Audit

`disposition` and `ed_los_hours` are post-triage fields — dropped at load time. Verified: the only columns in `train.csv` absent from `test.csv` are the target and these two leakage columns.

### 2 · Feature Engineering — ~150 features, 6 families

- **Physiological composites** — shock index (HR/SBP ≥ 1.0 → haemorrhage risk), MAP, pulse pressure, ROX index
- **ESI v5 threshold flags** — GCS < 9, SpO₂ < 90 %, RR > 25, SBP < 90 — each mirrors an ESI handbook decision rule
- **Age × vital interactions** — SBP of 110 mmHg means different things at age 8 vs age 80
- **MNAR missingness signals** — unrecorded BP predicts mean acuity 4.33 vs 3.27 for recorded
- **qSOFA + SIRS counts** — validated early-warning composites used by triage nurses
- **Per-site / per-nurse z-scores** — normalise documentation style across 5 clinical sites

### 3 · NLP Pipeline — 4 layers on chief-complaint text

- 50+ medical abbreviation expansions (`CP`→chest pain, `SOB`→shortness of breath, `MVC`→motor vehicle collision…)
- 14 keyword regex families (critical / shock / cardiac / neuro / sepsis / trauma / bleeding…)
- Dual TF-IDF: word 1–2-grams + char_wb 3–5-grams → TruncatedSVD (64 dims each)
- `all-MiniLM-L6-v2` sentence embeddings (384-d → 48-d via SVD) — GPU-batched on T4

### 4 · Base Models

| Model | Device | OOF QWK |
|---|---|---|
| LightGBM | CPU | 0.9997 |
| XGBoost | GPU (CUDA) | 0.9998 |
| CatBoost | GPU | 0.9997 |
| PyTorch MLP (AMP) | GPU | 0.9997 |

### 5 · Stacking

20-column OOF matrix (4 models × 5 classes) → L1 multinomial logistic regression (`penalty='l1', C=0.5, solver='saga'`). L1 zeros out redundant columns automatically.

### 6 · Ordinal Threshold Optimisation

Expected acuity = Σ k · P(ESI=k). Four thresholds learned via **Differential Evolution** (global) + **Nelder–Mead** (local, `xatol=1e-7`) — directly maximising QWK on OOF.

---

## Ablation Study

![Ablation](visuals/ablation.png)

| Variant | OOF QWK | Δ (basis pts) |
|---|---|---|
| LGBM alone (argmax) | 0.9997 | — |
| XGB alone | 0.9998 | +1.6 |
| CAT alone | 0.9997 | +0.1 |
| MLP alone | 0.9997 | +0.6 |
| Mean-blend of 4 | 0.9998 | +1.6 |
| Full L1 stack | 0.9998 | +2.2 |
| Mean-blend + DE/NM thresholds | 0.9999 | +2.0 |
| **Full stack + DE/NM (final)** | **0.9999** | **+2.5** |

---

## Clinical Safety Layer

![Safety](visuals/safety.png)

### Split Conformal Prediction
Non-conformity scores on 80k OOF patients give finite-sample marginal coverage guarantees.
Mean prediction set size: **0.985** at 95 % coverage · **0.980** at 90 % coverage.

### Asymmetric Cost Matrix
Under-triage costs **2× quadratically** more than over-triage.
QWK-optimal undertriage rate: **0.0001** (8 patients / 80 k).

### Undertriage Risk Score (URS)
```
URS = 0.50 · P(ESI ≤ 2) + 0.30 · [nurse ≠ model] + 0.20 · NEWS2/7
```
Test-set mean URS: **0.179** · max: **0.700** · ~2 % of patients flagged for senior review.

### Bootstrap Prediction Frailty
100-replicate Dirichlet-noise bootstrap. **0 fragile rows** in 20,000 test patients. Mean agreement: **0.979**.

---

## Confusion Matrix

![Confusion](visuals/confusion.png)

---

## Bias & Fairness Audit

![Fairness](visuals/fairness.png)

Subgroup QWK across sex (M/F/Other), language (8 groups), age-band (8 groups), site (5 hospitals).

**Worst gap: −0.0002** (age_band 1–5 yr, n=1,729) — 10× inside the deployment gate of ±0.002.

---

## Feature Importance

![Features](visuals/features.png)

---

## Failure Case Analysis

**15 total OOF errors** (9 overtriage · 6 undertriage) out of 80,000 patients.

The dominant failure pattern: **"acute angle closure glaucoma"** — a condition at the ESI-1/2 boundary depending on haemodynamic presentation. These are genuinely ambiguous cases even for experienced triage nurses, and the model correctly reflects this uncertainty in high URS scores.

Top feature drifts in misclassified vs correct cases:

| Feature | Error mean | Correct mean | Δ |
|---|---|---|---|
| sbp_x_age | 4,339 | 5,969 | −1,630 |
| news2_score | 12.4 | 3.4 | +8.9 |
| systolic_bp | 89.6 mmHg | 121.6 mmHg | −32 |

---

## Model Card

| | |
|---|---|
| **Intended use** | Decision-support for ED triage nurses. Output: point estimate + conformal set + URS + bootstrap agreement |
| **Out of scope** | Autonomous triage · neonates (< 1 yr) · real clinical deployment without prospective validation |
| **Training data** | 80,000 synthetic patients · 5 sites · strictly pre-triage features |
| **Monitoring** | Weekly subgroup-QWK audit · monthly conformal coverage check · continuous URS distribution monitoring |

---

## Repository Structure

```
triagegeist/
├── triagegeist_final.ipynb      # Full pipeline notebook (15 sections)
├── visuals/                     # Charts for writeup and README
│   ├── thumbnail.png
│   ├── ablation.png
│   ├── confusion.png
│   ├── fairness.png
│   ├── features.png
│   └── safety.png
├── outputs/
│   ├── submission.csv           # Final predictions (20,000 patients)
│   └── submission_supplementary.csv  # Per-patient probabilities, URS, conformal sets
└── README.md
```

---

## Authors

Abdurakhmonov Dostonbek · Akhror Bukhorov
