# Center‑of‑Pressure‑Constrained GRF Estimation

> **Note:** Code will be uploaded soon.

---

## 🌟 Highlights

-   **CoP-Constrained Learning** We introduce a **CoP Limiter**, a simple layer that guarantees physically valid predictions. It constrains the predicted Center of Pressure (CoP) to remain within a **Virtual Foot Boundary (VFB)** defined by anatomical landmarks.
-   **Improved Accuracy & Generalization**: Reduces CoP prediction error by **up to 60.7%** on validation data and demonstrates robust cross-domain generalization in zero-shot tests, improving GRF error by **up to 6.5%** without any fine-tuning.

---

## 📝 Summary

Ground Reaction Force (GRF) is an essential element for dynamics simulation in human motion analysis. The gold-standard for its measurement, the force plate, is **expensive, lab-bound, and can disrupt natural motion**. While recent learning-based methods have been proposed to predict GRF from motion data alone, they often **estimate physically implausible forces for unseen data** (BL), such as predicting forces originating outside the foot.

This study introduces the **CoP Limiter** (CL), a novel layer designed to guarantee physical plausibility. The layer constrains the predicted CoP to lie **within a subject-specific Virtual Foot Boundary (VFB)** defined from anatomical landmarks without changing the base model architecture.

-   **Training dataset** [AddBiomechanics Core](https://addbiomechanics.org/download_data.html) — 24 M frames, 273 subjects, 70 h motion‑capture + kinetics (accessed **2024‑09‑01**).
-   **Validation set** Hold‑out split from the AddBiomechanics Core Dataset.
-   **Zero‑shot test** 29 older adults performing four ADLs (5‑Times Chair Stand, Chair Stand, Pick‑Up, Step‑Up) — _no fine‑tuning_.
-   **Encoders** Feed‑Forward (FFN), Mamba (SSM), Transformer (Attention) — 117 M parameters each, window = 100 frames.

---

## 🔍 Results

### Quantitative

| Cohort         | Encoder     | Baseline CoP ↓ | **+ CoP‑Limiter** | Baseline GRF ↓ | **+ CoP‑Limiter** |
| -------------- | ----------- | -------------- | ----------------- | -------------- | ----------------- |
| **Validation** | FFN         | 0.056          | **0.022**         | 0.611          | **0.615**         |
|                | Mamba       | 0.034          | **0.019**         | 0.345          | **0.345**         |
|                | Transformer | 0.034          | **0.020**         | 0.328          | **0.326**         |
| **Zero‑shot**  | FFN         | 0.075          | **0.038**         | 1.119          | **1.121**         |
|                | Mamba       | 0.057          | **0.033**         | 0.844          | **0.806**         |
|                | Transformer | 0.057          | **0.034**         | 0.804          | **0.752**         |

_Metrics_: CoP error in meters (m), mass-normalized GRF error in N/kg. Lower is better.

### Qualitative (Baseline vs CoP‑Limiter)

<img src="docs/estimation_result.gif" width="100%">

> Red = force‑plate GRF, Blue = estimate, Green box = VFB. **BL** = baseline; **CL** = with CoP‑Limiter.

---

## 📄 Citation

Formal citation forthcoming — manuscript in preparation.

---

_Last updated : 2025‑06‑24_
