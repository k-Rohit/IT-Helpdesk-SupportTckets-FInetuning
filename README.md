# 🤖 IT Helpdesk AI Assistant — Domain-Specific LLM Fine-Tuning

A complete fine-tuning pipeline that adapts **Qwen2.5-1.5B** into a domain-specific IT Helpdesk
Assistant, using **Continued Pretraining (CPT) → Supervised Fine-Tuning (SFT) → DPO Preference
Alignment**, built with [Unsloth](https://github.com/unslothai/unsloth) on free-tier GPUs
(Google Colab / Kaggle T4).

---

## Business Problem

Internal IT support teams spend significant time answering repetitive tickets — network issues,
account access, hardware problems, billing-adjacent IT queries. This project builds an assistant
that understands IT-support terminology, ticket structure, and response tone, and gives more
specific, on-topic answers than a generic base model — even on questions not seen during training.

---

## Pipeline

```
Base Model (Qwen2.5-1.5B)
        │
        ▼
Stage 1: Non-Instruction Fine-Tuning (CPT)
   Raw ticket text → domain vocabulary & structure
        │
        ▼
Stage 2: Instruction Fine-Tuning (SFT)
   Question → Answer pairs → response behavior
        │
        ▼
Stage 3: DPO Preference Alignment
   Chosen vs Rejected pairs → refined preferences
        │
        ▼
Final Domain-Specific IT Helpdesk Assistant
```

---

## Dataset

**Source:** [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
(HuggingFace) — 61.8k IT support tickets (English + German), with `subject`, `body`, `answer`,
`type`, `queue`, `priority`, and `tags`.

---

## 🤗 HuggingFace Links

### Datasets (derived from the source data)

| Stage | Dataset |
|---|---|
| Non-Instruction (CPT) | [kumarrohit1707/it-helpdesk-cpt-data](https://huggingface.co/datasets/kumarrohit1707/it-helpdesk-cpt-data) |
| Instruction (SFT) | [kumarrohit1707/it-helpdesk-sft-data](https://huggingface.co/datasets/kumarrohit1707/it-helpdesk-sft-data) |
| Preference (DPO) | [kumarrohit1707/it-helpdesk-dpo-data](https://huggingface.co/datasets/kumarrohit1707/it-helpdesk-dpo-data) |

### Model Adapters

| Stage | Adapter |
|---|---|
| Stage 1: CPT | [kumarrohit1707/it-helpdesk-cpt-adapter](https://huggingface.co/kumarrohit1707/it-helpdesk-cpt-adapter) |
| Stage 2: SFT | [kumarrohit1707/it-helpdesk-sft-adapter](https://huggingface.co/kumarrohit1707/it-helpdesk-sft-adapter) |
| Stage 3: DPO | [kumarrohit1707/it-helpdesk-dpo-adapter](https://huggingface.co/kumarrohit1707/it-helpdesk-dpo-adapter) |

**Base model:** [unsloth/Qwen2.5-1.5B](https://huggingface.co/unsloth/Qwen2.5-1.5B)

> **Note:** The final `inference.py` script uses the **SFT adapter** rather than the DPO adapter
> as the primary model. While DPO improved response directness on well-represented ticket
> categories, evaluation surfaced inconsistent generation-stopping behavior and category-specific
> regressions (see [Limitations](#limitations--future-work) below) — SFT was the more reliable
> checkpoint for general use. Full reasoning is in `reports/fine_tuning_explanation.md`.

---

## Repository Structure

```
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
│
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
│
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
│
├── inference.py
└── README.md
```

---

## Training Summary

| Stage | Data Size | Epochs | Learning Rate | Final Loss |
|---|---|---|---|---|
| CPT | 1,000 documents | 3 | 5e-5 (embed: 5e-6) | 1.61 |
| SFT | 2,000 examples | 2–3 | 2e-4 | ~1.2 |
| DPO | ~80 preference pairs | 3 | 5e-6 | 0.21 (rewards/accuracy ~0.95–1.0) |

All stages use LoRA (QLoRA for CPT/SFT) on a single free-tier T4 GPU via Unsloth.

---

## Key Findings

- **CPT → SFT → DPO chaining**: each stage's adapter was merged into the base weights
  (`merge_and_unload()`) before applying a fresh LoRA config for the next stage, since target
  modules differ between stages (CPT includes `embed_tokens`/`lm_head`, SFT/DPO do not).
- **Base model** frequently hallucinated unrelated scenarios, leaked training artifacts
  ("Answer according to:"), or never stopped generation cleanly.
- **SFT** reliably adopted a professional support-ticket tone and stopped cleanly, but leaned
  heavily on asking clarifying questions — a direct reflection of the source data's own patterns.
- **DPO** improved directness/helpfulness on 6 of 10 evaluation questions but introduced
  regressions on underrepresented categories (security/compliance, niche hardware pairing),
  attributed to limited category coverage in the preference dataset after quality filtering.

Full before/after comparisons are in `reports/final_evaluation.md`.

---

## Limitations & Future Work

- Source data sampling was queue-agnostic; ~26% of the full dataset falls outside strictly
  IT-relevant categories (Sales, HR, Returns, General Inquiry). A future iteration would apply
  `queue`-based filtering consistently across all three stages, not just DPO.
- Security/compliance-related tickets remained a comparatively weaker category throughout the
  pipeline, likely due to underrepresentation in the source data.
- The DPO preference dataset (~80 pairs after filtering) is on the smaller end; a larger, more
  evenly-distributed preference set would likely resolve the category-specific regressions
  observed in final evaluation.
- Next step: combine the fine-tuned model with RAG over a verified internal knowledge base for
  production-grade factual grounding, since fine-tuning alone cannot guarantee accuracy on
  questions entirely outside the training distribution.

---

## Running Inference

```bash
python inference.py
```

Or import directly:
```python
from inference import ask_question
answer = ask_question("I am not able to connect to my wifi on my macbook")
```

---

## Tech Stack

Unsloth · TRL (SFTTrainer, DPOTrainer) · Transformers · PEFT (LoRA/QLoRA) · HuggingFace Hub ·
Qwen2.5-1.5B · Google Colab / Kaggle (T4 GPU)