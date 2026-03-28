---
title: SinCode
emoji: 💻
colorFrom: indigo
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
license: mit
short_description: Context-Aware Transliteration
sdk_version: 1.53.1
---

# SinCode: Neuro-Symbolic Transliteration Prototype

> **Context-Aware Singlish-to-Sinhala Transliteration with Code-Switching Support.**

Welcome to the interim prototype of **SinCode**, a final-year research project designed to solve the ambiguity of transliterating "Singlish" (phonetic Sinhala) into native Sinhala script.

## 🚀 Key Features

* **🧠 Hybrid Neuro-Symbolic Engine:** Combines the speed of rule-based logic with the contextual understanding of Deep Learning (XLM-Roberta).
* **🔀 Adaptive Code-Switching:** Intelligently detects English words (e.g., *"Assignment"*, *"Presentation"*) mixed within Sinhala sentences and preserves them automatically.
* **📚 Massive Vocabulary:** Powered by an optimized dictionary of **5.9 Million** Sinhala words to ensure high-accuracy suggestions.
* **⚡ Contextual Disambiguation:** Resolves ambiguous terms (e.g., detecting if *"nisa"* means *because* or *near*) based on the full sentence context.

## 🛠️ How to Use

1.  **Type** your Singlish sentence in the input box.
2.  Click the **Transliterate** button.
3.  View the **Result**.
4.  (Optional) Expand the **"See How It Works"** section to view the real-time scoring logic used by the system.

## 📏 Baseline Evaluation (New)

Use the evaluation script to measure current model quality before making tuning changes.

### 1) Prepare dataset

Create a CSV file with columns:

- `input` (Singlish / code-mixed input)
- `reference` (expected Sinhala output)

You can start from `eval_dataset_template.csv`.

### 2) Run evaluation

```bash
python evaluation.py --dataset eval_dataset_template.csv
```

Optional:

```bash
python evaluation.py --dataset your_dataset.csv --beam-width 5 --predictions-out eval_predictions.csv --diagnostics-out eval_diagnostics.json
```

### 3) Outputs

- `eval_predictions.csv`: per-sample prediction + metrics
- `eval_diagnostics.json`: per-word candidate scoring breakdown for error analysis

Reported aggregate metrics:

- Exact match
- Average Character Error Rate (CER)
- Average token accuracy
- Average English code-mix preservation

## 🤗 Hugging Face Spaces Notes

This project is compatible with Spaces. You can configure runtime paths with environment variables:

- `SICODE_DICTIONARY_PATH` (default: `dictionary.pkl`)
- `SICODE_MODEL_NAME` (default: `FacebookAI/xlm-roberta-base`)
- `SICODE_ENGLISH_CACHE` (optional path for `english_20k.txt` cache)

Example:

```bash
SICODE_DICTIONARY_PATH=dictionary.pkl
SICODE_MODEL_NAME=FacebookAI/xlm-roberta-base
```

The engine now auto-selects a writable cache path for English corpus downloads when running in restricted environments.

## 🏗️ System Architecture

This prototype utilizes a **Tiered Decoding Strategy**:
1.  **Tier 1 (English Filter):** Checks the Google-20k English Corpus to filter out technical terms.
2.  **Tier 2 (Dictionary Lookup):** Scans the 5.9M word database for exact Sinhala matches.
3.  **Tier 3 (Phonetic Rules):** Generates Sinhala text for unknown words using a rule-based engine.
4.  **Tier 4 (Neural Ranking):** The **XLM-R** model scores all possible candidates to pick the most grammatically correct sequence.

## ⚠️ Disclaimer

This is an **Interim Prototype** for demonstration purposes.
* While accurate for common phrases, edge cases may still exist.
* The system is currently optimized for demonstration performance and will be fine-tuned further.

---
**Developer:** Kalana Chandrasekara

**Supervisor:** Hiruni Samarage


*Final Year Research Project (2026)*
