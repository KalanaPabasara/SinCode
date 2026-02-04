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

# SinCode: Neuro-Symbolic Transliteration Prototype 🇱🇰

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
