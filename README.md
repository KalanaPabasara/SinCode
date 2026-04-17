---
title: SinCode
emoji: 🇱🇰
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# සිංCode — Singlish to Sinhala Transliterator

A model-driven, context-aware back-transliteration system that converts Romanised Sinhala (Singlish) to native Sinhala script.

## Architecture (v3)

```
Input sentence
    │
    ▼
Word Tokenizer
    │
    ├─ Sinhala script? ──────────────────────► Pass through unchanged
    │
    ├─ English vocab (len ≥ 3)? ─────────────► Pass through unchanged
    │
    └─ Singlish word?
            │
            ▼
     ByT5-small seq2seq
     (top-5 candidates)
            │
            ▼
     XLM-RoBERTa MLM reranker
     (contextual scoring)
            │
            ▼
      Best candidate
```

## Models

| Model | Role | Hub ID |
|-------|------|--------|
| ByT5-small | Singlish → Sinhala candidate generation | `Kalana001/byt5-small-singlish-sinhala` |
| XLM-RoBERTa | Contextual MLM reranking | `Kalana001/xlm-roberta-base-finetuned-sinhala` |
| mBart50 | Full-sentence Sinhala output mode | `Kalana001/mbart50-large-singlish-sinhala` |

## Modes

- **Code-Mixed Output** — Retains English words where contextually appropriate; Singlish words are transliterated using ByT5 + XLM-RoBERTa reranking.
- **Full Sinhala Output** — Transliterates the entire sentence to Sinhala script using mBart50.

## Environment Variables (optional)

Set these in HF Spaces → Settings → Repository secrets to enable Supabase feedback storage:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `SUPABASE_FEEDBACK_TABLE` | Table name (default: `feedback_submissions`) |

If not set, feedback is saved locally to `misc/feedback_submissions.jsonl`.
