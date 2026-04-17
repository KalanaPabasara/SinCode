---
title: SinCode
emoji: 🔤
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# à·ƒà·’à¶‚Code â€” Singlish to Sinhala Transliterator

A model-driven, context-aware back-transliteration system that converts Romanised Sinhala (Singlish) to native Sinhala script.

## Architecture (v3)

```
Input sentence
    â”‚
    â–¼
Word Tokenizer
    â”‚
    â”œâ”€ Sinhala script? â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Pass through unchanged
    â”‚
    â”œâ”€ English vocab (len â‰¥ 3)? â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Pass through unchanged
    â”‚
    â””â”€ Singlish word?
            â”‚
            â–¼
     ByT5-small seq2seq
     (top-5 candidates)
            â”‚
            â–¼
     XLM-RoBERTa MLM reranker
     (contextual scoring)
            â”‚
            â–¼
      Best candidate
```

## Models

| Model | Role | Hub ID |
|-------|------|--------|
| ByT5-small | Singlish â†’ Sinhala candidate generation | `Kalana001/byt5-small-singlish-sinhala` |
| XLM-RoBERTa | Contextual MLM reranking | `Kalana001/xlm-roberta-base-finetuned-sinhala` |
| mBart50 | Full-sentence Sinhala output mode | `Kalana001/mbart50-large-singlish-sinhala` |

## Modes

- **Code-Mixed Output** â€” Retains English words where contextually appropriate; Singlish words are transliterated using ByT5 + XLM-RoBERTa reranking.
- **Full Sinhala Output** â€” Transliterates the entire sentence to Sinhala script using mBart50.

## Environment Variables (optional)

Set these in HF Spaces â†’ Settings â†’ Repository secrets to enable Supabase feedback storage:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `SUPABASE_FEEDBACK_TABLE` | Table name (default: `feedback_submissions`) |

If not set, feedback is saved locally to `misc/feedback_submissions.jsonl`.

