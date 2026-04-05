"""Compare raw MLM quality: base vs fine-tuned model on Sinhala sentences."""
import sys, os, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sinhala sentences for perplexity measurement (natural Sinhala, not transliterated)
sinhala_sentences = [
    "මම හෙට පාසලට යනවා",
    "ඔයා කොහේද යන්නේ",
    "අපි එකට වැඩ කරමු",
    "මට ඒක තේරුණේ නෑ",
    "ගුරුවරයා පාඩම කිව්වා",
    "කාලය ඉතුරු කරගන්න ඕනි",
    "මම පොත කියවලා ඉවර කළා",
    "ඔයා ආවා වගේ මට හිතෙනවා",
    "අපේ වැඩ අද ඉවර වෙනවා",
    "ප්‍රශ්නය හොඳ වගේ පේනවා",
    "මම දන්නෙ නෑ ඒක ගැන",
    "ඔයා කිව්වට මම ගියේ",
    "හෙට පරීක්ෂණය තියෙනවා",
    "අපි පස්සෙ හම්බවෙමු",
    "මේ වැඩ හොඳ වගේ පේනවා",
]

def compute_pseudo_perplexity(model, tokenizer, sentences):
    """Compute pseudo-perplexity using masked token prediction."""
    model.eval()
    total_log_prob = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
            input_ids = inputs["input_ids"][0]
            
            # Skip special tokens
            non_special = [i for i in range(len(input_ids)) 
                          if input_ids[i] not in [tokenizer.bos_token_id, tokenizer.eos_token_id, 
                                                   tokenizer.pad_token_id, tokenizer.cls_token_id,
                                                   tokenizer.sep_token_id]]
            
            for idx in non_special:
                masked = input_ids.clone().unsqueeze(0)
                original_id = masked[0, idx].item()
                masked[0, idx] = tokenizer.mask_token_id
                
                outputs = model(masked, attention_mask=inputs["attention_mask"])
                logits = outputs.logits[0, idx]
                log_probs = torch.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[original_id].item()
                total_tokens += 1
    
    avg_nll = -total_log_prob / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_tokens

models = {
    "Base (xlm-roberta-base)": "FacebookAI/xlm-roberta-base",
    "Fine-tuned (v2)": os.path.join(os.path.dirname(__file__), "..", "xlm-roberta-sinhala-v2", "final"),
}

print("=" * 60)
print("  MLM Pseudo-Perplexity Comparison on Sinhala Text")
print("=" * 60)
print(f"  Test sentences: {len(sinhala_sentences)}")
print()

for name, path in models.items():
    print(f"Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path).to(device)
    
    ppl, avg_nll, n_tokens = compute_pseudo_perplexity(model, tokenizer, sinhala_sentences)
    print(f"  {name}:")
    print(f"    Pseudo-Perplexity : {ppl:.2f}")
    print(f"    Avg NLL           : {avg_nll:.4f}")
    print(f"    Tokens evaluated  : {n_tokens}")
    print()
    
    del model
    torch.cuda.empty_cache()

print("Lower perplexity = better Sinhala language understanding")
