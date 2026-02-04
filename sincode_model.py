import torch
import math
import re
import os
import requests
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- 0. SETUP ROBUST ENGLISH VOCAB ---
def load_english_corpus():
    # 1. Define Core "Safety" Words
    core_english = {
        "transliteration", "sincode", "prototype", "assignment", "singlish",
        "rest", "complete", "tutorial", "small", "mistakes", "game", "play",
        "type", "test", "online", "code", "mixing", "project", "demo", "today",
        "tomorrow", "presentation", "slide"
    }

    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
    file_path = "english_20k.txt"

    download_success = False

    # 2. Try to Load/Download 20k Corpus
    if not os.path.exists(file_path):
        try:
            print("🌐 Downloading English Corpus...")
            r = requests.get(url, timeout=5)
            with open(file_path, "wb") as f:
                f.write(r.content)
            download_success = True
        except:
            print("Internet Warning: Could not download English corpus. Using fallback list.")
    else:
        download_success = True

    # 3. Combine Lists
    full_vocab = core_english.copy()

    if download_success and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                downloaded_words = set(f.read().splitlines())
                full_vocab.update(downloaded_words)
        except:
            pass

    print(f"English Vocab Loaded: {len(full_vocab)} words")
    return full_vocab

ENGLISH_VOCAB = load_english_corpus()

# --- 1. RULE BASED ENGINE ---
# (Standard Rule Variables)
nVowels = 26
consonants = ["nnd", "nndh", "nng", "th", "dh", "gh", "ch", "ph", "bh", "jh", "sh", "GN", "KN", "Lu", "kh", "Th", "Dh", "S", "d", "c", "th", "t", "k", "D", "n", "p", "b", "m", "\\u005C" + "y", "Y", "y", "j", "l", "v", "w", "s", "h", "N", "L", "K", "G", "P", "B", "f", "g", "r"]
consonantsUni = ["ඬ", "ඳ", "ඟ", "ත", "ධ", "ඝ", "ච", "ඵ", "භ", "ඣ", "ෂ", "ඥ", "ඤ", "ළු", "ඛ", "ඨ", "ඪ", "ශ", "ද", "ච", "ත", "ට", "ක", "ඩ", "න", "ප", "බ", "ම", "‍ය", "‍ය", "ය", "ජ", "ල", "ව", "ව", "ස", "හ", "ණ", "ළ", "ඛ", "ඝ", "ඵ", "ඹ", "ෆ", "ග", "ර"]
vowels = ["oo", "o\\)", "oe", "aa", "a\\)", "Aa", "A\\)", "ae", "ii", "i\\)", "ie", "ee", "ea", "e\\)", "ei", "uu", "u\\)", "au", "\\a", "a", "A", "i", "e", "u", "o", "I"]
vowelsUni = ["ඌ", "ඕ", "ඕ", "ආ", "ආ", "ඈ", "ඈ", "ඈ", "ඊ", "ඊ", "ඊ", "ඊ", "ඒ", "ඒ", "ඒ", "ඌ", "ඌ", "ඖ", "ඇ", "අ", "ඇ", "ඉ", "එ", "උ", "ඔ", "ඓ"]
vowelModifiersUni = ["ූ", "ෝ", "ෝ", "ා", "ා", "ෑ", "ෑ", "ෑ", "ී", "ී", "ී", "ී", "ේ", "ේ", "ේ", "ූ", "ූ", "ෞ", "ැ", "", "ැ", "ි", "ෙ", "ු", "ො", "ෛ"]
specialConsonants = ["\\n", "\\h", "\\N", "\\R", "R", "\\r"]
specialConsonantsUni = ["ං", "ඃ", "ඞ", "ඍ", "ර්"+"\u200D", "ර්"+"\u200D"]
specialChar = ["ruu", "ru"]
specialCharUni = ["ෲ", "ෘ"]

def rule_based_transliterate(text):
    for i in range(len(specialConsonants)):
        text = text.replace(specialConsonants[i], specialConsonantsUni[i])
    for i in range(len(specialCharUni)):
        for j in range(len(consonants)):
            s = consonants[j] + specialChar[i]
            v = consonantsUni[j] + specialCharUni[i]
            r = s.replace(s+"/G", "")
            text = text.replace(r, v)
    for j in range(len(consonants)):
        for i in range(len(vowels)):
            s = consonants[j] + "r" + vowels[i]
            v = consonantsUni[j] + "්‍ර" + vowelModifiersUni[i]
            r = s.replace(s+"/G", "")
            text = text.replace(r, v)
        s = consonants[j] + "r"
        v = consonantsUni[j] + "්‍ර"
        r = s.replace(s+"/G", "")
        text = text.replace(r, v)
    for i in range(len(consonants)):
        for j in range(nVowels):
            s = consonants[i] + vowels[j]
            v = consonantsUni[i] + vowelModifiersUni[j]
            r = s.replace(s+"/G", "")
            text = text.replace(r, v)
    for i in range(len(consonants)):
        r = consonants[i].replace(consonants[i]+"/G", "")
        text = text.replace(r, consonantsUni[i] + "්")
    for i in range(len(vowels)):
        r = vowels[i].replace(vowels[i]+"/G", "")
        text = text.replace(r, vowelsUni[i])
    return text

# --- 2. DICTIONARY ADAPTER ---
class DictionaryAdapter:
    def __init__(self, dictionary_dict):
        self.dictionary = dictionary_dict

    def get_candidates(self, word):
        cands = []
        word_lower = word.lower()

        # 1. English Corpus Check
        if word_lower in ENGLISH_VOCAB:
            cands.append(word)

        # 2. Sinhala Dictionary Check
        if word in self.dictionary:
            cands.extend(self.dictionary[word])
        elif word_lower in self.dictionary:
            cands.extend(self.dictionary[word_lower])

        # 3. Clean & Return
        if cands:
            return list(dict.fromkeys(cands))

        # 4. Fallback: Subwords (Only if NO candidates found)
        length = len(word)
        if length > 3:
            for i in range(2, length - 1):
                part1 = word[:i]
                part2 = word[i:]
                p1_cands = self.dictionary.get(part1) or self.dictionary.get(part1.lower())
                p2_cands = self.dictionary.get(part2) or self.dictionary.get(part2.lower())

                if p1_cands and p2_cands:
                    cands1 = list(enumerate(p1_cands[:3]))
                    cands2 = list(enumerate(p2_cands[:3]))
                    for rank1, w1 in cands1:
                        for rank2, w2 in cands2:
                            cands.append(w1 + w2)

        if cands:
            return list(set(cands))
        return []

    def get_rule_output(self, word):
        return rule_based_transliterate(word)

# --- 3. BEAM SEARCH DECODER (With Enhanced Trace) ---
class BeamSearchDecoder:
    def __init__(self, model_name="FacebookAI/xlm-roberta-base", dictionary_path="dictionary.pkl", device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        import pickle
        with open(dictionary_path, "rb") as f:
            d_data = pickle.load(f)
        self.adapter = DictionaryAdapter(d_data)

    def batch_score(self, contexts, candidates):
        inputs = self.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        mask_token_id = self.tokenizer.mask_token_id
        scores = []
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        for i, target in enumerate(candidates):
            token_ids = inputs.input_ids[i]
            mask_indices = (token_ids == mask_token_id).nonzero(as_tuple=True)
            if len(mask_indices[0]) == 0:
                scores.append(-100.0); continue

            mask_pos = mask_indices[0].item()
            probs = torch.softmax(logits[i, mask_pos, :], dim=0)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)

            if not target_ids:
                scores.append(-100.0); continue

            word_score = sum([math.log(probs[tid].item() + 1e-9) for tid in target_ids])
            scores.append(word_score / len(target_ids))
        return scores

    def decode(self, sentence, beam_width=3):
        words = sentence.split()
        candidate_sets, penalties, future_context = [], [], []
        punct_pattern = re.compile(r"^(\W*)(.*?)(\W*)$")
        trace_logs = []

        for raw in words:
            match = punct_pattern.match(raw)
            prefix, core, suffix = match.groups() if match else ("", raw, "")

            if not core:
                candidate_sets.append([raw]); penalties.append([0.0]); future_context.append(raw)
                continue

            # 1. Get Candidates
            cands = self.adapter.get_candidates(core)
            rule_cand = self.adapter.get_rule_output(core)

            if not cands:
                cands = [rule_cand]
                curr_penalties = [0.0]
            else:
                curr_penalties = []
                has_english = any(c.lower() in ENGLISH_VOCAB for c in cands)

                for c in cands:
                    is_eng = c.lower() in ENGLISH_VOCAB
                    is_rule_match = (c == rule_cand)

                    if is_eng:
                        curr_penalties.append(0.0)
                    elif has_english:
                        curr_penalties.append(5.0)
                    elif is_rule_match:
                        curr_penalties.append(0.0)
                    else:
                        curr_penalties.append(2.0)

            final_cands = [prefix + c + suffix for c in cands]
            candidate_sets.append(final_cands[:6])
            penalties.append(curr_penalties[:6])
            best_idx = curr_penalties.index(min(curr_penalties))
            future_context.append(final_cands[best_idx])

        beam = [([], 0.0)]
        for t in range(len(words)):
            candidates = candidate_sets[t]
            curr_penalties = penalties[t]
            next_beam = []

            batch_ctx, batch_tgt, batch_meta = [], [], []

            for p_idx, (p_path, p_score) in enumerate(beam):
                for c_idx, cand in enumerate(candidates):
                    future = future_context[t+1:] if t+1 < len(words) else []
                    ctx = " ".join(p_path + [self.tokenizer.mask_token] + future)
                    batch_ctx.append(ctx)
                    batch_tgt.append(cand)
                    batch_meta.append((p_idx, c_idx))

            if batch_ctx:
                scores = self.batch_score(batch_ctx, batch_tgt)
                # --- TRACE LOGGING ---
                step_log = f"**Step {t+1}: {words[t]}**\n"
                for i, score in enumerate(scores):
                    p_idx, c_idx = batch_meta[i]
                    orig_path, orig_score = beam[p_idx]
                    final_score = score - curr_penalties[c_idx]
                    next_beam.append((orig_path + [batch_tgt[i]], orig_score + final_score))

                    # Add to log if score is reasonable (reduce noise)
                    if score > -25.0:
                        word = batch_tgt[i]
                        penalty = curr_penalties[c_idx]
                        step_log += f"- `{word}` (Pen: {penalty}) -> **{final_score:.2f}**\n"
                trace_logs.append(step_log)

            if not next_beam: continue
            beam = sorted(next_beam, key=lambda x: x[1], reverse=True)[:beam_width]

        final_output = " ".join(beam[0][0]) if beam else ""
        return final_output, trace_logs
