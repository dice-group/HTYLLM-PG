import os
import torch
import random
import numpy as np
import json
from datasets import load_dataset
import sentencepiece as spm
from gpt_2_multi_gpu import GPT, GPTConfig  # wherever your classes live

# ─── reproducibility ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── tokenizer ──────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/sp_model_131072.model")  # path to your .model file

# ─── model setup ─────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# match your training config!
config = GPTConfig(
    vocab_size=sp.get_piece_size(),
    block_size=1024,
    n_layer=24,   # or whatever you trained
    n_head=16,
    n_embd=1024,
)
model = GPT(config).to(device)

# load checkpoint
ckpt = torch.load("gpt2_model_step_49000.pt", map_location=device)

# Handle _orig_mod. prefix in state dict keys (from torch.compile or distributed training)
state_dict = ckpt["model_state_dict"]
if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
    # Strip _orig_mod. prefix from all keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.eval()

# ─── answer-generation fn ────────────────────────────────────────
def generate_answer(model, sp, context, question, options):
    prompt = f"{context}\n\nQuestion: {question}\nOptions:\n"
    for i,opt in enumerate(options):
        prompt += f"{chr(65+i)}. {opt}\n"
    prompt += "\nAnswer:"

    # encode + to device
    ids = sp.encode(prompt)
    x   = torch.tensor([ids], dtype=torch.long, device=device)
    
    # forward once
    with torch.no_grad():
        logits, _ = model(x)
    last_logits = logits[0, -1]               # (vocab_size,)
    
    # Only consider tokens that could produce A, B, C, D
    valid_answer_ids = []
    for ans in ["A", "B", "C", "D", "_A", "_B", "_C", "_D"]:
        # Get all token ids that could produce this answer
        possible_ids = [i for i in range(sp.get_piece_size()) 
                       if sp.IdToPiece(i).strip().upper() == ans]
        valid_answer_ids.extend(possible_ids)
    
    # Mask logits to only consider valid answers
    masked_logits = last_logits.clone()
    masked_logits[~torch.tensor([i in valid_answer_ids for i in range(len(masked_logits))], 
                               device=device)] = float('-inf')
    
    pred_id = int(torch.argmax(masked_logits))
    pred_piece = sp.IdToPiece(pred_id)       # e.g. " A" or "B"
    pred_char = pred_piece.strip().upper()   # "A", "B", etc.
    return pred_char

# ─── run eval ──────────────────────────────────────────────────
all_languages = ['acm_Arab', 'arz_Arab', 'ceb_Latn', 'fin_Latn', 'hin_Deva', 'ita_Latn', 'khm_Khmr', 'lvs_Latn', 'npi_Deva', 'pol_Latn', 'slv_Latn', 'swe_Latn', 'tso_Latn', 'xho_Latn', 'afr_Latn', 'asm_Beng', 'ces_Latn', 'fra_Latn', 'hin_Latn', 'jav_Latn', 'kin_Latn', 'mal_Mlym', 'npi_Latn', 'por_Latn', 'sna_Latn', 'swh_Latn', 'tur_Latn', 'yor_Latn', 'als_Latn', 'azj_Latn', 'ckb_Arab', 'fuv_Latn', 'hrv_Latn', 'jpn_Jpan', 'kir_Cyrl', 'mar_Deva', 'nso_Latn', 'snd_Arab', 'tam_Taml', 'ukr_Cyrl', 'zho_Hans', 'amh_Ethi', 'bam_Latn', 'dan_Latn', 'gaz_Latn', 'hun_Latn', 'kac_Latn', 'kor_Hang', 'mkd_Cyrl', 'nya_Latn', 'ron_Latn', 'som_Latn', 'tel_Telu', 'urd_Arab', 'zho_Hant', 'apc_Arab', 'ben_Beng', 'deu_Latn', 'grn_Latn', 'hye_Armn', 'kan_Knda', 'lao_Laoo', 'mlt_Latn', 'ory_Orya', 'rus_Cyrl', 'sot_Latn', 'tgk_Cyrl', 'urd_Latn', 'zsm_Latn', 'arb_Arab', 'ben_Latn', 'ell_Grek', 'guj_Gujr', 'ibo_Latn', 'kat_Geor', 'lin_Latn', 'mri_Latn', 'pan_Guru', 'shn_Mymr', 'spa_Latn', 'tgl_Latn', 'uzn_Latn', 'zul_Latn', 'arb_Latn', 'bod_Tibt', 'eng_Latn', 'hat_Latn', 'ilo_Latn', 'kaz_Cyrl', 'lit_Latn', 'mya_Mymr', 'pbt_Arab', 'sin_Latn', 'srp_Cyrl', 'tha_Thai', 'vie_Latn', 'ars_Arab', 'bul_Cyrl', 'est_Latn', 'hau_Latn', 'ind_Latn', 'kea_Latn', 'lug_Latn', 'nld_Latn', 'pes_Arab', 'sin_Sinh', 'ssw_Latn', 'tir_Ethi', 'war_Latn', 'ary_Arab', 'cat_Latn', 'eus_Latn', 'heb_Hebr', 'isl_Latn', 'khk_Cyrl', 'luo_Latn', 'nob_Latn', 'plt_Latn', 'slk_Latn', 'sun_Latn', 'tsn_Latn', 'wol_Latn']

results = {}

for lang in all_languages:
    print(f"→ {lang}")
    ds = load_dataset("facebook/belebele", lang, split="test")
    ds = ds.select(range(min(100, len(ds))))
    correct, total = 0, 0

    for ex in ds:
        ctx     = ex["flores_passage"]
        q       = ex["question"]
        opts    = [ex[f"mc_answer{i}"] for i in range(1,5)]
        gold    = chr(64 + int(ex["correct_answer_num"]))
        
        pred = generate_answer(model, sp, ctx, q, opts)
        print(pred, gold)
        if pred == gold:
            correct += 1
        total += 1

    acc = correct/total if total else 0.0
    results[lang] = acc
    print(f"  accuracy: {acc:.2%}")

# save out
with open("results_custom_gpt.json","w") as f:
    json.dump(results, f, indent=2)

# summary
for lang,acc in sorted(results.items(), key=lambda x:-x[1]):
    print(f"{lang:12s}: {acc:.2%}")
