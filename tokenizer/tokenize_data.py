import sentencepiece as spm
import os

# --- CONFIGURATION ---
MODEL_PATH       = "tokenizer/sp_model.model"   # your trained SentencePiece model
INPUT_PATH       = "../data/corpus.txt"       # raw text, one sentence per line
OUT_DIR          = "tokenizer/shards"           # where to write shard files
TOKENS_PER_SHARD = 100_000_000        # desired max tokens per shard

os.makedirs(OUT_DIR, exist_ok=True)

# --- INITIALIZE PROCESSOR ---
sp = spm.SentencePieceProcessor()
sp.load(MODEL_PATH)

# --- SHARDING LOOP ---
shard_idx   = 0
token_count = 0

# open first shard file
out_f = open(os.path.join(OUT_DIR, f"shard_{shard_idx:04d}.tok"), "w", encoding="utf-8")

with open(INPUT_PATH, "r", encoding="utf-8") as fin:
    for line in fin:
        pieces = sp.encode_as_pieces(line.strip())
        for piece in pieces:
            # write each token + space
            out_f.write(piece + " ")
            token_count += 1

            # if we've reached the limit, roll over to next shard
            if token_count >= TOKENS_PER_SHARD:
                out_f.write("\n")  # finish the current line
                out_f.close()

                shard_idx  += 1
                token_count = 0
                out_f = open(
                    os.path.join(OUT_DIR, f"shard_{shard_idx:04d}.tok"),
                    "w", encoding="utf-8"
                )

        # after each input line, end with a newline in shard
        out_f.write("\n")

# close final shard
out_f.close()
print(f"Done: created {shard_idx+1} shard(s) in `{OUT_DIR}`")
