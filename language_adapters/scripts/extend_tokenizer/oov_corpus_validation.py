import argparse, gzip, json, re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

def iter_texts(data_dir, max_lines=None):
    seen = 0
    for file in Path(data_dir).rglob("*.jsonl.gz"):
        with gzip.open(file, "rt", encoding="utf-8") as f:
            for line in f:
                if max_lines and seen >= max_lines:
                    return
                obj = json.loads(line)
                if "text" in obj:
                    yield obj["text"]
                    seen += 1

def analyze(tokenizer, texts, max_pieces):
    total_tok = unk_tok = total_words = over_frag = 0
    unk_id = tokenizer.unk_token_id
    for text in texts:
        words = text.split()
        enc = tokenizer(words, add_special_tokens=False, is_split_into_words=True)
        ids, wids = enc["input_ids"], enc.word_ids()
        total_tok += len(ids)
        unk_tok += sum(i == unk_id for i in ids)
        piece_counts = Counter()
        for wid in wids:
            if wid is not None:
                piece_counts[wid] += 1
        total_words += len(words)
        over_frag += sum(c > max_pieces for c in piece_counts.values())
    return total_tok, unk_tok, total_words, over_frag

def main(model, data_dir, max_lines, max_pieces):
    tok = AutoTokenizer.from_pretrained(model)
    tot, unk, wtot, frag = analyze(tok, iter_texts(data_dir, max_lines), max_pieces)
    print(f"Total tokens: {tot}")
    print(f"UNK tokens: {unk}  ({unk/tot*100:.2f}%)")
    print(f"Total words : {wtot}")
    print(f"Words with >{max_pieces} pieces: {frag}  ({frag/wtot*100:.2f}%)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Pretrained tokenizer/model path")
    p.add_argument("--data_dir", required=True, help="Directory with *.jsonl.gz files")
    p.add_argument("--max_lines", type=int, default=None, help="Limit lines read for speed")
    p.add_argument("--max_pieces", type=int, default=5, help="Pieces per word threshold")
    args = p.parse_args()
    main(args.model, args.data_dir, args.max_lines, args.max_pieces)
