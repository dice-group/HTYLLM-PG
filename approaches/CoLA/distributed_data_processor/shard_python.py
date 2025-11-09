import gzip, os, time, shutil, argparse
from multiprocessing import Pool, cpu_count

CHUNK = 512 * 1024 * 1024

def process_file(args):
    path, src, dst = args
    t0 = time.time()
    rel = os.path.relpath(path, src)
    out_dir = os.path.join(dst, os.path.dirname(rel))
    os.makedirs(out_dir, exist_ok=True)

    size = os.path.getsize(path)
    name = os.path.splitext(os.path.basename(path))[0]
    if size <= CHUNK:
        shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))
        print(f"[COPY] {rel} ({size/1e6:.1f} MB) in {time.time()-t0:.1f}s")
        return

    buf, chunks, idx, cur = bytearray(), [], 0, 0
    with gzip.open(path, "rb") as fin:
        for line in fin:
            buf.extend(line); cur += len(line)
            if cur >= CHUNK:
                chunks.append((idx, bytes(buf))); buf.clear(); cur = 0; idx += 1
        if buf: chunks.append((idx, bytes(buf)))
    for idx, data in chunks:
        out = os.path.join(out_dir, f"{name}_part_{idx:05d}.jsonl.gz")
        with gzip.open(out, "wb", compresslevel=1) as f: f.write(data)
    print(f"[SHARD] {rel} -> {len(chunks)} parts in {time.time()-t0:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Shard or copy .gz files by size.")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    src, dst = args.input, args.output
    os.makedirs(dst, exist_ok=True)
    files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs if f.endswith(".gz")]
    print(f"[INFO] Found {len(files)} files")
    start = time.time()
    with Pool(cpu_count()) as p:
        for _ in p.imap_unordered(process_file, [(f, src, dst) for f in files]): pass
    print(f"[TOTAL] Done all in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
