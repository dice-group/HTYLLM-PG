import argparse
import gzip
from pathlib import Path

def collect_lines(languages, input_dir, output_dir, lines_per_language):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    print(f"Input dir: {input_dir}, Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        print(f"\nProcessing language: {lang}")
        lang_input_path = input_dir / f"{lang}.jsonl"
        lang_output_path = output_dir / f"{lang}.jsonl"
        print(f"Looking in: {lang_input_path}")

        lang_output_path.mkdir(parents=True, exist_ok=True)

        gz_files = sorted(lang_input_path.glob("*.jsonl.gz"))
        print(f"Found files: {gz_files}")
        if not gz_files:
            print(f"⚠️ No files found for {lang}, skipping.")
            continue

        total_lines = 0
        file_index = 0

        for gz_file in gz_files:
            if total_lines >= lines_per_language:
                break

            print(f"Reading: {gz_file}")
            with gzip.open(gz_file, "rt", encoding="utf-8") as infile:
                output_path = lang_output_path / f"{file_index:05d}.jsonl.gz"
                print(f"Writing to: {output_path}")
                with gzip.open(output_path, "wt", encoding="utf-8") as outfile:
                    for line in infile:
                        if total_lines >= lines_per_language:
                            break
                        outfile.write(line)
                        total_lines += 1
                file_index += 1

        print(f"{lang}: Collected {total_lines} lines.")

def main():
    parser = argparse.ArgumentParser(description="Collect a subset of lines from compressed JSONL files per language.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, required=True,
                        help="Comma-separated list of languages, e.g., apc_Arab,arz_Arab")
    parser.add_argument("--lines", type=int, default=28000, help="Lines per language to sample")

    args = parser.parse_args()
    languages = [lang.strip() for lang in args.langs.split(",")]
    collect_lines(languages, args.input_dir, args.output_dir, args.lines)

if __name__ == "__main__":
    main()
