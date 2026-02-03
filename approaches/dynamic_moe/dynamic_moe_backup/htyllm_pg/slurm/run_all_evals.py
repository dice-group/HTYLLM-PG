import os
import sys
import argparse
import subprocess
import re
from pathlib import Path

def get_step_number(folder_name):
    match = re.search(r'step_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return -1

def main():
    parser = argparse.ArgumentParser(description="Submit evaluation jobs for all checkpoints.")
    parser.add_argument("--checkpoints-dir", type=str, default="../checkpoints_multilingual",
                        help="Path to the directory containing checkpoint folders (default: ../checkpoints_multilingual)")
    parser.add_argument("--script-path", type=str, default="htyllm_pg/slurm/convert_and_eval.sh",
                        help="Path to the convert_and_eval.sh script (default: htyllm_pg/slurm/convert_and_eval.sh)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--pattern", type=str, default="step_*", help="Glob pattern for checkpoint folders (default: step_*)")
    
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    script_path = Path(args.script_path).resolve()

    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    # Find all checkpoint folders
    checkpoints = []
    for entry in checkpoints_dir.glob(args.pattern):
        if entry.is_dir() and "step_" in entry.name:
            step = get_step_number(entry.name)
            if step >= 0:
                checkpoints.append((step, entry))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])

    if not checkpoints:
        print(f"No checkpoints found in {checkpoints_dir} matching pattern {args.pattern}")
        sys.exit(0)

    print(f"Found {len(checkpoints)} checkpoints.")

    for step, checkpoint_path in checkpoints:
        job_name = f"eval_step_{step}"
        
        # Construct sbatch command
        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            str(script_path),
            "--checkpoint", str(checkpoint_path)
        ]

        if args.dry_run:
            print(f"[Dry Run] {' '.join(cmd)}")
        else:
            print(f"Submitting job for step {step}...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job for {checkpoint_path}: {e}")

if __name__ == "__main__":
    main()
