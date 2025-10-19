NGPUS=8                        # ← adapt!
awk -v ngpu="$NGPUS" '
  /iteration/ {
      # grab the numbers that sit **after** the two key phrases
      for (i = 1; i <= NF; i++) {
          if ($i == "(TFLOP/s/GPU):") tflops = $(i+1);
          if ($i == "(ms):")          ms     = $(i+1);
      }
      # per-iteration, all-GPU FLOPs (in tera, so ×1e12 later)
      iter_flops_tera = tflops * ms / 1000 * ngpu;
      total_tera += iter_flops_tera;
  }
  END {
      total_flops = total_tera * 1e12;      # raw FLOPs
      printf "Total = %.3e FLOPs (≈ %.2f PFLOPs)\n",
             total_flops, total_flops/1e15;
  }
' train.log
