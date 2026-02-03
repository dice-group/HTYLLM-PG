# This is a helper do decide token budget for the trainings
We differentiate between the scale of the trainings
Every training will use llama3-8b for now. The budget calculation are based on results form 'approaches/CoLA/docs/compute_test_results.md'
We define different token budgets for trainings over 12, 72-95, 200, and 635 langs
Why 635, because the number of languages available in fineweb with more than 500 documents is 635. With less data we hypothise that oversampling will be to high (reference Glot500 paper https://aclanthology.org/2023.acl-long.61/. They only include langs with more than 30.000 sentneces. Why hyposhise that we can go less. 1. becasue we want to have more langs then them, second becasue out cluster and expert allocation should drastically boost cross lingual finetuning)

**Empirical throughput adjustment (important)**  
The FLOPs-based calculation below assumes ~407 TFLOPs/s/GPU, which would yield ~17.4k tokens/s/GPU.  
In practice, our current LLaMA‑Factory SFT setup on H100 (bs=2, grad_acc=2, seq_len≈2k) shows ~6.9k input tokens/s on 2 GPUs, i.e. **≈3.4–3.5k tokens/s/GPU (~0.2× theoretical)**.  
To keep the walltimes in `scripts/comparison/run_multilingual_ablation.sh` (2d/4d/5d/7d) we therefore use the **empirical rate** for the final budgets.

### Token budget for 12-language training

Given:  
- Effective tokens/s/GPU (empirical): $\approx 3{,}500\ \mathrm{tokens/s}$  
- GPUs: $2$  
- Time: $2\,\mathrm{days} = 172{,}800\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = (\mathrm{GPUs}) \cdot (\mathrm{tokens/s/GPU}) \cdot (\mathrm{time}) $

Tokens per GPU for 2 days:  
$ 3{,}500 \times 172{,}800 \approx 6.05\times10^{8} $

Tokens for 2 GPUs:  
$ 2 \times 6.05\times10^{8} \approx 1.21\times10^{9} \approx 1.2\,\mathrm{B} $

### Token budget for 72-95 lang tier

Given:  
- Effective tokens/s/GPU (empirical): $\approx 3{,}500\ \mathrm{tokens/s}$  
- GPUs: $4$  
- Time: $4\,\mathrm{days} = 345{,}600\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = (\mathrm{GPUs}) \cdot (\mathrm{tokens/s/GPU}) \cdot (\mathrm{time}) $

Tokens per GPU for 4 days:  
$ 3{,}500 \times 345{,}600 \approx 1.2096\times10^{9} $

Tokens for 4 GPUs:  
$ 4 \times 1.2096\times10^{9} \approx 4.8384\times10^{9} \approx 4.8\,\mathrm{B} $

### Token budget for 200 lang subset

Given:  
- Effective tokens/s/GPU (empirical): $\approx 3{,}500\ \mathrm{tokens/s}$  
- GPUs: $8$  
- Time: $5\,\mathrm{days} = 432{,}000\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = (\mathrm{GPUs}) \cdot (\mathrm{tokens/s/GPU}) \cdot (\mathrm{time}) $

Tokens per GPU for 5 days:  
$ 3{,}500 \times 432{,}000 \approx 1.512\times10^{9} $

Tokens for 8 GPUs:  
$ 8 \times 1.512\times10^{9} \approx 1.2096\times10^{10} \approx 12.1\,\mathrm{B} $

### Token budget for 635 lang subset
Now maybe the final model with the best variation can be trained with 635 langs once for on week on 16 GPUs

Given:  
- Effective tokens/s/GPU (empirical): $\approx 3{,}500\ \mathrm{tokens/s}$  
- GPUs: $16$  
- Time: $7\,\mathrm{days} = 604{,}800\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = (\mathrm{GPUs}) \cdot (\mathrm{tokens/s/GPU}) \cdot (\mathrm{time}) $

Tokens per GPU for 7 days:  
$ 3{,}500 \times 604{,}800 \approx 2.1168\times10^{9} \approx 2.12\,\mathrm{B} $

Tokens for 16 GPUs:  
$ 16 \times 2.1168\times10^{9} \approx 3.38688\times10^{10} \approx 33.9\,\mathrm{B} $

# Deicde sampling size per language
Now based on the token budget, for each lang in the language subsets, we define how much we sample from those

### Alpha-smoothing recipe (α = 0.3)

Given raw per-language sizes \(n_\ell\):

1. Weights:  
   $w_\ell = n_\ell^{0.3}$

2. Probabilities:  
   $p_\ell = \dfrac{w_\ell}{\sum_j w_j}$

3. Budgeted samples for total stage budget \(B\):  
   $B_\ell = B \cdot p_\ell$

---

### Toy example

Raw sizes:  
- English: $1{,}000{,}000$  
- Lang X: $10{,}000$

Weights:  
- $w_{\mathrm{EN}} = (1{,}000{,}000)^{0.3} = 10^{6\cdot0.3} = 10^{1.8} \approx 63.10$  
- $w_{\mathrm{X}} = (10{,}000)^{0.3} = 10^{4\cdot0.3} = 10^{1.2} \approx 15.85$

Probabilities:  
- $p_{\mathrm{EN}} = \dfrac{63.10}{63.10 + 15.85} \approx 0.799$  
- $p_{\mathrm{X}} = \dfrac{15.85}{63.10 + 15.85} \approx 0.201$

Stage budget $B = 1.2\,\mathrm{B}$ tokens:

- $B_{\mathrm{EN}} \approx 0.96\,\mathrm{B}$  
- $B_{\mathrm{X}} \approx 0.24\,\mathrm{B}$

(Optional: apply floors/ceilings and renormalize.)

# References and why

- XLM-R uses exponent/temperature sampling and finds α = 0.3 a good overall choice. Specifically they say "0.3 to be an optimal value for α, and use this for XLM-R." https://arxiv.org/pdf/1911.02116
- mT5 reports its final model uses α = 0.3 and discusses prior α values https://aclanthology.org/2021.naacl-main.41.pdf
- mDAPT uses α = 0.3 https://aclanthology.org/2021.findings-emnlp.290.pdf
- Glot500 applies the same method

However these are all pretraining approaches. The following finetuning approach also uses this, but its not peer reviewed yet
- MaLA-500 explicitly uses α = 0.3 for vocabulary extension and continued pretraining on Glot500      https://arxiv.org/pdf/2401.13303
