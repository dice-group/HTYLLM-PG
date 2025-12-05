# This is a helper do decide token budget for the trainings
We differentiate between the scale of the trainings
Every training will use llama3-8b for now. The budget calculation are based on results form 'approaches/CoLA/docs/compute_test_results.md'
We define different token budgets for trainings over 12, 72-95, 200, and 635 langs
Why 635, because the number of languages available in fineweb with more than 500 documents is 635. With less data we hypothise that oversampling will be to high (reference Glot500 paper https://aclanthology.org/2023.acl-long.61/. They only include langs with more than 30.000 sentneces. Why hyposhise that we can go less. 1. becasue we want to have more langs then them, second becasue out cluster and expert allocation should drastically boost cross lingual finetuning)

### Token budget for 12-language training

Given:  
- Training FLOPs/token: $23.4\,\mathrm{GFLOPs} = 23.4\times10^{9}$  
- Sustained FLOPs/s/GPU: $407\,\mathrm{TFLOPs/s} = 407\times10^{12}$  
- GPUs: $2$  
- Time: $2\,\mathrm{days} = 172{,}800\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = \dfrac{ (\mathrm{GPUs}) \cdot (\mathrm{FLOPs/s/GPU}) \cdot (\mathrm{time}) }{ \mathrm{FLOPs/token} } $

Compute tokens/s/GPU:  
$ \dfrac{407\times10^{12}}{23.4\times10^{9}} = \dfrac{407}{23.4}\times10^{3} \approx 17{,}393\ \mathrm{tokens/s} $

Tokens per GPU for 2 days:  
$ 17{,}393 \times 172{,}800 \approx 3.006\times10^{9} $

Tokens for 2 GPUs:  
$ 2 \times 3.006\times10^{9} \approx 6.01\times10^{9} \approx 6.0\,\mathrm{B} $

### Token budget for 72-95 lang tier

Given:  
- Training FLOPs/token: $23.4\,\mathrm{GFLOPs} = 23.4\times10^{9}$  
- Sustained FLOPs/s/GPU: $407\,\mathrm{TFLOPs/s} = 407\times10^{12}$  
- GPUs: $4$  
- Time: $4\,\mathrm{days} = 345{,}600\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = \dfrac{ (\mathrm{GPUs}) \cdot (\mathrm{FLOPs/s/GPU}) \cdot (\mathrm{time}) }{ \mathrm{FLOPs/token} } $

Compute tokens/s/GPU:  
$ \dfrac{407\times10^{12}}{23.4\times10^{9}} = \dfrac{407}{23.4}\times10^{3} \approx 17{,}393\ \mathrm{tokens/s} $

Tokens per GPU for 4 days:  
$ 17{,}393 \times 345{,}600 \approx 6.011\times10^{9} $

Tokens for 4 GPUs:  
$ 4 \times 6.011\times10^{9} \approx 24.044\times10^{9} \approx 24.0\,\mathrm{B} $

### Token budget for 200 lang subset

Given:  
- Training FLOPs/token: $23.4\,\mathrm{GFLOPs} = 23.4\times10^{9}$  
- Sustained FLOPs/s/GPU: $407\,\mathrm{TFLOPs/s} = 407\times10^{12}$  
- GPUs: $8$  
- Time: $5\,\mathrm{days} = 432{,}000\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = \dfrac{ (\mathrm{GPUs}) \cdot (\mathrm{FLOPs/s/GPU}) \cdot (\mathrm{time}) }{ \mathrm{FLOPs/token} } $

Compute tokens/s/GPU:  
$ \dfrac{407\times10^{12}}{23.4\times10^{9}} = \dfrac{407}{23.4}\times10^{3} \approx 17{,}393.16\ \mathrm{tokens/s} $

Tokens per GPU for 5 days:  
$ 17{,}393.16 \times 432{,}000 \approx 7.513846154\times10^{9} $

Tokens for 8 GPUs:  
$ 8 \times 7.513846154\times10^{9} \approx 6.011076923\times10^{10} \approx 60.1\,\mathrm{B} $

### Token budget for 635 lang subset
Now maybe the final model with the best variation can be trained with 635 langs once for on week on 16 GPUs

Given:  
- Training FLOPs/token: $23.4\,\mathrm{GFLOPs} = 23.4\times10^{9}$  
- Sustained FLOPs/s/GPU: $407\,\mathrm{TFLOPs/s} = 407\times10^{12}$  
- GPUs: $16$  
- Time: $7\,\mathrm{days} = 604{,}800\,\mathrm{s}$

Formula:  
$ \mathrm{tokens} = \dfrac{ (\mathrm{GPUs}) \cdot (\mathrm{FLOPs/s/GPU}) \cdot (\mathrm{time}) }{ \mathrm{FLOPs/token} } $

Compute tokens/s/GPU:  
$ \dfrac{407\times10^{12}}{23.4\times10^{9}} = \dfrac{407}{23.4}\times10^{3} \approx 17{,}393.16\ \mathrm{tokens/s} $

Tokens per GPU for 7 days:  
$ 17{,}393.16 \times 604{,}800 \approx 1.0519384615\times10^{10} \approx 10.52\,\mathrm{B} $

Tokens for 16 GPUs:  
$ 16 \times 10.52\times10^{9} \approx 1.683101538\times10^{11} \approx 168.3\,\mathrm{B} $

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

Stage budget $B = 6.0\,\mathrm{B}$ tokens:

- $B_{\mathrm{EN}} \approx 4.79\,\mathrm{B}$  
- $B_{\mathrm{X}} \approx 1.21\,\mathrm{B}$

(Optional: apply floors/ceilings and renormalize.)

# References and why

- XLM-R uses exponent/temperature sampling and finds α = 0.3 a good overall choice. Specifically they say "0.3 to be an optimal value for α, and use this for XLM-R." https://arxiv.org/pdf/1911.02116
- mT5 reports its final model uses α = 0.3 and discusses prior α values https://aclanthology.org/2021.naacl-main.41.pdf
- mDAPT uses α = 0.3 https://aclanthology.org/2021.findings-emnlp.290.pdf
- Glot500 applies the same method

However these are all pretraining approaches. The following finetuning approach also uses this, but its not peer reviewed yet
- MaLA-500 explicitly uses α = 0.3 for vocabulary extension and continued pretraining on Glot500      https://arxiv.org/pdf/2401.13303

