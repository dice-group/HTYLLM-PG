## 🧪 Setup (Conda)

Create and activate the environment:

```bash
conda create -n icebreaker python=3.12.9
conda activate icebreaker
pip install -e .
```

https://www.kaggle.com/datasets/rtatman/3-million-german-sentences/data 

https://www.kaggle.com/code/steffenhaeussler/train-a-language-model-from-scratch#Model-Training

https://huggingface.co/docs/transformers/main/model_doc/gpt2

tikoken examples: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

This dataset is released under a CC-BY 4.0 license. If you use this dataset in your work, please cite the following paper:

D. Goldhahn, T. Eckart & U. Quasthoff: Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200 Languages.
In: Proceedings of the 8th International Language Resources and Evaluation (LREC'12), 2012



Run mbert_train.sh

or alternatively run below commands.

tokenizer:

python src/data/tokenizer.py

output will be under tokenized_data as binary files

train:

python src/model/mBERT_2.py
