## Setup

### Create and activate the environment:
1. Execute the following command:
```
source venv-jupyterhub/bin/activate
```
2. Install all required packages.
### Setup folder structure:
- Put jsonl files of languages into ./data folder.
- Put necessary scripts (tokenizer.py, desired model script (e.g. gemma-3-4b.py), lm_harness_eval.py, and lm_eval_runner.py) into ./scripts folder.

## Tokenization
Execute the following command:
```
python scripts/tokenizer.py data/{languageCode}.jsonl
```
The tokenized data will be saved at data/tokenized_data.
### Huggingface Login
1. Execute the following command:
```
huggingface-cli login
```
2. Enter your *login token*.

## Fine-Tuning
The basic command looks like:
```
python scripts/gemma-3-4b.py data/tokenized_data/{languageCode}.bin
```
If you want to run the training even beyond the terminal session you can use the `nohup` prefix, e.g. `nohup bash -c "python scripts/gemma-3-4b.py data/tokenized_data/{languageCode}.bin"`
Also, saving the output into a log was very helpful. You can simply do that by adding a desired txt file in the end to log to, e.g. `nohup bash -c "python scripts/gemma-3-4b.py data/tokenized_data/{languageCode}.bin" > {languageCode}_gemma-3_log.txt 2>&1 &` <br>*(This is what my final execution command looked like.)*

The model's checkpoints will be saved at models/gemma3_fine_tuned_model_{languageCode}.

## Evaluation
Execute the following command:
```
python scripts/lm_eval_runner.py models/gemma3_fine_tuned_model_{languageCode}/checkpoints
```
If you want to counteract a Memory Overload, also add the prefix `TORCH_COMPILE=0`.
The `nohup` prefix and logging as specified above can also be applied here.

##
***Note**: {languageCode} is a placeholder for the desired language code, i.e. one of `arb_Arab`, `bod_Tibt`, `spa_Latn`, `swh_Latn`, `tam_Taml`, and `tur_Latn`.
