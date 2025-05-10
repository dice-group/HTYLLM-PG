# Lora Language Adapter Training Guide

1. **Create a Conda environment with Python 3.10.17**:

   ```bash
   conda create -n htyllm python=3.10.17
   conda activate htyllm
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Tokenize Data

Run the script:

```bash
./tokenize_adapter_data.sh
```

- Ensure that the **data path in the script** points to your **downloaded FineWeb2 dataset**.
- The script will save **tokenized data** to the configured output directory.

---

## Train Adapter

1. Edit `train_language_adapter.sh` and set:
   - `TOKENIZED_DATA_DIR` to the path where tokenized data was saved
   - `OUTPUT_DIR` to your desired output directory

2. Run training:

   ```bash
   ./train_language_adapter.sh
   ```

3. Training outputs (including logs and checkpoints) will be saved in the output directory.

---

## Visualize results with TensorBoard

To monitor training progress you can visualize the resutls with vs code tensorboard extension.
Download the extension, when using it, point to the log dir wherer training evaluation results are saved
