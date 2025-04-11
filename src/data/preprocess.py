import os
from pathlib import Path
import shutil
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import load_dataset

def clean_text(examples: dict) -> dict:
    """
    Cleans a batch of raw text examples.

    Assumes each line in the raw text file has the following format:
        <id>\t<timestamp>:<actual sentence>

    The cleaning process splits each line by the tab ('\t') and then
    by the colon (':') to retrieve the actual sentence. It then strips
    any leading/trailing whitespace.
    """
    cleaned_texts = []
    for text in examples["text"]:
        if text.strip():
            parts = text.split("\t")
            if len(parts) >= 2:
                # Take the part after the first tab, then split by colon,
                # taking the substring after the colon and stripping whitespace.
                sentence = parts[1].split(":", 1)[-1].strip()
                cleaned_texts.append(sentence)
            else:
                # If not in expected format, simply strip the text
                cleaned_texts.append(text.strip())
    return {"text": cleaned_texts}

def main():
    # 1. Ensure local directory for dataset (root-level "datasets/")
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define the dataset file to look for
    data_file = dataset_dir / "deu_news_2015_3M-sentences.txt"

    # 3. If the file doesn't exist, download it using KaggleHub
    if not data_file.exists():
        print(f"Dataset file not found at: {data_file}")
        print("Downloading from KaggleHub (rtatman/3-million-german-sentences)...")
        # Remove the target_dir parameter (not supported)
        downloaded_path = kagglehub.dataset_download("rtatman/3-million-german-sentences")
        print("Downloaded dataset files to:", downloaded_path)

        # 4. Copy the required file into the datasets/ directory
        downloaded_file = Path(downloaded_path) / "deu_news_2015_3M-sentences.txt"
        if downloaded_file.exists():
            shutil.copy(downloaded_file, data_file)
            print("Copied dataset file to:", data_file)
        else:
            print("Warning: Expected file not found in downloaded dataset.")
            return
    else:
        print(f"Dataset file found locally at: {data_file}")

    # 5. Load the dataset using Hugging Face Datasets
    print("Loading raw data...")
    try:
        dataset = load_dataset(
            "text",
            data_files=str(data_file),
            split="train",
            streaming=False  # Loads data into memory (change to True for large files)
        )
        print(f"Loaded dataset with {len(dataset)} examples.")
    except Exception as e:
        print(f"Error loading dataset from {data_file}: {e}")
        return

    # 6. Clean the dataset
    print("Cleaning the dataset...")
    dataset = dataset.map(clean_text, batched=True)
    print("Cleaning complete.")

    # 7. Split the dataset (99% train, 1% test)
    print("Splitting the dataset into train and test splits...")
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # 8. Convert to pandas DataFrame and save as Parquet in a root-level 'pre_processed_data/' directory
    processed_data_path = Path("pre_processed_data")
    processed_data_path.mkdir(exist_ok=True, parents=True)

    train_file = processed_data_path / "train_data.snap.parquet"
    test_file = processed_data_path / "eval_data.snap.parquet"

    print("Saving datasets as Parquet files...")
    pd.DataFrame({"text": train_dataset["text"]}).to_parquet(train_file, compression="snappy")
    pd.DataFrame({"text": test_dataset["text"]}).to_parquet(test_file, compression="snappy")

    print(f"Train data saved to: {train_file}")
    print(f"Evaluation data saved to: {test_file}")
    print("Processing finished.")

if __name__ == "__main__":
    main()
