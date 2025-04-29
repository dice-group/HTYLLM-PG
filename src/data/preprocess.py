from pathlib import Path
import pandas as pd
import sys
from random import random
from datasets import load_dataset

def size_in_bytes(example: str) -> int:
    """Calculate the size of a single dataset example in bytes."""
    return sys.getsizeof(example)

def clean_text_streaming(example: str) -> str:
    """
    Cleans a single raw text example.

    Assumes each line in the raw text file has the following format:
        <id>\t<timestamp>:<actual sentence>
    """
    cleaned_texts = []
    for line in example.split("\n"):
        if line.strip():  # Ignore empty lines
            parts = line.split("\t")
            if len(parts) >= 2:
                # Extract the actual sentence
                sentence = parts[1].split(":", 1)[-1].strip()
                cleaned_texts.append(sentence)
            else:
                # If the format is unexpected, include the cleaned line
                cleaned_texts.append(line.strip())
    return "\n".join(cleaned_texts)

def process_limited_dataset(dataset, size_limit_gb=10, test_split_ratio=0.01):
    """
    Process and split the dataset dynamically in streaming mode.

    Args:
        dataset: The Hugging Face dataset in streaming mode
        size_limit_gb: The maximum size to process, in gigabytes
        test_split_ratio: Percent of examples to put into the test set

    Returns:
        train_data: A list of processed training examples
        test_data: A list of processed test examples
    """
    train_data = []
    test_data = []
    total_size = 0
    total_examples = 0
    size_limit_bytes = size_limit_gb * (1024 ** 3)  # Convert GB to bytes

    for raw_example in dataset:
        raw_text = raw_example["text"]

        # Clean the raw text
        cleaned_example = clean_text_streaming(raw_text)

        # Measure size of the cleaned example
        example_size = size_in_bytes(cleaned_example)
        total_size += example_size

        print(f"Progress: {round(total_size/size_limit_bytes*100, 2)} %")

        # Split into training and testing
        if random() < test_split_ratio:
            test_data.append(cleaned_example)
        else:
            train_data.append(cleaned_example)

        total_examples += 1

        # Stop processing once we hit the predefined size limit
        if total_size >= size_limit_bytes:
            print(f"Processed up to {size_limit_gb} GB. Stopping...")
            break

    print(f"Total examples processed: {total_examples}")
    print(f"Total size processed: {total_size / (1024 ** 3):.2f} GB")

    return train_data, test_data

def main():
    print("Downloading and streaming the dataset from Hugging Face...")

    try:
        # Load the "train" split in streaming mode
        dataset = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)

        # Process and split dataset dynamically
        train_data, test_data = process_limited_dataset(dataset, size_limit_gb=10)
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return

    print(f"Train data size: {len(train_data)} examples.")
    print(f"Test data size: {len(test_data)} examples.")

    # Save the split datasets to disk
    processed_data_path = Path("pre_processed_data")
    processed_data_path.mkdir(exist_ok=True, parents=True)

    train_file = processed_data_path / "train_data.snap.parquet"
    test_file = processed_data_path / "eval_data.snap.parquet"

    print("Saving processed datasets...")
    pd.DataFrame({"text": train_data}).to_parquet(train_file, compression="snappy")
    pd.DataFrame({"text": test_data}).to_parquet(test_file, compression="snappy")

    print(f"Train data saved to: {train_file}")
    print(f"Evaluation data saved to: {test_file}")
    print("Processing finished.")

if __name__ == "__main__":
    main()