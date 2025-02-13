
OUTPUT_FILE_TEMPLATE = "/shared-file-storage/preprocessed_data/preprocessed_{rank}.txt"

def preprocess_text(text):
    return text.lower().strip().split()

def load_dataset(*args, **kwargs):
    # TODO: There should be some logic about loading the dataset here. As we only have 2 Gigabytes of memory available per process, maybe that should play a role as well ;)
    raise NotImplementedError("This function has not been implemented yet.")

def write_preprocessed_text(preprocessed_text, rank):
    with open(OUTPUT_FILE_TEMPLATE.format(rank=rank), "w") as f:
        for line in preprocessed_text:
            f.write("\t".join(line))
            f.write("\n")

def main():
    # TODO: Here you should try to check the rank of this process and the total number of processes that are spawned
    local_rank = None # TODO: This should be the rank of the process
    total_procs = None # TODO: This should be the total number of processes that are spawned
    # Extract the text to process
    text_to_process = load_dataset() # TODO: pass the relevant arguments (if any)
    # Preprocess the text
    preprocessed_text = [preprocess_text(text) for text in text_to_process]
    # TODO: Write this somewhere
    write_preprocessed_text(preprocessed_text, rank=local_rank)
    raise NotImplementedError("This function has not been fully implemented yet.")

if __name__ == "__main__":
    main()