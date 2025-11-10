from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

tokenizer: Tokenizer = Tokenizer.from_file(path="tokenizer.json") 

tokenizer =  PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<|endoftext|>", eos_token="<|endoftext|>",)

encoded = tokenizer.encode("Hello World")
print(encoded)
print(tokenizer.decode(encoded))
