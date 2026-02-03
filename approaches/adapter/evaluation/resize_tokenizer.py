from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("/upb/users/j/joeldag/profiles/unix/cs/helper_tokenizer_resized_250880")
tokenizer.add_tokens(["<extra_token_{}>".format(i) for i in range(250680 - len(tokenizer))])
tokenizer.save_pretrained("/upb/users/j/joeldag/profiles/unix/cs/helper_tokenizer_resized_250680")
