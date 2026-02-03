import sentencepiece as spm

print("Training tokenizer...")

spm.SentencePieceTrainer.train(
    input="../data/corpus.txt",          
    model_prefix="sp_model_131072",     
    vocab_size=131072,            
    model_type="bpe",        
    character_coverage=0.9995,   
    split_digits=True,           
    max_sentence_length=16384,   
    num_threads=128,             
    input_sentence_size=2_000_000,  
    shuffle_input_sentence=True, 
    pad_id=0,                    
    unk_id=1,                    
    bos_id=2,                    
    eos_id=3                     
)

print("Tokenizer trained successfully.")