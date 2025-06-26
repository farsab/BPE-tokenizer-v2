from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def train_bpe_tokenizer(text_file, vocab_size=10000, save_path="tokenizer/"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train([text_file], trainer)
    
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(f"{save_path}/bpe_tokenizer.json")
    return f"{save_path}/bpe_tokenizer.json"
