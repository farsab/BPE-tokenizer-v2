from tokenizer_tools.custom_bpe_tokenizer import train_bpe_tokenizer
import os

def main():
    sample_text = """Deep learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain."""
    os.makedirs("data", exist_ok=True)
    text_path = "data/sample.txt"
    with open(text_path, "w") as f:
        f.write(sample_text)
    
    tokenizer_path = train_bpe_tokenizer(text_path)
    print(f"Trained BPE tokenizer saved to: {tokenizer_path}")

if __name__ == "__main__":
    main()
