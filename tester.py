import tokenizer as TOKENIZER

def encode(text):
    tokenizer = TOKENIZER.BPETokenizer(method="encode-decode")
    return tokenizer.encode(text)

def decode(token_ids):
    tokenizer= TOKENIZER.BPETokenizer(method="encode-decode")
    return tokenizer.decode(token_ids)

sentence="Hello my name is sylo"
encoded= encode(sentence)
decoded = decode(encoded)

print(encoded)
print(f"original sentence: {sentence} \n")
print(f"sentence length: {len(sentence)} \n")
print(f"encoded length: {len(encoded)} \n")
print(f"decoded sentence: {decoded} \n")
