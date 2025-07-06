import tokenizer as TOKENIZER

def encode(text):
    tokenizer = TOKENIZER.BPETokenizer()
    return tokenizer.encode(text)

def decode(token_ids):
    tokenizer= TOKENIZER.BPETokenizer()
    return tokenizer.decode(token_ids)

sentence="This is the final test and update for the BPE Tokenizer. It should work perfectly now, and I am very happy with the results. I hope you are too! :)"
encoded= encode(sentence)
decoded = decode(encoded)

print(f"original sentence: {sentence} \n")
print(f"encoded sentence: {encoded} \n")
print(f"sentence length: {len(sentence)} \n")
print(f"encoded length: {len(encoded)} \n")
print(f"decoded sentence: {decoded} \n")

