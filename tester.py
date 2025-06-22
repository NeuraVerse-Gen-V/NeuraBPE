import tokenizer as TOKENIZER

def encode(text):
    tokenizer = TOKENIZER.BPETokenizer(method="encode-decode")
    return tokenizer.encode(text)

def decode(token_ids):
    tokenizer= TOKENIZER.BPETokenizer(method="encode-decode")
    return tokenizer.decode(token_ids)

with open("data.txt", "r", encoding="utf-8") as file:
    data = file.read()


sentence="Hello there i guess my encoder-decoder finally works, i am so happy to see this, i hope it works for you too, if not then please let me know and i will try to fix it as soon as possible, thank you for your patience and support, i really appreciate it, have a great day ahead, take care and stay safe, bye for now, see you soon, goodbye, farewell, adios, ciao, au revoir, auf wiedersehen, sayonara, shalom, peace out, cheers, hasta la vista, catch you later, keep in touch, keep smiling, keep shining"
encoded= encode(sentence)
decoded = decode(encoded)

print(f"original sentence: {sentence} \n")
print(f"sentence length: {len(sentence)} \n")
print(f"encoded length: {len(encoded)} \n")
print(f"decoded sentence: {decoded} \n")
