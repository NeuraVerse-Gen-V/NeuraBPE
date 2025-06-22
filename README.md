# 🔤 Neura BPE

A custom **Byte Pair Encoding (BPE) Tokenizer** built from scratch in Python for fast and flexible tokenization of text datasets. Supports training a new vocabulary and encoding/decoding using the saved vocab.

---

## 🏦 Features

* Trains on `.txt` or `.csv` datasets
* Multi-process BPE merging for faster training
* Special token support: `<pad>`, `<unk>`, `<sos>`, `<eos>`, `\n`, `[`, `]`
* JSON vocab export
* Easy encode-decode interface

---

## 🚀 Installation

Requires:

```bash
pip install tqdm
```

Place `tokenizer.py` in your project directory.

---

## 🧐 Training the Tokenizer

Train a new vocabulary on your dataset (`data.txt` or `data.csv`):
______________Or______________ 
Simply use the Pre-trained **vocab.json**

```python
from tokenizer import TokenizerTrainer

trainer = TokenizerTrainer("data.csv", n_merges=20000)
```

Vocabulary will be saved as `vocab.json`.

---

## ✍️ Usage Example

```python
import tokenizer as TOKENIZER

def encode(text):
    tokenizer = TOKENIZER.BPETokenizer(method="encode-decode")
    return tokenizer.encode(text)

def decode(token_ids):
    tokenizer = TOKENIZER.BPETokenizer(method="encode-decode")
    return tokenizer.decode(token_ids)

sentence = "Hello there i guess my encoder-decoder finally works..."
encoded = encode(sentence)
decoded = decode(encoded)

print("Original:", sentence)
print("Encoded length:", len(encoded))
print("Decoded:", decoded)
```

---

## 📁 File Structure

* `tokenizer.py` – Main BPE tokenizer and trainer
* `vocab.json` – Saved vocabulary after training
* `data.txt` / `data.csv` – Input training data

---

## ⚙️ Advanced Details

* Uses ASCII values for base tokens (0–255)
* Vocabulary extends using most frequent bigrams
* Parallel merge implemented with Python `multiprocessing`
* Header Row is not skipped so trainer can start from 1st line in case no header was made

---

## 📌 Version

Current version: `v1.0.0`  
(You can increment this as you add features or fix bugs.)

---

## 🔧 Maintenance

This project is currently **maintained** by [NeuraVerse].  
Expect updates, improvements, and potential bug fixes based on feedback and use cases.


## 📜 License

Licensed under the **Apache License 2.0**.
See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

---

## 🧠 Author

Developed by Sylo — Engineering in CSE + AI/ML | YouTuber | Programmer | Gamer | Content Creator
