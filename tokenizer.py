import json
import csv
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_file(filepath="data.txt"):
    data = ""
    
    try:
        with open(filepath, "r",encoding="utf-8") as ri:
            if filepath.endswith(".csv"):
                reader = csv.reader(ri)
                reader.__next__()  # Skip header row if present
                for row in tqdm(reader,desc="Loading CSV data: "):
                    data += " ".join(row) + " "
                print("CSV loading complete")
            else:
                for line in tqdm(ri, desc="Loading text data: "):
                    data += line.strip() + " "
        return data
    except FileNotFoundError:
        print("File not Found, stopping data loader")
        return ""
    except Exception as e:
        raise e

def count_pairs(tokens):
    pairs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in pairs:
            pairs[pair] += 1
        else:
            pairs[pair] = 1
    return pairs

def merge_chunk(tokens, most_common, new_token_id):

    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == most_common:
            merged.append(new_token_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


class BPETokenizer():
    def __init__(self,method=None,data=None,n_vocab=10000):
        if method not in ["train", "encode-decode"]:
            raise ValueError("Method must be either 'train' or 'encode-decode'.")
        
        if data is None and method=="train":
            raise ValueError("Data must be provided for BPE Tokenizer.")
        
        self.data = data
        self.n_vocab = n_vocab

    def train(self):
        data = self.data
        n_vocab = self.n_vocab
        tokens = [ord(a) for a in data if ord(a) < 256]
        vocab = {ord(a): a for a in data if ord(a) < 256}
        starting_index = 256
        merged_pairs = {}
        max_merges = n_vocab

        print("Starting BPE training...")
        pbar = tqdm(total=max_merges, desc="Merging pairs")

        while True:
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            most_common = max(pairs, key=pairs.get)
            new_token_id = starting_index
            merged_pairs[most_common] = new_token_id
            vocab[new_token_id] = most_common
            starting_index += 1
            pbar.update(1)

            # Parallel merge
            n = cpu_count()
            chunk_size = len(tokens) // n + 1
            chunks = [tokens[i:i + chunk_size + 1] for i in range(0, len(tokens), chunk_size)]
            tasks = [(chunk, most_common, new_token_id) for chunk in chunks]

            with Pool(n) as pool:
                results = pool.starmap(merge_chunk, tasks)

            # Stitch chunks
            tokens = []
            for i, chunk in enumerate(results):
                if i > 0 and tokens and chunk:
                    if tokens[-1] == chunk[0]:
                        chunk = chunk[1:]
                tokens.extend(chunk)

            if len(merged_pairs) >= max_merges:
                print(f"\nTokenizer trained with vocab size of {len(vocab)}")
                break

        pbar.close()

        sp_tok_index = max(vocab.keys()) + 1
        special_tokens = {
            "<pad>": sp_tok_index,
            "<unk>": sp_tok_index + 1,
            "<sos>": sp_tok_index + 2,
            "<eos>": sp_tok_index + 3,
            "\n": sp_tok_index + 4,
            "[": sp_tok_index + 5,
            "]": sp_tok_index + 6
        }
        vocab.update({v: k for k, v in special_tokens.items()})

        return vocab



    def encode(self, text):
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {str(v): int(k) for k, v in vocab.items()}
        tokens = []

        def split_text(text):
            return [a for a in text]

        def cluster(splits):
            while True:
                pairs = count_pairs(splits)
                if not pairs:
                    break

                # Get most frequent pair
                most_freq = max(pairs, key=pairs.get)
                merged = str([vocab.get(most_freq[0], vocab["<unk>"]), vocab.get(most_freq[1], vocab["<unk>"])])

                if merged not in vocab:
                    break

                i = 0
                new_splits = []
                while i < len(splits):
                    if i < len(splits) - 1 and (splits[i], splits[i + 1]) == most_freq:
                        new_splits.append(merged)
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                splits = new_splits

            return splits

        splits = split_text(text)
        merged_tokens = cluster(splits)

        for token in merged_tokens:
            if token.startswith("["):
                tokens.append(vocab[token])
            else:
                tokens.append(vocab.get(token, vocab["<unk>"]))

        return tokens

    def decode(self, token_ids):
        with open("vocab.json", "r" ,encoding="utf-8") as f:
            vocab = json.load(f)
        
        # Convert token_id -> token (reverse vocab)
        reverse_vocab = {int(k): v for k, v in vocab.items()}
        decoded=""

        def flatten(token_id):
            flatten_tokens = []
            if isinstance(token_id, list):
                for tid in token_id:
                    flatten_tokens.extend(flatten(tid))
            else:
                val = reverse_vocab.get(token_id, "<unk>")
                if isinstance(val, list):
                    flatten_tokens.extend(flatten(val))
                else:
                    flatten_tokens.append(val)

            return flatten_tokens

        decoded= decoded.join(flatten(token_ids))
        return decoded



class TokenizerTrainer():
    def __init__(self, Filepath,n_vocab=1000):
        self.filepath = Filepath
        self.data= load_file(self.filepath)
        if self.data is None:
            raise ValueError("Data could not be loaded from the file, this is likely due to missing file.")
        self.bpe_tokenizer = BPETokenizer(method="train",data=self.data, n_vocab=n_vocab)
        self.vocab = self.bpe_tokenizer.train()
        self.save_vocab()
    
    def save_vocab(self, filepath="vocab.json"):
        with open(filepath, "w") as f:
            json.dump(self.vocab, f, indent=4)
        print(f"Vocabulary saved to {filepath}")



if __name__ == "__main__":
    Trainer = TokenizerTrainer("data.csv",n_vocab=10000)