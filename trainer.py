import json
import csv
from tqdm import tqdm
import regex as re




def load_file(filepath: str = "data.txt"):
    """
    Loads text or CSV data from a file and returns it as a single string.
    
    Args:
        filepath (str): Path to the file.
        
    Returns:
        str: Concatenated data from the file.
    """
    data_parts = []
    try:
        with open(filepath, "r", encoding="utf-8") as ri:
            if filepath.endswith(".csv"):
                reader = csv.reader(ri)
                for row in tqdm(reader, desc="Loading CSV data"):
                    data_parts.append(" ".join(row))
                print("CSV loading complete")
            else:
                # Count total lines for progress bar
                total_lines = sum(1 for _ in open(filepath, "r", encoding="utf-8"))
                ri.seek(0)
                for line in tqdm(ri, desc="Loading text data", total=total_lines):
                    data_parts.append(line.strip())
        return " ".join(data_parts)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Stopping data loader.")
        return ""
    except Exception as e:
        print(f"Error loading file: {e}")
        return ""

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def split_text_with_progress(text, chunk_size=10000):
    # ONLY USED IN TRAINING TO SEE THE VISUAL PROGRESS OF SPLITTING TEXT

    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    words = []
    num_chunks = (len(text) + chunk_size - 1) // chunk_size
    for i in tqdm(range(num_chunks), desc="Splitting text chunks"):
        chunk = text[i*chunk_size:(i+1)*chunk_size]
        words.extend(re.findall(GPT4_SPLIT_PATTERN, chunk))
    return words



class BPETokenizer():
    def __init__(self,data=None,n_vocab=10000):
        if data is None:
            raise ValueError("Data must be provided for BPE Tokenizer.")
        
        self.data = data[:12_500_000] 
        self.n_vocab = n_vocab

    def train(self):
        ids = [list(ch.encode("utf-8")) for ch in tqdm(split_text_with_progress(self.data), desc="Encoding words to utf-8")]
        del self.data  # Free memory after loading data

        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in tqdm(range(self.n_vocab), desc="Training BPE Tokenizer: "):
            stats={}
            for chunk in ids:
                for pair in zip(chunk,chunk[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
            
            if not stats:
                break
            pair= max(stats, key=stats.get)

            idx=256+i

            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[str(pair)] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        #generate tokens from 0 to 256
        tokens = {chr(i):i for i in range(256)}

        tokens.update(merges)

        # Add special tokens
        special_tokens = {
            "<eos>": 256 + len(merges)+1,
            "<pad>": 256 + len(merges)+2,
            "<unk>": 256 + len(merges)+3,
            "<sos>": 256 + len(merges)+4
        }
        tokens.update(special_tokens)
        return tokens
    
class TokenizerTrainer():
    def __init__(self, Filepath,n_vocab=1000):
        self.filepath = Filepath
        self.data= load_file(self.filepath)

        if self.data is None:
            raise ValueError("Data could not be loaded from the file, this is likely due to missing file.")
        print(f"Data loaded from {self.filepath} with length {len(self.data)} characters.")

        self.bpe_tokenizer = BPETokenizer(data=self.data, n_vocab=n_vocab)
        del self.data # Free memory after loading data

        print("Starting training BPE Tokenizer...")

        self.vocab = self.bpe_tokenizer.train()
        self.save_vocab()
    
    def save_vocab(self, filepath="vocab2.json"):
        with open(filepath, "w",encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=4)
        print(f"Vocabulary saved to {filepath}")

if __name__ == "__main__":
    Trainer = TokenizerTrainer("data.txt",n_vocab=50000)