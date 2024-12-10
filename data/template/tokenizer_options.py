import os
import pickle
import tempfile
import numpy as np
import sentencepiece as spm
import tiktoken
from tqdm import tqdm  # For progress bars
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, args):
        self.args = args

    def tokenize(self, data):
        raise NotImplementedError("Tokenize method must be implemented by subclasses.")

    def detokenize(self, ids):
        raise NotImplementedError("Detokenize method must be implemented by subclasses.")

    def save_meta(self, meta):
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @staticmethod
    def get_key_from_meta(keyname):
        meta_path = 'meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                return meta.get(keyname)
        return None


class NumericRangeTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        self.min_token = args.min_token
        self.max_token = args.max_token
        self.stoi = None
        self.itos = None

    def tokenize(self, data):
        tokens = []
        encountered_tokens = set()
        lines = data.strip().split('\n')
        for line in tqdm(lines, desc="Tokenizing Numeric Range"):
            try:
                num = int(line)
                if self.min_token <= num <= self.max_token:
                    tokens.append(num)
                    encountered_tokens.add(num)
                else:
                    print(f"Warning: Number {num} is outside the specified range and will be skipped.")
            except ValueError:
                print(f"Warning: Invalid number '{line}' will be skipped.")

        all_tokens = list(range(self.max_token, -1, -1))
        self.stoi = {str(num): i for i, num in enumerate(all_tokens)}
        self.itos = {i: str(num) for i, num in enumerate(all_tokens)}

        indexed_tokens = [self.stoi[str(token)] for token in tokens]
        meta = {
            "vocab_size": len(self.stoi),
            "tokenizer": "numeric_range",
            "min_token": self.min_token,
            "max_token": self.max_token,
            "stoi": self.stoi,
            "itos": self.itos,
            "encountered_tokens": sorted(encountered_tokens, reverse=True)
        }
        self.save_meta(meta)
        return indexed_tokens

    def detokenize(self, ids):
        return '\n'.join([self.itos[id] for id in ids])


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, args, input_files=None):
        super().__init__(args)
        self.vocab_size = args.vocab_size
        self.spm_model_file = args.spm_model_file
        self.spm_vocab_file = args.spm_vocab_file
        self.skip_tokenization = args.skip_tokenization
        self.input_files = input_files
        self.sp = None

        if self.spm_model_file:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.spm_model_file)
        elif input_files:
            self.sp = self.train_sentencepiece_model()

    def train_sentencepiece_model(self):
        spm_model_prefix = "trained_spm_model"
        num_threads = os.cpu_count()
        input_arg = ""
        if isinstance(self.input_files, list):
            with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
                for input_file in self.input_files:
                    with open(input_file, "r") as infile:
                        tmpfile.write(infile.read())
                input_arg = tmpfile.name
        else:
            input_arg = self.input_files

        spm.SentencePieceTrainer.train(
            num_threads=num_threads,
            user_defined_symbols="\n, ",
            input=input_arg,
            model_prefix=spm_model_prefix,
            split_digits=True,
            vocab_size=self.vocab_size,
            model_type="bpe",
        )
        print("SentencePiece model training complete.")

        if isinstance(self.input_files, list):
            os.remove(input_arg)

        sp = spm.SentencePieceProcessor()
        sp.load(f"{spm_model_prefix}.model")
        return sp

    def tokenize(self, data):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        ids = self.sp.encode_as_ids(data)
        stoi = {self.sp.id_to_piece(id): id for id in range(self.sp.GetPieceSize())}
        itos = {id: self.sp.id_to_piece(id) for id in range(self.sp.GetPieceSize())}

        meta = {
            "vocab_size": self.sp.GetPieceSize(),
            "tokenizer": "sentencepiece",
            "stoi": stoi,
            "itos": itos,
        }
        self.save_meta(meta)
        return ids

    def detokenize(self, ids):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        return self.sp.decode_ids(ids)


class TiktokenTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        self.tiktoken_encoding = args.tiktoken_encoding
        self.enc = tiktoken.get_encoding(self.tiktoken_encoding)
        self.vocab_size = self.enc.n_vocab

    def tokenize(self, data):
        ids = self.enc.encode_ordinary(data)
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": self.tiktoken_encoding,
        }
        self.save_meta(meta)
        return ids

    def detokenize(self, ids):
        return self.enc.decode(ids)


class CustomTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        if args.tokens_file is None:
            raise ValueError("Tokens file must be provided for custom tokenization method.")
        with open(args.tokens_file, "r") as f:
            self.tokens = [line.strip() for line in f.readlines() if line.strip()]
            self.tokens = [token.replace("\\n", "\n").replace("\\t", "\t") for token in self.tokens]
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for i, token in enumerate(self.tokens)}

    def tokenize(self, data):
        encoded_data = []
        i = 0
        covered_chars = 0
        data_len = len(data)
        pbar = tqdm(total=data_len, desc="Tokenizing Custom Tokens")
        while i < data_len:
            matched = False
            for token in self.tokens:
                token_len = len(token)
                if data.startswith(token, i):
                    encoded_data.append(self.stoi[token])
                    i += token_len
                    covered_chars += token_len
                    pbar.update(token_len)
                    matched = True
                    break
            if not matched:
                i += 1  # Skip character if no token matches
                pbar.update(1)
        pbar.close()
        coverage = covered_chars / data_len
        print(f"Data coverage by tokens: {coverage*100:.2f}%")
        meta = {"vocab_size": len(self.tokens), "stoi": self.stoi, "itos": self.itos}
        self.save_meta(meta)
        return encoded_data

    def detokenize(self, ids):
        return ''.join([self.itos[id] for id in ids])

class CharTokenizer(Tokenizer):
    def __init__(self, args, train_data, val_data):
        super().__init__(args)
        self.reuse_chars = args.reuse_chars
        if self.reuse_chars:
            self.chars = self.get_key_from_meta('chars')
            if self.chars is None:
                raise ValueError("No chars found in meta.pkl. Cannot reuse chars.")
        else:
            self.chars = sorted(list(set(train_data + (val_data if val_data else ""))))
            print(f"All unique characters: {''.join(self.chars)}")
            print(f"Vocab size: {len(self.chars)}")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def tokenize(self, data):
        data_len = len(data)
        ids = []
        pbar = tqdm(total=data_len, desc="Tokenizing Characters")
        for ch in data:
            ids.append(self.stoi[ch])
            pbar.update(1)
        pbar.close()
        meta = {"vocab_size": len(self.chars), "itos": self.itos, "stoi": self.stoi, "chars": self.chars}
        self.save_meta(meta)
        return ids

    def detokenize(self, ids):
        return ''.join([self.itos[id] for id in ids])


class CustomCharTokenizerWithByteFallback(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        if args.custom_chars_file is None:
            raise ValueError("Custom characters file must be provided for this tokenizer.")
        with open(args.custom_chars_file, "r", encoding="utf-8") as f:
            self.custom_chars = [line.strip() for line in f if line.strip()]

        # Build vocab
        self.build_vocab()

    def build_vocab(self):
        # Assign IDs to custom characters
        self.stoi = {ch: i for i, ch in enumerate(self.custom_chars)}
        self.itos = {i: ch for i, ch in enumerate(self.custom_chars)}
        self.custom_char_count = len(self.custom_chars)

        # Assign IDs to bytes (0-255)
        self.byte_stoi = {byte: i + self.custom_char_count for i, byte in enumerate(range(256))}
        self.byte_itos = {i + self.custom_char_count: byte for i, byte in enumerate(range(256))}

        # Update total vocab size
        self.vocab_size = self.custom_char_count + 256  # 256 bytes

        # Merge the dictionaries for easy lookup
        self.stoi.update(self.byte_stoi)
        self.itos.update(self.byte_itos)

        # Save meta information
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "custom_char_with_byte_fallback",
            "custom_chars": self.custom_chars,
            "stoi": self.stoi,
            "itos": self.itos,
            "custom_char_count": self.custom_char_count,
        }
        self.save_meta(meta)

    def tokenize(self, data):
        ids = []
        data_len = len(data)
        pbar = tqdm(total=data_len, desc="Tokenizing with Byte Fallback")
        for ch in data:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                # Byte fallback
                byte_sequence = ch.encode('utf-8')
                for byte in byte_sequence:
                    ids.append(self.stoi[byte])
            pbar.update(1)
        pbar.close()
        return ids

    def detokenize(self, ids):
        chars = []
        byte_buffer = []
        for id in ids:
            if id < self.custom_char_count:
                # It's a custom character
                chars.append(self.itos[id])
            else:
                # It's a byte
                byte_buffer.append(self.itos[id])
                # Check if the next token is not a byte or if it's the last token
                if (len(byte_buffer) > 0 and
                    (len(chars) + len(byte_buffer) == len(ids) or
                     ids[ids.index(id) + 1] < self.custom_char_count)):
                    # Convert byte buffer to character
                    byte_array = bytes(byte_buffer)
                    chars.append(byte_array.decode('utf-8', errors='replace'))
                    byte_buffer = []
        return ''.join(chars)

class Qwen2Tokenizer(Tokenizer):
    def __init__(self, args):
        """
        Initialize the Qwen2Tokenizer using Hugging Face's AutoTokenizer.
        """
        super().__init__(args)
        self.huggingface_model_name = f"Qwen/Qwen2-{args.qwen2_model.split('_')[1].replace('p', '.').upper()}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name)

        # Save vocab size and other meta information
        self.vocab_size = self.tokenizer.vocab_size
        self.special_tokens = self.tokenizer.special_tokens_map

    def tokenize(self, data):
        print(f"Tokenizing data of size: {len(data)}")
        chunk_size = 1024
        ids = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            ids.extend(self.tokenizer.encode(chunk, add_special_tokens=True))
        print(f"Generated {len(ids)} token IDs.")
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "qwen2",
            "qwen2_model": self.huggingface_model_name,
            "special_tokens": self.special_tokens,
        }
        self.save_meta(meta)
        return ids

    def detokenize(self, ids):
        """
        Detokenize token IDs into a string.
        Args:
            ids (List[int]): List of token IDs to convert back to text.

        Returns:
            str: Decoded string.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)