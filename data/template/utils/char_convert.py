import string
import argparse
import re
import spacy
from tqdm import tqdm
from pathlib import Path

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

TOKENS_FILE = "tokensfile.txt" # methods will write their alphabet here

def emit_tokenlist(tokens):
    """
    Write *tokens* (iterable of single-char strings) to *TOKENS_FILE*.
    """

    Path(TOKENS_FILE).write_text(
        ''.join(tokens),
        encoding="utf-8",
    )

def transform_cvp(text):
    """
    The original basic transformation:
      - Punctuation -> '1'
      - Vowels -> '2'
      - Consonants -> '3'
      - Everything else -> '_'
    """
    punctuation = string.punctuation
    vowels = 'aeiouAEIOU'
    consonants = ''.join([c for c in string.ascii_letters if c not in vowels])

    transformed = []
    for char in text:
        if char in punctuation:
            transformed.append('1')
        elif char in vowels:
            transformed.append('2')
        elif char in consonants:
            transformed.append('3')
        else:
            # Whitespace, digits, etc.
            transformed.append('_')

    result = ''.join(transformed)

    # Save tokenlist
    emit_tokenlist(["1", "2", "3", "_"])

    return result


def transform_part_of_speech(text):
    """
    Use spaCy to transform text based on Part-of-Speech (POS):

    Table of POS -> single-char:
    -----------------------------------
    ADJ   -> 'a'     (adjective)
    ADP   -> 'b'     (adposition: in, to, during)
    ADV   -> 'c'     (adverb)
    AUX   -> 'd'     (auxiliary verb: is, has, will)
    CONJ  -> 'e'     (conjunction)
    CCONJ -> 'f'     (coordinating conjunction: and, or, but)
    DET   -> 'g'     (determiner: the, a, an)
    INTJ  -> 'h'     (interjection: oh, wow, hey)
    NOUN  -> 'i'     (common noun)
    NUM   -> 'j'     (numeral: one, 2017)
    PART  -> 'k'     (particle: 's, not)
    PRON  -> 'l'     (pronoun: I, you, she)
    PROPN -> 'm'     (proper noun: London, Mary)
    PUNCT -> 'n'     (punctuation: . , ? ! )
    SCONJ -> 'o'     (subordinating conjunction: if, that)
    SYM   -> 'p'     (symbol: $, %, ©, 😝)
    VERB  -> 'q'     (verb)
    X     -> 'r'     (other, e.g. misclassified tokens)
    SPACE -> (special handling)

    Special replacements:
      - Any space character ' ' -> '_'
      - Any newline '\n'       -> '\n'
      - Anything not recognized -> 'x'

    We ensure the output has the **same number of characters** as the input,
    even for large files, by processing in chunks.
    """

    # Mapping from spaCy POS to single character
    pos_map = {
        "ADJ":   'a',
        "ADP":   'b',
        "ADV":   'c',
        "AUX":   'd',
        "CONJ":  'e',
        "CCONJ": 'f',
        "DET":   'g',
        "INTJ":  'h',
        "NOUN":  'i',
        "NUM":   'j',
        "PART":  'k',
        "PRON":  'l',
        "PROPN": 'm',
        "PUNCT": 'n',
        "SCONJ": 'o',
        "SYM":   'p',
        "VERB":  'q',
        "X":     'r',
        # 'SPACE' handled separately
    }

    # We'll store the transformed result as a list of characters (one per input char)
    result = list(text)
    text_length = len(text)

    # Helper function: yield (chunk_of_text, chunk_start_index)
    # so that each chunk is below spaCy's nlp.max_length
    def chunk_text(full_text, chunk_size):
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i : i + chunk_size], i

    # Decide a chunk size that is safely below nlp.max_length
    # (We use some margin; you can adjust as needed.)
    chunk_size = min(nlp.max_length, 1000000)

    # Process each chunk separately and fill in `result`
    for chunk_str, chunk_start in chunk_text(text, chunk_size):
        doc = nlp(chunk_str)
        # For each token in this chunk:
        for token in tqdm(doc, desc="Transforming chunk", leave=False):
            start_i = chunk_start + token.idx  # offset in the *full* text
            token_len = len(token.text)

            if token.is_space:
                # Replace each whitespace character
                for offset in range(token_len):
                    c = token.text[offset]
                    if c == '\n':
                        result[start_i + offset] = '\n'
                    else:
                        result[start_i + offset] = '_'
            else:
                # Non-whitespace token
                mapped_char = pos_map.get(token.pos_, 'x')
                for offset in range(token_len):
                    c = token.text[offset]
                    if c == '\n':
                        result[start_i + offset] = '\\n'
                    else:
                        result[start_i + offset] = mapped_char

    transformed_text = ''.join(result)

    # emit token list for this method
    emit_tokenlist(sorted(set(pos_map.values()) | {" ", "_", "\n"}))

    return transformed_text

# Helper for transform_in_word_position
def build_position_chars(max_positions: int = 64) -> str:
    """
    Return exactly *max_positions* distinct characters suitable for one-char
    positional placeholders.

    Starts with:
        1–9, A–Z, a–z  (total 61)
    then continues through the Unicode range beginning at U+00A1, adding every
    printable, non-whitespace code point until the requested length is met.
    """
    base = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    if max_positions <= len(base):
        return "".join(base[:max_positions])

    cp = 0x00A1  # first printable char after Latin-1 controls/space
    while len(base) < max_positions:
        ch = chr(cp)
        if ch.isprintable() and not ch.isspace():
            base.append(ch)
        cp += 1
    return "".join(base)


def transform_in_word_position(
    text: str,
    max_positions: int = 64,
    token_file: str | Path = TOKENS_FILE,
) -> str:
    """
    Encode every non-whitespace “word” in *text* by replacing its characters with
    a single symbol that indicates 1-based position (wrapping modulo
    *max_positions*).  Whitespace characters are mapped to '_' so the original
    spacing remains visually evident.

    A file named *token_file* is (re)written containing the full set of possible
    symbols—one per line—in the exact order used for encoding.

    Parameters
    ----------
    text : str
        The source text to transform.
    max_positions : int, default 64
        The size of the lookup table before positions wrap.
    token_file : str or Path, default "tokenlist.txt"
        Destination path for the emitted token list.

    Returns
    -------
    str
        The transformed text.
    """
    # Build (or retrieve cached) lookup string
    position_chars: str = build_position_chars(max_positions)
    max_idx: int = len(position_chars)  # might differ if user supplied low N

    # ── Emit tokenlist.txt ────────────────────────────────────────────────────
    #   Each symbol → its own line, final newline added for POSIX friendliness.
    emit_tokenlist(["_"] + list(position_chars))

    # ── Transform the input text ─────────────────────────────────────────────
    # Split so that whitespace chunks are preserved and re-encoded explicitly.
    chunks = re.split(r"(\s+)", text)
    encoded: list[str] = []

    for chunk in chunks:
        if chunk == "":
            continue  # artefact of split
        if chunk.isspace():
            # Keep spacing, but show as underscores for visibility
            encoded.append("_" * len(chunk))
        else:
            # Replace each char with its position marker
            out = [
                position_chars[(i - 1) % max_idx]  # 1-based index, wrap modulo
                for i, _ in enumerate(chunk, start=1)
            ]
            encoded.append("".join(out))

    return "".join(encoded)

def transform_position_since_newline(
    text: str,
    max_positions: int = 64,
    token_file: str | Path = TOKENS_FILE,
) -> str:
    """
    Encode every non-newline character with a marker that represents its
    1-based column index *since the last newline* (wrapping modulo
    *max_positions*).  Newlines are left as-is and reset the column counter.

    Other whitespace (spaces, tabs) becomes '_' to preserve layout visibility.
    """
    position_chars: str = build_position_chars(max_positions)
    max_idx: int = len(position_chars)

    # Emit (or overwrite) tokenlist.txt
    emit_tokenlist(["_", "\n"] + list(position_chars))

    out: list[str] = []
    col: int = 0  # column index, 0 before first char

    for ch in text:
        if ch == "\n":
            out.append("\n")
            col = 0  # reset at newline
        elif ch.isspace():
            out.append("_")
            col += 1
        else:
            col += 1
            out.append(position_chars[(col - 1) % max_idx])

    return "".join(out)


def transform_file(filename, method, max_positions):
    """
    Transforms a file in-place using the selected method.
    """
    try:
        with open(filename, 'r+', encoding='utf-8') as file:
            # Read the entire file content
            file_content = file.read()

            if method == 'cvp':
                transformed_content = transform_cvp(file_content)
            elif method == 'part_of_speech':
                transformed_content = transform_part_of_speech(file_content)
            elif method == 'in_word_position':
                transformed_content = transform_in_word_position(
                    file_content, max_positions=max_positions
                )
            elif method == 'since_newline':
                transformed_content = transform_position_since_newline(
                    file_content, max_positions=max_positions
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Overwrite file with transformed text
            file.seek(0)
            file.write(transformed_content)
            file.truncate()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a text file by replacing characters.")
    parser.add_argument("input_file", help="The input text file to transform.")
    parser.add_argument(
        "--method", 
        choices=["cvp", "part_of_speech", "in_word_position", "since_newline"],
        default="cvp",
        help="Which transformation method to use."
    )
    args = parser.add_argument(
        "--max-positions",
        type=int,
        default=64,
        help="Maximum distinct position markers before wrapping (used by "
             "`in_word_position` and `since_newline`).",
    )
    args = parser.parse_args()
    transform_file(args.input_file, args.method, args.max_positions)

