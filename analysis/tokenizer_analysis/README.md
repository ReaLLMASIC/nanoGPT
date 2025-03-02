# Open Source CJK Analysis Tool

This repository contains the **Open Source CJK Analysis Tool**, a Python script (`open_source_cjk_analysis.py`) that provides detailed analysis of tokenizers with respect to Chinese (C), Japanese (J), and Korean (K) characters. The tool can analyze token coverage, symbol representation, and overlaps among these languages in tokenizer vocabularies.

## Features

- **Subcategory Analysis**: 
  - For each subcategory (e.g., Hiragana, Unified Ideographs, etc.), the script calculates:
    - Total tokens in the vocabulary containing characters from the subcategory.
    - Number of unique symbols (characters) found in the vocabulary.
    - Total possible characters in the subcategory.
    - Percentage of symbols represented in the vocabulary.
    
- **Total Tokens in Categories**:
  - Calculates the total number of tokens falling into any of the C, J, or K categories without double-counting.

- **Overlap Analysis**:
  - Analyzes overlaps between the C, J, and K categories, providing counts for:
    - Tokens in multiple categories (e.g., CJ, CK, JK, CJK).

## Requirements

### Python Libraries

- `transformers`
- `rich`
- `tiktoken`

Install dependencies using:
```bash
pip install transformers rich tiktoken
```

## Usage

1. Place the `open_source_cjk_analysis.py` script in your working directory.
2. Run the script using Python:
   ```bash
   python open_source_cjk_analysis.py
   ```

3. The script analyzes the tokenizers specified in the `tokenizers` list:
   ```python
   tokenizers = [
       {"name": "google/gemma-7b", "is_tiktoken": False},
       {"name": "o200k_base", "is_tiktoken": True},
       {"name": "mistralai/Mistral-7B-Instruct-v0.3", "is_tiktoken": False},
   ]
   ```
   You can modify this list to include additional tokenizers or replace existing ones.

## Output

The script provides the following outputs:

### 1. **Subcategory Analysis Table**

| Language | Subcategory       | Token Count | Unique Symbols Found | Total Possible Characters | % of Symbols in Range |
|----------|-------------------|-------------|-----------------------|---------------------------|------------------------|
| C        | Unified Ideographs| 1234        | 892                  | 20000                     | 4.46%                 |
| ...      | ...               | ...         | ...                  | ...                       | ...                   |
| **TOTAL**|                   | 5678        | 2345                 | 67890                     | 12.34%                |

### 2. **Total Tokens in Categories Table**

| Category | Token Count |
|----------|-------------|
| C        | 4567        |
| J        | 1234        |
| K        | 890         |
| **TOTAL (Any C, J, K)** | 5678 |

### 3. **Overlap Analysis Table**

| Overlap  | Token Count |
|----------|-------------|
| CJ       | 456         |
| CK       | 234         |
| JK       | 123         |
| CJK      | 45          |

### 4. **Summary**

- Total tokens in the tokenizer: `256000`
- Total tokens in any C, J, or K category (no double-counting): `5678`.

## Contributing

Contributions to improve the analysis or add features are welcome! Please submit a pull request or open an issue if you encounter any problems.
