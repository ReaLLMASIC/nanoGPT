# Ternary Trit Packing

This repository provides a Python implementation to efficiently pack and unpack
ternary values (-1, 0, 1) into bytes. It encodes 5 trits into a single byte,
achieving near-optimal compression and enabling SIMD-friendly unpacking without
division or modulo operations.

## Credits

This work is inspired by a technique described by [Compilade
(2024-06-26)](https://github.com/ggerganov/llama.cpp/pull/8151) and related
discussions around efficient quantization for LLM weights (BitNet b1.58). The
original concept and code snippets were dedicated to the public domain (CC0).

## License

This reimplementation and all code in this directory is released under the [Apache 2.0 License](LICENSE).

