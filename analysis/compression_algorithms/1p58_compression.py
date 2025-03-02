import numpy as np
from rich import print
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from typing import Union


def encode_five_trits_to_byte(five_trit_val: int) -> int:
    """
    Convert a 5-trit integer (range [0..242]) into a single byte [0..255].

    This scaling involves a ratio of 256/243. Since integer division truncates,
    we add (243 - 1) to the numerator before dividing to effectively round up,
    preserving a consistent mapping from 5-trit values to bytes.
    """
    # Note: 243 = 3^5, covering all combinations of 5 trits (0..242)
    numerator = five_trit_val * 256 + (243 - 1)
    encoded_byte = numerator // 243
    return encoded_byte

def decode_byte_to_five_trits(encoded_byte: int) -> tuple[int, int]:

    """
    Reverse the scaling done by encode_five_trits_to_byte.

    Given a byte representing 5 packed trits, we think of it as a fraction less than 1.
    Multiplying by 3 shifts digits upwards. By examining the upper bits after scaling,
    we can extract each trit in sequence.

    However, this function only demonstrates the extraction logic for one iteration.
    The full unpacking is done by repeatedly applying a similar step and isolating bits.
    """
    # When unpacking, we multiply by 3 and then look at the high bits.
    # This is done repeatedly outside this function to extract all five trits.
    # Here we simply show the conceptual step:
    temp = encoded_byte * 3
    # The top bits reveal the trit:
    trit = temp >> 8
    # Update the byte for the next iteration:
    next_byte = temp & 0xFF
    return trit, next_byte

def pack_trits(digits: np.ndarray) -> bytearray:
    """
    Pack a numpy array of ternary values (-1, 0, 1) into bytes.
    Each group of 5 trits maps onto one byte.

    Steps:
    1. Convert each trit from {-1,0,1} to {0,1,2}.
    2. Combine each group of 5 ternary digits into a single base-3 integer (0..242).
    3. Scale that integer into a byte [0..255] using encode_five_trits_to_byte.
    """
    assert len(digits) % 5 == 0, "Input length must be multiple of 5."
    n_groups = len(digits) // 5

    packed = bytearray()
    for i in range(n_groups):
        group_val = 0
        # Aggregate 5 trits into a base-3 number:
        for j in range(5):
            trit = int(digits[5*i + j])
            # Map -1,0,1 to 0,1,2
            normalized = (max(-1, min(trit, 1))) + 1
            group_val = group_val * 3 + normalized

        # Convert this 0..242 range value into one byte.
        encoded = encode_five_trits_to_byte(group_val)
        packed.append(encoded)

    return packed

from typing import Union

def unpack_trits(packed: Union[bytes, bytearray]) -> np.ndarray:
    """
    Unpack bytes back into trits (-1,0,1).

    We reverse the packing:
    - Each byte represents 5 trits.
    - Repeatedly multiply by 3, extract the top bits for the current trit,
      and keep the remaining bits for the next extraction.
    """
    trits = []
    for enc_byte in packed:
        b = enc_byte
        for _ in range(5):
            # Multiply by 3 to shift and extract the high bits as a trit:
            temp = b * 3
            extracted_trit = (temp >> 8) - 1  # map 0,1,2 back to -1,0,1
            trits.append(extracted_trit)
            # Keep lower 8 bits for next iteration
            b = temp & 0xFF
    return np.array(trits, dtype=np.int8)

def print_comparison(original: np.ndarray, packed: bytearray):
    """
    Display a table comparing original and packed sizes, along with compression ratio.
    """
    original_size = len(original)  # each original trit as int8 is 1 byte
    packed_size = len(packed)

    compression_ratio = original_size / packed_size if packed_size else 1
    compression_percent = (packed_size / original_size * 100) if original_size else 100

    table = Table(title="Size Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Data", justify="center", style="cyan", no_wrap=True)
    table.add_column("Size (bytes)", justify="right", style="green")
    table.add_column("Ratio", justify="right", style="yellow")

    table.add_row("Original", str(original_size), "-")
    table.add_row("Packed", str(packed_size), f"~{compression_percent:.2f}% of original")
    print(table)

    summary = f"[bold]Compression Ratio:[/bold] {compression_ratio:.2f}\n[bold]Compressed Percentage:[/bold] {compression_percent:.2f}%"
    print(Panel(summary, title="Summary", subtitle="Compression Details", border_style="bright_blue"))

def main():
    console = Console()

    # Example usage
    np.random.seed(42)
    length = 10 * 5
    data = np.random.randint(-1, 2, size=length, dtype=np.int8)

    console.rule("[bold green]Original Data")
    print("[bold]Original Data (length={}):[/bold]".format(len(data)))
    print(data)

    packed = pack_trits(data)
    console.rule("[bold green]Packed Bytes")
    print("[bold]Packed Bytes (length={}):[/bold]".format(len(packed)))
    print(list(packed))

    unpacked = unpack_trits(packed)
    console.rule("[bold green]Unpacked Data")
    print("[bold]Unpacked Data (length={}):[/bold]".format(len(unpacked)))
    print(unpacked)

    # Check correctness
    correct = np.array_equal(data, unpacked)
    print("\n[bold]Verification:[/bold]", "[green]PASS[/green]" if correct else "[red]FAIL[/red]")

    # Print comparison of sizes
    console.rule("[bold green]Comparison")
    print_comparison(data, packed)

    # Spacing
    print()
    print()

    # Known pattern test
    small_data = np.array([-1, 0, 1, 1, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    console.rule("[bold green]Again With a Known Pattern")
    print("[bold]Original Small Data (length={}):[/bold]".format(len(small_data)))
    print(small_data)

    small_packed = pack_trits(small_data)
    print("\n[bold]Packed Small Data (length={}):[/bold]".format(len(small_packed)))
    print(list(small_packed))

    small_unpacked = unpack_trits(small_packed)
    print("[bold]Unpacked Small Data (length={}):[/bold]".format(len(small_unpacked)))
    print(small_unpacked)

    # Check correctness for the small example
    correct_small = np.array_equal(small_data, small_unpacked)
    print("\n[bold]Verification (small):[/bold]", "[green]PASS[/green]" if correct_small else "[red]FAIL[/red]")

    # Print comparison for small data
    console.rule("[bold green]Small Data Comparison")
    print_comparison(small_data, small_packed)


if __name__ == "__main__":
    main()


