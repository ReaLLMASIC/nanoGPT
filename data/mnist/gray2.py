from PIL import Image
import argparse
import os
import sys
import numpy as np


def convert_image_to_ascii(image_path, output_size, grayscale, levels, chars=None):
    """Convert a single image into ASCII art and return it as a multi‑line string."""
    # Load and resize the image
    img = Image.open(image_path).resize(output_size, Image.LANCZOS)

    if grayscale:
        img = img.convert("L")  # Convert to grayscale
        if levels < 256:
            # Quantise grayscale into <levels> buckets
            img = img.point(lambda p: (p * levels) // 256 * (256 // levels))

    # Fallback/default char lists for common level counts
    if chars is None:
        if levels == 2:
            chars = "@."
        elif levels == 3:
            chars = "@*-"
        elif levels == 4:
            chars = "@#+-"
        elif levels == 5:
            chars = "@#+-."
        elif levels == 6:
            chars = "@#+-:."
        elif levels == 7:
            chars = "@#*+-:."
        elif levels == 8:
            chars = "@%#*+-:."
        else:
            sys.exit(f"number of levels {levels} not supported")

    # Map pixel values → characters
    char_array = np.array(list(chars))
    scale_factor = 256 // levels
    img_np = np.array(img)
    ascii_img = char_array[img_np // scale_factor]

    return "\n".join("".join(row) for row in ascii_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to ASCII art.")
    parser.add_argument("--output-dimensions", type=str, default="16x16",
                        help="Output dimensions for ASCII art, e.g., 8x8 or 16x16")
    parser.add_argument("--levels", type=int, default=2,
                        help="Number of grayscale levels (2–9 supported)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images to convert")
    parser.add_argument("--output-dir", type=str, default="grayscale_images",
                        help="Directory to save per‑image ASCII art files")

    # Appending behaviour
    parser.add_argument("--append-to-file", action="store_true",
                        help="Append ASCII art (and token maps) to consolidated files")
    parser.add_argument("--output-file", type=str, default="input.txt",
                        help="Consolidated ASCII art file")
    parser.add_argument("--row-file", type=str, default="input_row.txt",
                        help="Row‑index tokens output file")
    parser.add_argument("--column-file", type=str, default="input_column.txt",
                        help="Column‑index tokens output file")
    parser.add_argument("--abs-file", type=str, default="input_abs.txt",
                        help="Absolute‑index tokens output file")
    parser.add_argument("--number-placement", type=str, default="before",
                        choices=["before", "after"],
                        help="Put file‑derived label before or after ASCII art block")

    parser.add_argument("--chars", type=str, default=None,
                        help="Custom darkest→lightest character ramp")

    args = parser.parse_args()

    # ───────── basic setup ──────────
    output_dimensions = tuple(map(int, args.output_dimensions.split("x")))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.append_to_file:
        out_f = open(args.output_file, "a")
        row_f = open(args.row_file, "a")
        col_f = open(args.column_file, "a")
        abs_f = open(args.abs_file, "a")
    else:
        out_f = row_f = col_f = abs_f = None

    # ───────── helper tables ──────────
    TOKEN_CHARS = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
    )

    def int_to_char(i: int) -> str:
        """Return a unique printable (non‑newline) character for index *i*."""
        if i < len(TOKEN_CHARS):
            return TOKEN_CHARS[i]
        return chr(0x100 + i - len(TOKEN_CHARS))

    def build_row_tokens(lines):
        return "\n".join(int_to_char(r) * len(line) for r, line in enumerate(lines))

    def build_col_tokens(lines):
        return "\n".join("".join(int_to_char(c) for c in range(len(line))) for line in lines)

    def build_abs_tokens(lines):
        acc = 0
        blocks = []
        for line in lines:
            blocks.append("".join(int_to_char(acc + i) for i in range(len(line))))
            acc += len(line)
        return "\n".join(blocks)

    # ───────── main loop ──────────
    for fn in os.listdir(args.image_dir):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            continue

        img_path = os.path.join(args.image_dir, fn)
        ascii_art = convert_image_to_ascii(
            img_path,
            output_size=output_dimensions,
            grayscale=True,
            levels=args.levels,
            chars=args.chars,
        )

        if args.append_to_file:
            label = os.path.splitext(fn)[0].split("_")[-1]

            lines = ascii_art.split("\n")
            row_map = build_row_tokens(lines)
            col_map = build_col_tokens(lines)
            abs_map = build_abs_tokens(lines)

            if args.number_placement == "before":
                out_f.write(f"{label}\n{ascii_art}\n")
                row_f.write(f"{'_' * len(label)}\n{row_map}\n")
                col_f.write(f"{'_' * len(label)}\n{col_map}\n")
                abs_f.write(f"{'_' * len(label)}\n{abs_map}\n")
            else:  # after
                out_f.write(f"{ascii_art}\n{label}\n")
                row_f.write(f"{row_map}\n{'_' * len(label)}\n")
                col_f.write(f"{col_map}\n{'_' * len(label)}\n")
                abs_f.write(f"{abs_map}\n{'_' * len(label)}\n")
        else:
            out_path = os.path.join(
                args.output_dir, os.path.splitext(fn)[0] + ".txt"
            )
            with open(out_path, "w") as f_out:
                f_out.write(ascii_art)
            print(f"ASCII art saved to {out_path}")

    # ───────── cleanup ──────────
    for fp in (out_f, row_f, col_f, abs_f):
        if fp:
            fp.close()

