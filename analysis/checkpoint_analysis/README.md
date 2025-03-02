# Model Parameter Exploration

These are utility scripts designed to inspect and navigate through the parameter
tree of a trained GPT model checkpoint. It allows users to explore the hierarchy
of model parameters interactively and inspect their values.

## Features

- Load a GPT model checkpoint.
- Display the hierarchical structure of the model's parameters.
- Interactively navigate through the parameter tree.
- View parameter tensor values.

## Requirements

- Python 3.x
- PyTorch
- A valid GPT checkpoint file (e.g., `out/ckpt.pt`).

## Usage

### Running the Script

This script must be executed from the **main directory of the repository**. The
typical path to a checkpoint file is `out/ckpt.pt`.

```bash
python checkpoint_analysis/checkpoint_explorer.py <ckpt_path> [--device <device>]
```

### Positional Arguments

- `<ckpt_path>`: The path to the model checkpoint file (e.g., `out/ckpt.pt`).

### Optional Arguments

- `--device`: The device to load the model on (`cpu` by default, e.g., `cuda` for GPU).

### Example Command

```bash
python checkpoint_analysis/checkpoint_explorer.py out/ckpt.pt --device cuda
```

### Interactive Navigation

1. **Start at Root**: The script begins at the root of the parameter tree.
2. **Explore Submodules**: Use numbers to explore submodules or parameters.
3. **Go Back**: Enter `b` to go back to the previous level.
4. **Quit**: Enter `q` to exit the script.
5. **View Parameter Values**: When selecting a parameter (leaf node), its tensor value will be displayed. Press `Enter` to return to the previous level.

### Notes

- The script processes the checkpoint to ensure compatibility by renaming keys starting with `_orig_mod.`.
- Parameter tensor values longer than 1000 characters will be truncated in the display for readability.

## Troubleshooting

- Ensure the script is run from the **main repository directory**.
- Verify that the checkpoint file exists at the specified path.
- Check for dependencies like PyTorch before running the script.
