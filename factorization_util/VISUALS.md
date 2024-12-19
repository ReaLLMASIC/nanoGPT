# `view_npy_heatmap.py`

`view_npy_heatmap.py` is a Python script that allows you to visualize the contents of a `.npy` file as a heatmap using Seaborn. This is particularly useful for inspecting 2D arrays, such as matrices or images, stored in `.npy` files.

## Features

- **Load and visualize 2D numpy arrays** stored in `.npy` files.
- **Customizable colormap** to adjust the heatmap's appearance.
- **Option to save** the heatmap as an image file.
- **Simple command-line interface** for easy usage.

## Prerequisites

Before using the script, make sure you have the following Python packages installed:

- `numpy`
- `seaborn`
- `matplotlib`
- `argparse`

You can install these packages using pip:

```bash
pip install numpy seaborn matplotlib
```

## Usage

### Basic Usage

To visualize a `.npy` file as a heatmap, run the script with the path to the file:

```bash
python view_npy_heatmap.py path/to/yourfile.npy
```

### Customizing the Colormap

You can specify a different colormap using the `--cmap` option. For example, to use the `plasma` colormap:

```bash
python view_npy_heatmap.py path/to/yourfile.npy --cmap plasma
```

### Saving the Heatmap

If you want to save the heatmap as an image file instead of displaying it, use the `--save` option:

```bash
python view_npy_heatmap.py path/to/yourfile.npy --save output.png
```

This will save the heatmap to `output.png`.

### Full Example

To view a heatmap with the `inferno` colormap and save it as `heatmap.png`:

```bash
python view_npy_heatmap.py path/to/yourfile.npy --cmap inferno --save heatmap.png
```

## Supported Data Types

This script currently supports 2D numpy arrays. If you attempt to load a 1D or
higher-dimensional array, the script will display an error.

## License

This script is provided as-is under the MIT License. Feel free to modify and distribute it as needed.
