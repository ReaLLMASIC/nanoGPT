import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Olmo with custom attention.")
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["softmax", "relu", "relu_norm", "softplus", "softplus_norm", "sigmoid", "sigmoid_norm"],
        default="softmax",
        help="Select which attention variant to use."
    )
    parser.add_argument("--clip", default=False, action=argparse.BooleanOptionalAction, help="Whether to clip the model variant.")
    parser.add_argument("--gate", default=False, action=argparse.BooleanOptionalAction, help="Whether to gate the model variant.")
    parser.add_argument(
        "--obo_variant",
        type=str,
        choices=["none", "obo", "learned_obo_per_layer", "learned_obo_per_layer_per_head"],
        default="none",
        help="Specify the off-by-one attention variant: 'none' for no off-by-one, 'obo' for off-by-one, 'learned_obo_per_layer' for learned off-by-one for each head, 'learned_obo_per_layer_per_head' for learned obo for each layer and head."
    )
    parser.add_argument("--quantize", default=False, action=argparse.BooleanOptionalAction, help="Whether to perform Quantize Aware Training.")
    args = parser.parse_args()
    return args