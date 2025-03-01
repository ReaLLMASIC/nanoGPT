# initialization_variations.py

import torch

def direct_init(vocab_size: int, n_embd: int):
    """
    Minimal example of 'direct' hypercube initialization.
    For demonstration, we'll just create 2^n_embd corners
    and then pick the first `vocab_size`.
    """
    n_corners = 2 ** n_embd
    n_corners = min(n_corners, vocab_size)

    if vocab_size > n_corners:
        raise ValueError(
            f"Not enough corners (2^{n_embd}={n_corners}) for vocab_size={vocab_size} in 'direct' mode."
        )
    corners = torch.zeros((n_corners, n_embd))
    for i in range(n_corners):
        for d in range(n_embd):
            corners[i, d] = (i >> d) & 1
    return corners[:vocab_size, :]

def one_hot_init(vocab_size: int, n_embd: int):
    """
    Create a one-hot embedding matrix of shape [vocab_size, n_embd].
    We assert n_embd >= vocab_size so that each row can have exactly one 1
    in a distinct column.
    """
    if n_embd < vocab_size:
        raise ValueError("For 'one-hot' init, n_embd must be >= vocab_size.")
    weight = torch.zeros((vocab_size, n_embd))
    for i in range(vocab_size):
        weight[i, i] = 1.0
    return weight


init_dictionary = {
    "gaussian": None,   # fall back to the default Gaussian in model.py
    "hypercube": direct_init,
    "onehot": one_hot_init,
}

