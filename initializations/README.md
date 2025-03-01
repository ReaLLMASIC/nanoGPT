# Initialization Variations

These are alternatives to the conventional guassian normalization.

## One Hot Initialization

When the embedding dimensions are greater than the number of vectors, we can
perform a one-hot initialization.

## Hypercube initialization

This initializes based on the corners of a hypercube:

```python
python3 hypercube.py direct --dim 384 --max_vectors 50304
```

And saves into `corners.npy` file.


## Gradient Optimziation


The following will show before and after with a gradient improvement.

However, the present algorithm is O(n^2) and may require a lot of time to complete.

```python
python3 gradient_optimization.py \
    --dimensions 6 \
    --num_vectors 32 \
    --iterations 300 \
    --lr 0.01
```

[!image](./images/min_angle_dim8.png)

The npy files are saved before and after.
