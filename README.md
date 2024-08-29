# Grokking Modular Arithmetic (MLX)

<p align="center">
  <img src="media/grokking.png" alt="Grokking" width="50%">
</p>

_Grokking_: a phenomenon observed in neural nets, where after an initial phase of overfitting (or memorization), the model suddenly achieves perfect generalization, inspired by [Power et al. (2022)](https://arxiv.org/abs/2201.02177). We incoporate some modern Transformer tricks (e.g., RoPE, RMSNorm, SiLU, etc.) and achieve grokking in $\lt$ 150 epochs on modular division when $p=97$ on 50% of the training data using a 2 layer, 1 head, 128 dim net.

## Background

We define modular arithmetic for the following operations given a prime modulus $p$ and $(a, b)$ for $0 \leq a \lt p, 0 \lt b \lt p$:
- **Addition**: $a \circ b = a + b \mod p$
- **Subtraction**: $a \circ b = a - b \mod p$
- **Multiplication**: $a \circ b = a \cdot b \mod p$
- **Division**: $a \circ b = a / b \mod p$, using Fermat’s Little Theorem given by $a \cdot b^{p-2} \mod p$

## Running

Run with default params for $a / b \mod p$ and save the result in [`media/grokking.png`](media/grokking.png):
```bash
python main.py
```
- **`main.py`**: training and evaluation loops
- **`models.py`**: defines the Transformer model
- **`data.py`**: generate the dataset

## Dependencies

Install the dependencies (optimized for Apple silicon; yay for [MLX](https://github.com/ml-explore/mlx)!):
```bash
pip install -r requirements.txt
```
