"""
HEPPO standardization & quantization primitives (paper Section II).

Implements:
  - DynamicRewardStandardizer: running mean/std (Welford's algorithm,
    Eqs. 6-9) applied to rewards BEFORE GAE, kept in standardized
    form throughout (paper found this gives the ~50% reward boost,
    Section V-A / Experiment 5).
  - block_standardize_values / block_destandardize_values: per-batch
    mean/std standardization of the critic's values (Section II-B),
    quantized then de-standardized back to original scale at use time.
  - uniform_quantize / uniform_dequantize: n-bit uniform quantization
    (paper recommends n >= 8, Section V-B).

These are pure software analogues of the fixed-point hardware blocks
described in Section III; here everything runs in float32 but the
quantize/dequantize round-trip introduces the same discretization
error the FPGA's n-bit codewords would, so you can study its effect
on training (Figs. 8-10) without building hardware.
"""

from __future__ import annotations
import torch


class DynamicRewardStandardizer:
    """Running (mean, std) standardizer for rewards, updated online.

    Implements Eqs. 6-9 of the paper (Welford's algorithm), i.e. it
    accounts for *all* rewards seen so far rather than re-standardizing
    within each batch/epoch independently (which the paper shows
    causes divergence, Section II-A).
    """

    def __init__(self, epsilon: float = 1e-8, device: str | torch.device = "cpu"):
        self.epsilon = epsilon
        self.device = device
        self.n = 0
        self.mean = torch.tensor(0.0, device=device)
        # S_n: Welford's running sum-of-squared-deviations (Eq. 8)
        self.s = torch.tensor(0.0, device=device)

    @property
    def std(self) -> torch.Tensor:
        if self.n < 1:
            return torch.tensor(1.0, device=self.device)
        return torch.sqrt(self.s / self.n + self.epsilon)

    def update(self, rewards: torch.Tensor) -> None:
        """Update running statistics with a batch of new rewards (any shape)."""
        flat = rewards.detach().reshape(-1).to(self.device)
        for r in flat:
            self.n += 1
            delta = r - self.mean
            self.mean = self.mean + delta / self.n          # Eq. 6 / Eq. 7
            delta2 = r - self.mean
            self.s = self.s + delta * delta2                # Eq. 8

    def standardize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Map raw rewards -> standardized form using *current* running stats."""
        return (rewards - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean.item(), "s": self.s.item()}

    def load_state_dict(self, sd: dict) -> None:
        self.n = sd["n"]
        self.mean = torch.tensor(sd["mean"], device=self.device)
        self.s = torch.tensor(sd["s"], device=self.device)


def block_standardize_values(values: torch.Tensor, eps: float = 1e-8):
    """Block standardization of values (Section II-B).

    Computes a single (mean, std) over the whole batch ("block") of
    values and standardizes to zero mean / unit variance. Unlike
    rewards, this is recomputed fresh each batch (not running),
    because the critic's output distribution shifts as it trains
    (paper Fig. 2) -- a running statistic would lag behind and the
    paper found dynamic standardization of values hurts the loss.

    Returns (standardized_values, mu_v, sigma_v). mu_v/sigma_v must be
    stored alongside the quantized values so they can be used to
    de-standardize later (paper step 5, "Reconstruction").
    """
    mu_v = values.mean()
    sigma_v = values.std(unbiased=False) + eps
    standardized = (values - mu_v) / sigma_v
    return standardized, mu_v.detach(), sigma_v.detach()


def block_destandardize_values(standardized: torch.Tensor, mu_v: torch.Tensor,
                                sigma_v: torch.Tensor) -> torch.Tensor:
    """Inverse of block_standardize_values: v = v_std * sigma_v + mu_v."""
    return standardized * sigma_v + mu_v


def uniform_quantize(x: torch.Tensor, n_bits: int = 8, clip: float = 5.0):
    """Uniform n-bit quantization of an already-standardized tensor.

    `clip` bounds the standardized range that gets mapped onto the
    2**n_bits codewords (values are ~N(0,1) after standardization, so
    +/-5 std covers virtually all mass). Values outside the range are
    clamped, matching a fixed-point hardware quantizer's saturation
    behavior.

    Returns the (still-float, but quantization-discretized) tensor.
    A real hardware/integer deployment would instead return an int
    tensor of codewords; we keep it float here so this function is a
    drop-in differentiable-friendly stand-in (gradients aren't pushed
    through values/rewards during GAE anyway, only during the critic
    loss, which uses the de-standardized, de-quantized value).
    """
    levels = 2 ** n_bits
    x_clamped = torch.clamp(x, -clip, clip)
    # map [-clip, clip] -> [0, levels-1] integers, then back to float range
    scale = (levels - 1) / (2 * clip)
    codeword = torch.round((x_clamped + clip) * scale)
    codeword = torch.clamp(codeword, 0, levels - 1)
    x_quant = codeword / scale - clip
    return x_quant


def uniform_dequantize(x_quant: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
    """No-op placeholder for symmetry with the paper's pipeline description.

    In this float-based implementation, uniform_quantize already
    returns the de-quantized (but discretized) value, so dequantization
    is the identity. Kept as a separate function so the call sites in
    the training loop mirror Section II-C's two-step description
    (quantize -> [store] -> de-quantize) even though no extra work
    happens here.
    """
    return x_quant
