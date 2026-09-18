"""Selective state-space encoder (Mamba-style), implemented in pure PyTorch.

Why this belongs here rather than another dilated CNN:

X-ray light curves are long (1 s cadence over hours) and the informative
structure is sparse and multi-scale -- a two-minute rise sitting inside a
two-hour decay.  A TCN reaches long context only by stacking dilations, and a
transformer costs O(T^2).  A selective SSM carries a recurrent state with
*input-dependent* dynamics: it can hold the pre-flare background for an hour,
then let the state turn over quickly the moment a rise starts.  That gating is
exactly the behaviour a flare nowcaster needs, and it is linear in T.

No `mamba-ssm` dependency: that package needs CUDA compilation and does not
build on every platform this has to run on.  The scan here is a plain
sequential loop, which at T = 360 costs a few hundred small kernel launches --
slower than the fused kernel, but portable, readable and correct.  Swap in the
fused implementation if you move to very long sequences.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d, ChannelNorm1d


#: Timesteps per parallel block.  See :func:`selective_scan` for why this is
#: a compromise rather than "as large as possible".
SCAN_CHUNK = 64


def _hillis_steele(dA: torch.Tensor, dBx: torch.Tensor) -> torch.Tensor:
    """Doubling scan over the whole time axis: O(log T) steps, O(T) memory."""
    a, b = dA, dBx
    t = a.shape[1]
    step = 1
    while step < t:
        # Shift right along time; identity for `a` is 1, for `b` it is 0.
        a_prev = F.pad(a[:, :-step], (0, 0, 0, 0, step, 0), value=1.0)
        b_prev = F.pad(b[:, :-step], (0, 0, 0, 0, step, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        step *= 2
    return b


def selective_scan(dA: torch.Tensor, dBx: torch.Tensor,
                   chunk: int = SCAN_CHUNK) -> torch.Tensor:
    """Solve h[t] = dA[t] * h[t-1] + dBx[t], h[-1] = 0.

    The recurrence is first-order linear and therefore *associative*: the pair
    (a, b) composes as (a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2).  A doubling
    scan can then compute every state in O(log T) steps instead of O(T).

    Both extremes measured badly on real hardware, which is why this is
    chunked (RTX 3050, B=64, T=360, d_inner=128, d_state=16, fwd+bwd):

    =================  ==========  =========
    variant            time        peak mem
    =================  ==========  =========
    sequential loop     3665 ms     1081 MiB
    full doubling scan   531 ms     5580 MiB
    =================  ==========  =========

    The loop is slow because a 360-deep autograd graph means thousands of tiny
    kernels in the backward pass.  The full doubling scan is fast but makes
    nine passes over ~190 MiB tensors and peaks at 5.6 GiB -- which does not
    fit comfortably on a 6 GB card.

    Chunking gets both: a doubling scan inside blocks of ``chunk`` steps, and
    a short sequential carry across the T/chunk blocks.  Memory stays
    proportional to the chunk, and the sequential part is only a handful of
    iterations.
    """
    b_, t, d, n = dA.shape
    if t <= chunk:
        return _hillis_steele(dA, dBx)

    outputs = []
    carry = torch.zeros(b_, d, n, device=dA.device, dtype=dA.dtype)
    for s in range(0, t, chunk):
        a_blk = dA[:, s:s + chunk]
        b_blk = dBx[:, s:s + chunk]
        # States within the block, assuming the block starts from zero ...
        local = _hillis_steele(a_blk, b_blk)
        # ... then fold in the state carried from all previous blocks.
        cum_a = torch.cumprod(a_blk, dim=1)
        h = local + cum_a * carry.unsqueeze(1)
        outputs.append(h)
        carry = h[:, -1]
    return torch.cat(outputs, dim=1)


def selective_scan_sequential(dA: torch.Tensor, dBx: torch.Tensor) -> torch.Tensor:
    """Reference implementation of :func:`selective_scan`, O(T) and obvious.

    Kept solely so the fast path can be tested against something that is
    transparently correct.
    """
    b_, t, d, n = dA.shape
    h = torch.zeros(b_, d, n, device=dA.device, dtype=dA.dtype)
    out = []
    for i in range(t):
        h = dA[:, i] * h + dBx[:, i]
        out.append(h)
    return torch.stack(out, dim=1)


class SelectiveSSM(nn.Module):
    """One selective state-space layer.

    State recursion, per channel d and state index n:

        h[t] = exp(dt[t] * A[d, n]) * h[t-1] + dt[t] * B[t, n] * x[t, d]
        y[t] = sum_n C[t, n] * h[t, n] + D[d] * x[t, d]

    ``dt``, ``B`` and ``C`` are projected from the input at every step, which
    is what makes the model *selective*: the effective time constant adapts to
    what the Sun is doing rather than being fixed at training time.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = CausalConv1d(self.d_inner, self.d_inner, d_conv,
                                 groups=self.d_inner)
        # dt, B and C are all read off the input.
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # A is parameterised as -exp(A_log) so it stays strictly negative and
        # the recursion can never blow up, however the optimiser moves.
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Bias init so the initial timesteps span [dt_min, dt_max]: without
        # this the state either forgets instantly or never updates.
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, T, D)
        b, t, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)

        xi = self.conv(xi.transpose(1, 2)).transpose(1, 2)
        xi = F.silu(xi)

        proj = self.x_proj(xi)
        dt_raw, bb, cc = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))              # (B, T, d_inner)

        A = -torch.exp(self.A_log)                          # (d_inner, d_state)
        # Discretise: a zero-order hold on A, and the simple Euler form for B.
        dA = torch.exp(dt.unsqueeze(-1) * A)                # (B, T, d_inner, N)
        dBx = (dt.unsqueeze(-1) * bb.unsqueeze(2)) * xi.unsqueeze(-1)

        states = selective_scan(dA, dBx)                    # (B, T, d_inner, N)
        y = torch.einsum("btdn,btn->btd", states, cc)

        y = y + self.D * xi
        y = y * F.silu(z)
        return self.out_proj(y)


class SSMBlock(nn.Module):
    """Pre-norm residual wrapper around a selective SSM."""

    def __init__(self, d_model: int, d_state: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ssm(self.norm(x)))


class SSMEncoder(nn.Module):
    """Drop-in replacement for TCNEncoder: (B, C, T) -> (B, H, T), causal.

    ``checkpoint`` trades compute for memory, and is on by default because it
    has to be.  The scan keeps O(T x d_inner x d_state) intermediates *per
    layer*, and a two-encoder model stacks eight of them: benchmarking one
    scan in isolation at 3.8 GiB looked fine and then ran a 6 GB card out of
    memory in training, because autograd was holding all eight.  Recomputing
    each block in the backward pass removes that multiplier.
    """

    def __init__(self, in_ch: int, hidden: int, n_layers: int = 4,
                 d_state: int = 8, dropout: float = 0.1,
                 checkpoint: bool = True):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList(
            [SSMBlock(hidden, d_state=d_state, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.out_norm = ChannelNorm1d(hidden)
        self.checkpoint = checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).transpose(1, 2)   # (B, T, H)
        for blk in self.blocks:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
        return self.out_norm(h.transpose(1, 2))
