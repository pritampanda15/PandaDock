"""
Batched limited-memory BFGS.

Stage four. The CPU relaxes every trial pose with SciPy's L-BFGS-B, and stage
two measured how much that matters: 3.3M raw Monte Carlo evaluations reached
-6.601 kcal/mol where the CPU's much smaller L-BFGS-refined search reached
-7.036. Local optimisation is not a refinement of this search, it is most of it.

SciPy cannot be used here because it solves one problem at a time, and so can
`torch.optim.LBFGS`, which treats the whole parameter tensor as a single problem
and would couple every chain's line search together. This is therefore a
from-scratch two-loop recursion that keeps one independent history, step length
and convergence flag per chain.

The optimiser is written against a flat parameter vector of shape (B, D) and a
closure returning energy and gradient. It knows nothing about translations,
rotations or torsions, so stage five can hand it 6 + n_torsions parameters
without changes here.

Two batching decisions worth stating, because both are places where a
single-problem implementation does not generalise:

* Line search is backtracking Armijo, not strong Wolfe. Wolfe needs a per-chain
  loop with early exit on a bracketing condition; batched, that degenerates into
  running every chain for the worst chain's iteration count while masking. The
  curvature condition Wolfe would enforce is recovered by simply skipping the
  history update when s.y <= 0, which is what keeps the approximation positive
  definite.
* Converged chains are masked out rather than removed. Compacting the batch
  would need a gather every iteration and would renumber chains, and the whole
  batch runs at the speed of its slowest member either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

try:
    import torch

    torch_available = True
except ImportError:  # pragma: no cover - exercised only in a base install
    torch = None
    torch_available = False


@dataclass
class LBFGSConfig:
    """
    Defaults chosen to match the CPU's `minimize(..., method="L-BFGS-B")` call.

    `max_iter` and `history` mirror its maxiter=60 and maxcor=10 so that the two
    optimisers are given the same budget when their results are compared.
    """

    max_iter: int = 60
    history: int = 10
    grad_tol: float = 1e-6
    step_tol: float = 1e-12
    # Armijo sufficient-decrease constant; 1e-4 is the standard choice.
    c1: float = 1e-4
    max_line_search: int = 20
    initial_step: float = 1.0


def _batched_dot(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Row-wise dot product, (B, D) x (B, D) -> (B, 1)."""
    return (a * b).sum(dim=-1, keepdim=True)


def batched_lbfgs(
    x0: "torch.Tensor",
    closure: Callable[["torch.Tensor"], Tuple["torch.Tensor", "torch.Tensor"]],
    config: Optional[LBFGSConfig] = None,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Minimise a batch of independent problems that share a parameter shape.

    Args:
        x0: (B, D) starting points, one per chain.
        closure: maps (B, D) to energy (B,) and gradient (B, D). Called once per
            iteration plus once per line-search trial.
        config: see LBFGSConfig.

    Returns:
        x: (B, D) best parameters found for each chain
        energy: (B,) energy at those parameters
        converged: (B,) bool, whether the gradient tolerance was met

    Every chain's returned energy is the energy of its returned parameters; the
    two are never taken from different iterations.
    """
    if not torch_available:  # pragma: no cover
        raise ImportError("batched_lbfgs needs torch")

    cfg = config or LBFGSConfig()
    x = x0.clone()
    energy, grad = closure(x)

    n_chains, n_dof = x.shape
    history_s: list = []
    history_y: list = []
    history_rho: list = []

    # `active` is the mask of chains still being optimised. A chain leaves when
    # its gradient is small or its line search can make no further progress; it
    # then simply stops being updated, keeping its last accepted state.
    active = grad.norm(dim=-1) > cfg.grad_tol

    for _ in range(cfg.max_iter):
        if not bool(active.any()):
            break

        # ---- direction: two-loop recursion over the stored curvature pairs ----
        q = grad.clone()
        alphas = []
        for s_i, y_i, rho_i in zip(
            reversed(history_s), reversed(history_y), reversed(history_rho)
        ):
            alpha = rho_i * _batched_dot(s_i, q)
            q = q - alpha * y_i
            alphas.append(alpha)

        if history_s:
            s_last, y_last = history_s[-1], history_y[-1]
            yy = _batched_dot(y_last, y_last)
            gamma = _batched_dot(s_last, y_last) / yy.clamp_min(1e-16)
            # A non-positive scaling would point the step uphill; fall back to
            # an unscaled gradient step for those chains rather than trusting it.
            gamma = torch.where(gamma > 0, gamma, torch.ones_like(gamma))
        else:
            gamma = torch.ones_like(q[:, :1])

        z = gamma * q
        for s_i, y_i, rho_i, alpha in zip(
            history_s, history_y, history_rho, reversed(alphas)
        ):
            beta = rho_i * _batched_dot(y_i, z)
            z = z + s_i * (alpha - beta)

        direction = -z

        # A curvature history built from a rough line search can occasionally
        # yield an ascent direction. Steepest descent is always valid, so use it
        # for exactly those chains instead of taking an uphill step.
        slope = _batched_dot(grad, direction)
        ascent = slope >= 0
        direction = torch.where(ascent, -grad, direction)
        slope = torch.where(ascent, -_batched_dot(grad, grad), slope)

        # ---- backtracking line search, one step length per chain ----
        # A unit step is right for a quasi-Newton direction, which is already
        # scaled by the curvature estimate. On the first iteration there is no
        # history, so the direction is a raw gradient whose magnitude has nothing
        # to do with the distance to the minimum: starting at 1.0 there means
        # backtracking all the way down every time, which measured at ~20 closure
        # evaluations per iteration. Scaling the first step by 1/|g| is the
        # standard remedy (Nocedal & Wright, eq. 3.60).
        if history_s:
            step = torch.full(
                (n_chains, 1), cfg.initial_step, dtype=x.dtype, device=x.device
            )
        else:
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-16)
            step = torch.clamp(1.0 / grad_norm, max=cfg.initial_step)
        accepted = torch.zeros(n_chains, dtype=torch.bool, device=x.device)
        new_x = x.clone()
        new_energy = energy.clone()

        for _ in range(cfg.max_line_search):
            # Unlike the Monte Carlo loop, this does synchronise, deliberately.
            # A typical line search accepts in one or two trials, so testing the
            # mask costs one sync and saves up to eighteen closure evaluations;
            # running the loop blind to avoid the sync would be far more
            # expensive than the sync it avoids.
            if not bool((active & ~accepted).any()):
                break
            trial_x = x + step * direction
            trial_energy, _ = closure(trial_x)

            sufficient = trial_energy <= (
                energy + cfg.c1 * (step * slope).squeeze(-1)
            )
            take = sufficient & active & (~accepted)

            new_x = torch.where(take.unsqueeze(-1), trial_x, new_x)
            new_energy = torch.where(take, trial_energy, new_energy)
            accepted = accepted | take

            # Only chains still searching shrink their step.
            shrink = (~accepted) & active
            step = torch.where(shrink.unsqueeze(-1), step * 0.5, step)

        # ---- accept, and update the curvature history ----
        moved = accepted & active
        s = torch.where(moved.unsqueeze(-1), new_x - x, torch.zeros_like(x))

        x = torch.where(moved.unsqueeze(-1), new_x, x)
        energy = torch.where(moved, new_energy, energy)
        new_grad = grad
        if bool(moved.any()):
            _, new_grad = closure(x)

        y = new_grad - grad
        sy = _batched_dot(s, y)

        # Skip the update where curvature is non-positive or the chain did not
        # move: storing those pairs is what makes an L-BFGS approximation drift
        # away from positive definite.
        usable = (sy.squeeze(-1) > 1e-12) & moved
        s = torch.where(usable.unsqueeze(-1), s, torch.zeros_like(s))
        y = torch.where(usable.unsqueeze(-1), y, torch.zeros_like(y))
        rho = torch.where(
            usable.unsqueeze(-1), 1.0 / sy.clamp_min(1e-16), torch.zeros_like(sy)
        )

        history_s.append(s)
        history_y.append(y)
        history_rho.append(rho)
        if len(history_s) > cfg.history:
            history_s.pop(0)
            history_y.pop(0)
            history_rho.pop(0)

        grad = new_grad

        # A chain that could not accept any step has converged as far as this
        # line search can take it.
        active = active & moved & (grad.norm(dim=-1) > cfg.grad_tol)

    converged = grad.norm(dim=-1) <= cfg.grad_tol
    return x, energy, converged
