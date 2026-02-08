"""
Shared easing curve utilities for TrentNodes.

All functions accept and return torch.Tensor values in [0, 1].
GPU-compatible -- no numpy, no Python loops, fully vectorized.
"""

import torch
import math


def linear(t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation (no easing)."""
    return t


def ease_in(t: torch.Tensor) -> torch.Tensor:
    """Quadratic ease-in: slow start, fast end."""
    return t * t


def ease_out(t: torch.Tensor) -> torch.Tensor:
    """Quadratic ease-out: fast start, slow end."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t: torch.Tensor) -> torch.Tensor:
    """Cubic ease-in-out: slow start and end, fast middle."""
    return 3.0 * t * t - 2.0 * t * t * t


def smooth(t: torch.Tensor) -> torch.Tensor:
    """Hermite smoothstep: smooth start and end."""
    return t * t * (3.0 - 2.0 * t)


def bounce(t: torch.Tensor) -> torch.Tensor:
    """Simple parabolic bounce: slight overshoot near end."""
    return t * (2.0 - t)


def elastic(t: torch.Tensor) -> torch.Tensor:
    """Elastic easing with slight overshoot."""
    p = 0.3
    s = p / 4.0
    result = (
        1.0
        - torch.pow(2.0, -10.0 * t)
        * torch.sin((t - s) * (2.0 * math.pi) / p)
    )
    # Clamp boundary values to avoid numerical noise
    return torch.where(
        t <= 0.0, torch.zeros_like(t),
        torch.where(t >= 1.0, torch.ones_like(t), result)
    )


def cubic_bezier(
    t: torch.Tensor,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
) -> torch.Tensor:
    """
    CSS-style cubic bezier easing.

    Endpoints are fixed at (0, 0) and (1, 1). Control points
    (p1_x, p1_y) and (p2_x, p2_y) shape the curve.

    Uses Newton-Raphson iteration to solve B_x(u) = t for the
    bezier parameter u, then evaluates B_y(u).

    Args:
        t: Input positions in [0, 1].
        p1_x: First control point x.
        p1_y: First control point y.
        p2_x: Second control point x.
        p2_y: Second control point y.

    Returns:
        Eased values in [0, 1].
    """
    # Newton-Raphson: solve for u where B_x(u) = t
    u = t.clone()
    for _ in range(8):
        u2 = u * u
        u3 = u2 * u
        omu = 1.0 - u
        omu2 = omu * omu

        # B_x(u) = 3*p1_x*u*(1-u)^2 + 3*p2_x*u^2*(1-u) + u^3
        bx = 3.0 * p1_x * u * omu2 + 3.0 * p2_x * u2 * omu + u3

        # B_x'(u)
        dbx = (
            3.0 * p1_x * (1.0 - 4.0 * u + 3.0 * u2)
            + 3.0 * p2_x * (2.0 * u - 3.0 * u2)
            + 3.0 * u2
        )
        dbx = torch.clamp(dbx, min=1e-7)
        u = torch.clamp(u - (bx - t) / dbx, 0.0, 1.0)

    # Evaluate B_y(u)
    u2 = u * u
    u3 = u2 * u
    omu = 1.0 - u
    omu2 = omu * omu
    by = 3.0 * p1_y * u * omu2 + 3.0 * p2_y * u2 * omu + u3
    return torch.clamp(by, 0.0, 1.0)


# Registry for lookup by name string
EASING_FUNCTIONS = {
    "linear": linear,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "smooth": smooth,
    "bounce": bounce,
    "elastic": elastic,
}


def apply_easing(
    t: torch.Tensor,
    method: str,
    **kwargs,
) -> torch.Tensor:
    """
    Apply a named easing curve to t values.

    For cubic_bezier, pass p1_x, p1_y, p2_x, p2_y as kwargs.

    Args:
        t: Input values in [0, 1].
        method: Easing function name.
        **kwargs: Extra args for cubic_bezier.

    Returns:
        Eased values in [0, 1].
    """
    t = torch.clamp(t, 0.0, 1.0)
    if method == "cubic_bezier":
        return cubic_bezier(
            t,
            kwargs.get("p1_x", 0.25),
            kwargs.get("p1_y", 0.1),
            kwargs.get("p2_x", 0.25),
            kwargs.get("p2_y", 1.0),
        )
    fn = EASING_FUNCTIONS.get(method, linear)
    return fn(t)
