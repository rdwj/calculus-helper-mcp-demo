"""Tests for the integrate tool."""

import pytest
import sympy as sp
from fastmcp.exceptions import ToolError
from unittest.mock import AsyncMock

from src.calc import parse_expression
from src.tools.integrate import integrate


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indefinite_integral():
    """∫x² dx should contain x**3/3 and note the omitted constant."""
    ctx = AsyncMock()
    result = await integrate(
        expression="x**2",
        variable="x",
        ctx=ctx,
    )

    assert "x**3/3" in result["result"], f"Expected x**3/3 in {result['result']!r}"
    assert any("constant" in note.lower() for note in result["assumptions"])
    ctx.info.assert_called_once()


@pytest.mark.asyncio
async def test_definite_exact():
    """∫₀¹ x dx = 1/2 (exact)."""
    result = await integrate(
        expression="x",
        variable="x",
        lower_bound="0",
        upper_bound="1",
        ctx=None,
    )

    parsed = parse_expression(result["result"])
    assert parsed == sp.Rational(1, 2), (
        f"Expected 1/2, got {result['result']!r}"
    )
    assert result["is_exact"] is True


@pytest.mark.asyncio
async def test_erf_closed_form():
    """∫₀¹ exp(-x²) dx = sqrt(pi)*erf(1)/2 — SymPy closes this with erf."""
    result = await integrate(
        expression="exp(-x**2)",
        variable="x",
        lower_bound="0",
        upper_bound="1",
        ctx=None,
    )

    assert "erf" in result["result"], (
        f"Expected 'erf' in result, got {result['result']!r}"
    )
    assert result["is_exact"] is True


@pytest.mark.asyncio
async def test_numerical_mode():
    """numerical=True should return a decimal approximation starting with '0.74'."""
    result = await integrate(
        expression="exp(-x**2)",
        variable="x",
        lower_bound="0",
        upper_bound="1",
        numerical=True,
        ctx=None,
    )

    assert result["is_exact"] is False, "Numerical result must have is_exact=False"
    assert result["result"].startswith("0.74"), (
        f"Expected ~0.7468..., got {result['result']!r}"
    )


@pytest.mark.asyncio
async def test_divergent_integral():
    """∫₁^∞ (1/x) dx diverges; result should be 'oo' with a divergence note."""
    result = await integrate(
        expression="1/x",
        variable="x",
        lower_bound="1",
        upper_bound="oo",
        ctx=None,
    )

    assert result["result"] == "oo", f"Expected 'oo', got {result['result']!r}"
    assert any("diverge" in note.lower() for note in result["assumptions"])


@pytest.mark.asyncio
async def test_symbolic_bounds_pi():
    """Bounds can be symbolic expressions: ∫₀^π sin(x) dx = 2."""
    result = await integrate(
        expression="sin(x)",
        variable="x",
        lower_bound="0",
        upper_bound="pi",
        ctx=None,
    )

    parsed = parse_expression(result["result"])
    assert parsed == sp.Integer(2), f"Expected 2, got {result['result']!r}"
    assert result["is_exact"] is True


@pytest.mark.asyncio
async def test_ctx_none_allowed():
    """Tool must not error when ctx=None."""
    result = await integrate(
        expression="x**2",
        variable="x",
        ctx=None,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_lower_bound_raises():
    """Providing only lower_bound must raise a ToolError about needing both bounds."""
    with pytest.raises(ToolError, match=r"(?i)both bounds|both.*bounds"):
        await integrate(
            expression="x",
            variable="x",
            lower_bound="0",
            upper_bound=None,
            ctx=None,
        )


@pytest.mark.asyncio
async def test_only_upper_bound_raises():
    """Providing only upper_bound must raise a ToolError about needing both bounds."""
    with pytest.raises(ToolError, match=r"(?i)both bounds|both.*bounds"):
        await integrate(
            expression="x",
            variable="x",
            lower_bound=None,
            upper_bound="1",
            ctx=None,
        )


@pytest.mark.asyncio
async def test_numerical_without_bounds_raises():
    """numerical=True with no bounds should raise a ToolError."""
    with pytest.raises(ToolError, match="lower_bound"):
        await integrate(
            expression="x**2",
            variable="x",
            numerical=True,
            ctx=None,
        )


@pytest.mark.asyncio
async def test_caret_in_expression_raises():
    """Using '^' in the integrand must raise a ToolError coaching the fix."""
    with pytest.raises(ToolError, match=r"\*\*"):
        await integrate(
            expression="x^2",
            variable="x",
            ctx=None,
        )


@pytest.mark.asyncio
async def test_caret_in_bound_raises():
    """Using '^' in a bound must raise a ToolError coaching the fix."""
    with pytest.raises(ToolError, match=r"\*\*"):
        await integrate(
            expression="x**2",
            variable="x",
            lower_bound="0",
            upper_bound="2^3",
            ctx=None,
        )
