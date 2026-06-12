"""
MCP Math Server (Video 18).

A local stdio MCP server exposing arithmetic operations.
Run standalone:  python mcp_math_server.py
Connected to by: langgraph_mcp_backend.py via MultiServerMCPClient (stdio transport)

All tools are async — MCP servers are async-only by design.
"""
import fastmcp

mcp = fastmcp.FastMCP("Math Server")


@mcp.tool()
async def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@mcp.tool()
async def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


@mcp.tool()
async def modulus(a: float, b: float) -> float:
    """Return the remainder of a divided by b."""
    if b == 0:
        raise ValueError("Cannot compute modulus with zero divisor.")
    return a % b


if __name__ == "__main__":
    mcp.run()
