"""
MCP Expense Tracker Server (Video 18).

A local stdio MCP server for tracking personal expenses.
Persists data to expenses.json in the working directory.
Run standalone:  python mcp_expense_server.py
Connected to by: langgraph_mcp_backend.py via MultiServerMCPClient (stdio transport)

The instructor demoed a deployed remote version (streamable_http transport).
This version runs locally for the tutorial.
"""
import json
import os
import fastmcp

mcp = fastmcp.FastMCP("Expense Tracker")

EXPENSES_FILE = os.path.join(os.path.dirname(__file__), "expenses.json")


def _load() -> list:
    if os.path.exists(EXPENSES_FILE):
        with open(EXPENSES_FILE) as f:
            return json.load(f)
    return []


def _save(expenses: list) -> None:
    with open(EXPENSES_FILE, "w") as f:
        json.dump(expenses, f, indent=2)


@mcp.tool()
async def add_expense(
    amount: float,
    description: str,
    date: str,
    category: str = "General",
) -> str:
    """
    Add a new expense.
    date format: YYYY-MM-DD  (e.g. 2026-06-11)
    category: Education, Food, Travel, Entertainment, Health, etc.
    """
    expenses = _load()
    entry = {
        "id": len(expenses) + 1,
        "amount": amount,
        "description": description,
        "date": date,
        "category": category,
    }
    expenses.append(entry)
    _save(expenses)
    return (
        f"Added expense #{entry['id']}: {description} — "
        f"₹{amount:.2f} on {date} (Category: {category})"
    )


@mcp.tool()
async def list_expenses(from_date: str = "", to_date: str = "") -> str:
    """
    List recorded expenses, optionally filtered by date range.
    from_date / to_date: YYYY-MM-DD strings, both optional.
    """
    expenses = _load()
    if not expenses:
        return "No expenses recorded yet."

    if from_date or to_date:
        expenses = [
            e for e in expenses
            if (not from_date or e["date"] >= from_date)
            and (not to_date or e["date"] <= to_date)
        ]

    if not expenses:
        return "No expenses found for the given date range."

    lines = [
        f"- [{e['date']}] {e['description']} — ₹{e['amount']:.2f} ({e['category']})"
        for e in expenses
    ]
    return "Expenses:\n" + "\n".join(lines)


@mcp.tool()
async def summarize_expenses(from_date: str = "", to_date: str = "") -> str:
    """
    Summarize total expenses by category, optionally filtered by date range.
    from_date / to_date: YYYY-MM-DD strings, both optional.
    """
    expenses = _load()
    if not expenses:
        return "No expenses recorded yet."

    if from_date or to_date:
        expenses = [
            e for e in expenses
            if (not from_date or e["date"] >= from_date)
            and (not to_date or e["date"] <= to_date)
        ]

    if not expenses:
        return "No expenses in that date range."

    by_cat: dict = {}
    total = 0.0
    for e in expenses:
        cat = e.get("category", "General")
        by_cat[cat] = by_cat.get(cat, 0.0) + e["amount"]
        total += e["amount"]

    cat_lines = "\n".join(f"  {cat}: ₹{amt:.2f}" for cat, amt in by_cat.items())
    return f"Total: ₹{total:.2f}\nBy category:\n{cat_lines}"


if __name__ == "__main__":
    mcp.run()
