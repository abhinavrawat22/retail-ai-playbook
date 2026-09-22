"""
Tool definitions for the Store Ops Concierge agent.

Each tool is intentionally narrow-scoped (least privilege): it only reads
the in-memory data it needs and returns structured, truthful data (or a
clear "not found" message) rather than letting the LLM guess. This is a
best practice for agent tools: never let the model "fill in" data that
should come from a system of record.
"""

from langchain_core.tools import tool

from data import INVENTORY, ORDERS, CUSTOMERS, POLICIES, MAX_DISCOUNT_PCT


@tool
def check_inventory(sku: str, store_id: str) -> str:
    """Look up the on-hand quantity for a given SKU at a given store.
    Use this whenever the user asks about stock levels, availability, or
    whether an item is in stock at a specific store."""
    for row in INVENTORY:
        if row["sku"].upper() == sku.upper() and row["store_id"].upper() == store_id.upper():
            status = "IN STOCK" if row["quantity_on_hand"] > 0 else "OUT OF STOCK"
            low_stock = row["quantity_on_hand"] <= row["reorder_threshold"]
            return (
                f"{row['product_name']} ({sku}) at store {store_id}: "
                f"{row['quantity_on_hand']} units on hand [{status}]"
                f"{' - LOW STOCK, below reorder threshold' if low_stock else ''}"
            )
    return f"No inventory record found for SKU '{sku}' at store '{store_id}'."


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status, items, and expected delivery date for a given order ID.
    Use this whenever the user asks 'where is my order' or about order status,
    delivery date, or order total. Do NOT invent order details."""
    for order in ORDERS:
        if order["order_id"].upper() == order_id.upper():
            return (
                f"Order {order['order_id']}: status={order['status']}, "
                f"items={order['items']}, total=${order['total']}, "
                f"expected_delivery={order['expected_delivery']}, "
                f"customer_email={order['customer_email']}, "
                f"customer_phone={order['customer_phone']}"
            )
    return f"No order found with ID '{order_id}'."


@tool
def calculate_discount(customer_tier: str, cart_total: float) -> str:
    """Calculate the discount amount and final price for a customer given their
    loyalty tier (Gold, Silver, Bronze) and cart total. Enforces the store's
    maximum discount policy regardless of what is requested."""
    tier_rates = {"gold": 15, "silver": 10, "bronze": 5}
    rate = tier_rates.get(customer_tier.strip().lower(), 0)
    # Defense in depth: clamp to policy ceiling even if a caller/prompt tries
    # to push a higher rate. Never trust the LLM to self-enforce this.
    rate = min(rate, MAX_DISCOUNT_PCT)
    discount_amount = round(cart_total * rate / 100, 2)
    final_price = round(cart_total - discount_amount, 2)
    return (
        f"Tier={customer_tier}, applied discount rate={rate}%, "
        f"discount_amount=${discount_amount}, final_price=${final_price} "
        f"(policy max discount is {MAX_DISCOUNT_PCT}%)"
    )


@tool
def search_store_policy(query: str) -> str:
    """Search store policy documents (returns, discounts, store hours) for
    an answer to a policy question. Use this before answering any question
    about return windows, discount limits, or store hours."""
    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    scored = []
    for doc in POLICIES:
        haystack = (doc["title"] + " " + doc["text"]).lower()
        score = sum(haystack.count(term) for term in query_terms)
        if score > 0:
            scored.append((score, doc))
    if not scored:
        return "No matching policy document found."
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return f"[{best['title']}] {best['text']}"


ALL_TOOLS = [check_inventory, get_order_status, calculate_discount, search_store_policy]
