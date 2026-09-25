"""
Tool definitions for the Store Ops Concierge agent.

Each tool is intentionally narrow-scoped (least privilege): it only reads
the in-memory data it needs and returns structured, truthful data (or a
clear "not found" message) rather than letting the LLM guess. This is a
best practice for agent tools: never let the model "fill in" data that
should come from a system of record.
"""

from langchain_core.tools import tool, ToolException

from data import INVENTORY, ORDERS, CUSTOMERS, POLICIES, MAX_DISCOUNT_PCT

# --- Simulated transient failure demo -------------------------------------
# To demonstrate "tool call fails, agent retries and resumes from the failed
# step" in a live workshop, this specific order ID fails exactly once per
# process lifetime (module-level state), then succeeds on the very next
# attempt for the same ID - mimicking a flaky downstream order-status API.
_SIMULATED_FAILURE_ORDER_ID = "ORD5002"
_simulated_failure_triggered: dict[str, bool] = {}


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
    normalized_id = order_id.upper()

    # Simulated transient failure demo: the first call for this specific order
    # ID raises, so the workshop can show the agent catching a tool error and
    # retrying on its own. Every call after the first (for this ID, in this
    # process) succeeds normally.
    if normalized_id == _SIMULATED_FAILURE_ORDER_ID and not _simulated_failure_triggered.get(normalized_id):
        _simulated_failure_triggered[normalized_id] = True
        raise ToolException(
            f"Temporary error: order-status service timed out while looking up "
            f"'{order_id}'. This is a simulated transient failure for the retry demo."
        )

    for order in ORDERS:
        if order["order_id"].upper() == normalized_id:
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

# Enable per-tool error handling: if any tool raises (e.g. get_order_status's
# simulated transient failure, or an unexpected bug), LangChain feeds the
# exception text back into the agent's scratchpad as a normal observation
# instead of letting it propagate up and crash executor.invoke(). This is what
# lets the LLM "see" a tool error and decide to retry (see SYSTEM_PROMPT rule 7
# in agent.py) rather than the whole turn failing.
for _t in ALL_TOOLS:
    _t.handle_tool_error = True


def reset_simulated_failures() -> None:
    """Resets the one-time simulated failure demo (see get_order_status) so it
    can be re-triggered again in the same running process - used by the
    Streamlit sidebar's 'Reset failure demo' button."""
    _simulated_failure_triggered.clear()
