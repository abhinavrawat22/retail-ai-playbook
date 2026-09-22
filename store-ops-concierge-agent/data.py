"""
In-memory sample data for the Store Ops Concierge lab.

This intentionally avoids a real database so the lab has zero infra setup.
All "tables" are plain Python lists of dicts, reset every time the app restarts.
Swap these for a SQLite/Postgres-backed repository later without changing
the tool function signatures in tools.py.
"""

# --- Inventory -----------------------------------------------------------
INVENTORY = [
    {"sku": "SKU1001", "product_name": "Wireless Mouse", "store_id": "ST01", "quantity_on_hand": 42, "reorder_threshold": 10},
    {"sku": "SKU1001", "product_name": "Wireless Mouse", "store_id": "ST02", "quantity_on_hand": 3, "reorder_threshold": 10},
    {"sku": "SKU1002", "product_name": "Bluetooth Headphones", "store_id": "ST01", "quantity_on_hand": 0, "reorder_threshold": 5},
    {"sku": "SKU1002", "product_name": "Bluetooth Headphones", "store_id": "ST02", "quantity_on_hand": 18, "reorder_threshold": 5},
    {"sku": "SKU1003", "product_name": "USB-C Charging Cable", "store_id": "ST01", "quantity_on_hand": 120, "reorder_threshold": 20},
    {"sku": "SKU1004", "product_name": "Portable Speaker", "store_id": "ST03", "quantity_on_hand": 7, "reorder_threshold": 10},
]

# --- Customers -------------------------------------------------------------
CUSTOMERS = [
    {"customer_id": "CUST01", "name": "Jane Doe", "loyalty_tier": "Gold"},
    {"customer_id": "CUST02", "name": "John Smith", "loyalty_tier": "Silver"},
    {"customer_id": "CUST03", "name": "Priya Nair", "loyalty_tier": "Bronze"},
]

# --- Orders ------------------------------------------------------------
# Deliberately includes PII fields (email/phone) so the guardrails lab can
# demonstrate redaction when the agent handles order lookups.
ORDERS = [
    {
        "order_id": "ORD5001",
        "customer_id": "CUST01",
        "customer_email": "jane.doe@example.com",
        "customer_phone": "+1-555-201-3344",
        "status": "Shipped",
        "items": ["SKU1001", "SKU1003"],
        "total": 44.98,
        "expected_delivery": "2026-09-25",
    },
    {
        "order_id": "ORD5002",
        "customer_id": "CUST02",
        "customer_email": "john.smith@example.com",
        "customer_phone": "+1-555-201-3355",
        "status": "Processing",
        "items": ["SKU1002"],
        "total": 89.99,
        "expected_delivery": "2026-09-28",
    },
    {
        "order_id": "ORD5003",
        "customer_id": "CUST03",
        "customer_email": "priya.nair@example.com",
        "customer_phone": "+1-555-201-3366",
        "status": "Delivered",
        "items": ["SKU1004"],
        "total": 59.99,
        "expected_delivery": "2026-09-15",
    },
]

# --- Store policies (used by the lightweight "search_store_policy" tool) --
# A simple keyword-searchable list stands in for a vector store, so the lab
# needs no embeddings/vector DB. Swap for the RAG pipeline from the earlier
# hands-on if you want true semantic retrieval.
POLICIES = [
    {
        "id": "returns",
        "title": "Return Policy",
        "text": (
            "Items may be returned within 30 days of delivery with a receipt or order ID. "
            "Opened electronics can be returned within 15 days if all accessories are included. "
            "Refunds are issued to the original payment method within 5-7 business days."
        ),
    },
    {
        "id": "discounts",
        "title": "Discount & Loyalty Policy",
        "text": (
            "Gold tier customers receive up to 15% discount, Silver tier up to 10%, "
            "Bronze tier up to 5%. The maximum discount any associate or agent may apply, "
            "regardless of tier or promotion stacking, is 25% of cart total."
        ),
    },
    {
        "id": "store_hours",
        "title": "Store Hours",
        "text": (
            "Standard store hours are 9 AM - 9 PM Monday-Saturday, and 10 AM - 6 PM on Sunday. "
            "Hours may vary on public holidays."
        ),
    },
]

# Discount ceiling enforced in code (defense in depth, not just prompt instructions)
MAX_DISCOUNT_PCT = 25
