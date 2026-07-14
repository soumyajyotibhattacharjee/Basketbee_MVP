"""
BasketBee — Executive MBA Working Prototype (single-file, self-contained)
==========================================================================
AI-assisted quick-commerce comparison and basket-optimization platform demo.
Three navigation views:
  1. Customer MVP        — free-text + structured basket, city/pincode aware,
                            brand-preference and substitution-approval portals,
                            deterministic 5-platform comparison engine.
  2. Company Dashboard   — B2B fill-rate / conversion / demand dashboard with
                            city, pincode, category and brand filters.
  3. Demo Data Explorer  — searchable, filterable, downloadable view of the
                            full embedded product/brand/offer dataset.

100% OFFLINE. No external APIs, no scraping, no database, no environment
variables. Everything is generated once from a fixed random seed, so the app
is fully reproducible. Brand names used are a manually curated, illustrative
academic catalogue based on commonly visible public retail information —
see the disclaimer rendered at the top of every page.

Only native Streamlit UI primitives are used throughout (st.title, st.header,
st.subheader, st.caption, st.container(border=True), st.columns, st.metric,
st.info/success/warning/error, st.dataframe, st.data_editor, st.selectbox,
st.multiselect, st.slider, st.number_input, st.checkbox, st.button,
st.progress, st.bar_chart, st.download_button, st.tabs, st.segmented_control
with a graceful fallback). No unsafe_allow_html, no custom HTML/CSS, no JS.

HOW TO RUN
    pip install streamlit pandas numpy
    streamlit run app.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 0. PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BasketBee — Prototype", page_icon="🐝", layout="wide")

RANDOM_SEED = 42

DISCLAIMER_TEXT = (
    "Academic prototype disclaimer: Brand names, products, prices, ratings and availability shown "
    "here are illustrative, fictional, and deterministically generated for an Executive MBA "
    "graduation project. They do not represent live listings, endorsements or commercial "
    "partnerships. No live platform APIs are connected, no live prices/ratings/stock/ETA are "
    "displayed, and no orders or payments are processed. Brand and product information reflects a "
    "manually curated illustrative catalogue based on commonly visible public retail information. "
    "BasketBee has no implied endorsement, authorization or partnership with any brand or platform "
    "shown. Natural-language interpretation below is simulated through deterministic keyword rules, "
    "not a live LLM. A real commercial product would require authorized data integrations, brand "
    "licensing where applicable, stronger consent controls and production-grade data governance."
)


def render_disclaimer() -> None:
    st.caption("⚠️ " + DISCLAIMER_TEXT)


# ---------------------------------------------------------------------------
# NATIVE UI HELPERS — no custom HTML/CSS anywhere in this file, so contrast
# is always correct in both Streamlit light and dark themes.
# ---------------------------------------------------------------------------

def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    with st.container(border=True):
        st.caption(eyebrow.upper())
        st.title(title)
        st.caption(subtitle)
        render_disclaimer()


def section_header(title: str, subtitle: str = "") -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def render_stats_line(stats: list) -> None:
    st.caption(" • ".join(stats))


# ---------------------------------------------------------------------------
# 1. GEOGRAPHY — 5 cities × 4 pincodes = 20 pincodes
# ---------------------------------------------------------------------------

CITY_PINCODES = {
    "Bengaluru": {
        "560001": "CBD / MG Road",
        "560034": "Koramangala",
        "560102": "HSR Layout",
        "560066": "Whitefield",
    },
    "Mumbai": {
        "400001": "Fort / South Mumbai",
        "400053": "Andheri West",
        "400076": "Powai",
        "400601": "Thane",
    },
    "Delhi NCR": {
        "110001": "Connaught Place",
        "110017": "Saket",
        "122001": "Gurugram",
        "201301": "Noida",
    },
    "Hyderabad": {
        "500001": "Abids",
        "500032": "Gachibowli",
        "500081": "HITEC City",
        "500072": "Kukatpally",
    },
    "Chennai": {
        "600001": "George Town",
        "600017": "T. Nagar",
        "600096": "Perungudi",
        "600119": "Sholinganallur",
    },
}

CITIES = list(CITY_PINCODES.keys())

PINCODE_TO_CITY: dict = {}
PINCODE_TO_LOCALITY: dict = {}
for _city, _pins in CITY_PINCODES.items():
    for _pin, _loc in _pins.items():
        PINCODE_TO_CITY[_pin] = _city
        PINCODE_TO_LOCALITY[_pin] = _loc

ALL_PINCODES = list(PINCODE_TO_CITY.keys())


def pincode_label(pin: str) -> str:
    return f"{pin} — {PINCODE_TO_LOCALITY[pin]}"


# City-level illustrative multipliers (fee, ETA, stock-probability, demand).
CITY_MULTIPLIERS = {
    "Bengaluru": {"fee_mult": 1.00, "eta_mult": 1.00, "stock_mult": 1.00, "demand_mult": 1.10},
    "Mumbai":    {"fee_mult": 1.10, "eta_mult": 1.15, "stock_mult": 0.97, "demand_mult": 1.15},
    "Delhi NCR": {"fee_mult": 1.05, "eta_mult": 1.08, "stock_mult": 0.98, "demand_mult": 1.05},
    "Hyderabad": {"fee_mult": 0.95, "eta_mult": 0.95, "stock_mult": 1.02, "demand_mult": 0.95},
    "Chennai":   {"fee_mult": 0.97, "eta_mult": 1.00, "stock_mult": 1.00, "demand_mult": 0.90},
}

# Pincode/locality-level illustrative multipliers layered on top of the city multiplier.
PINCODE_MULTIPLIERS = {
    "560001": {"fee_mult": 1.05, "eta_mult": 0.90, "stock_mult": 1.03, "demand_mult": 1.20},
    "560034": {"fee_mult": 1.05, "eta_mult": 0.92, "stock_mult": 1.05, "demand_mult": 1.25},
    "560102": {"fee_mult": 1.00, "eta_mult": 0.95, "stock_mult": 1.04, "demand_mult": 1.15},
    "560066": {"fee_mult": 0.95, "eta_mult": 1.10, "stock_mult": 0.97, "demand_mult": 1.00},

    "400001": {"fee_mult": 1.10, "eta_mult": 0.92, "stock_mult": 1.02, "demand_mult": 1.20},
    "400053": {"fee_mult": 1.05, "eta_mult": 0.95, "stock_mult": 1.03, "demand_mult": 1.15},
    "400076": {"fee_mult": 1.00, "eta_mult": 1.05, "stock_mult": 1.00, "demand_mult": 1.05},
    "400601": {"fee_mult": 0.92, "eta_mult": 1.15, "stock_mult": 0.95, "demand_mult": 0.95},

    "110001": {"fee_mult": 1.08, "eta_mult": 0.90, "stock_mult": 1.03, "demand_mult": 1.20},
    "110017": {"fee_mult": 1.02, "eta_mult": 0.98, "stock_mult": 1.01, "demand_mult": 1.05},
    "122001": {"fee_mult": 1.00, "eta_mult": 1.05, "stock_mult": 0.99, "demand_mult": 1.10},
    "201301": {"fee_mult": 0.95, "eta_mult": 1.08, "stock_mult": 0.97, "demand_mult": 1.00},

    "500001": {"fee_mult": 1.00, "eta_mult": 0.95, "stock_mult": 1.01, "demand_mult": 1.05},
    "500032": {"fee_mult": 1.02, "eta_mult": 0.95, "stock_mult": 1.05, "demand_mult": 1.15},
    "500081": {"fee_mult": 1.03, "eta_mult": 0.93, "stock_mult": 1.06, "demand_mult": 1.20},
    "500072": {"fee_mult": 0.93, "eta_mult": 1.05, "stock_mult": 0.98, "demand_mult": 0.95},

    "600001": {"fee_mult": 1.02, "eta_mult": 0.95, "stock_mult": 1.00, "demand_mult": 1.05},
    "600017": {"fee_mult": 1.05, "eta_mult": 0.92, "stock_mult": 1.02, "demand_mult": 1.15},
    "600096": {"fee_mult": 0.97, "eta_mult": 1.00, "stock_mult": 1.01, "demand_mult": 1.00},
    "600119": {"fee_mult": 0.95, "eta_mult": 1.05, "stock_mult": 0.99, "demand_mult": 0.95},
}

assert len(CITIES) == 5
for _c in CITIES:
    assert len(CITY_PINCODES[_c]) == 4
assert len(ALL_PINCODES) == 20
assert set(PINCODE_MULTIPLIERS.keys()) == set(ALL_PINCODES)

# ---------------------------------------------------------------------------
# 2. PLATFORMS
# ---------------------------------------------------------------------------

PLATFORMS = ["Swiggy Instamart", "Zepto", "Amazon Now", "Flipkart Minutes", "Blinkit"]
assert len(PLATFORMS) == 5

PLATFORM_BASE_FEES = {
    "Blinkit":           {"delivery_fee": 22.0, "handling_fee": 6.0, "free_delivery_threshold": 199.0, "eta": 10},
    "Zepto":             {"delivery_fee": 19.0, "handling_fee": 7.0, "free_delivery_threshold": 149.0, "eta": 12},
    "Swiggy Instamart":  {"delivery_fee": 20.0, "handling_fee": 7.0, "free_delivery_threshold": 149.0, "eta": 14},
    "Amazon Now":        {"delivery_fee": 15.0, "handling_fee": 5.0, "free_delivery_threshold": 299.0, "eta": 22},
    "Flipkart Minutes":  {"delivery_fee": 18.0, "handling_fee": 6.0, "free_delivery_threshold": 249.0, "eta": 20},
}

PLATFORM_BIAS = {"Blinkit": 1.00, "Zepto": 0.97, "Swiggy Instamart": 1.02,
                  "Amazon Now": 1.06, "Flipkart Minutes": 1.03}
PLATFORM_STOCK_PROB = {"Blinkit": 0.93, "Zepto": 0.88, "Swiggy Instamart": 0.90,
                        "Amazon Now": 0.84, "Flipkart Minutes": 0.86}

RECOMMENDATION_WEIGHTS = {
    "Price-first": {"cost": 0.55, "delivery": 0.15, "availability": 0.15, "rating": 0.10, "delivery_count": 0.05},
    "Speed-first": {"cost": 0.15, "delivery": 0.50, "availability": 0.25, "rating": 0.05, "delivery_count": 0.05},
    "Balanced":    {"cost": 0.30, "delivery": 0.25, "availability": 0.25, "rating": 0.10, "delivery_count": 0.10},
}

CFT_PER_EXTRA_DELIVERY = 50   # Convenience Friction Threshold, INR per extra delivery
SUBSTITUTION_PRICE_BAND = 0.30  # +/-30% of baseline average cost
INCOMPLETE_BASKET_PENALTY = 0.35
BRAND_LOCK_FAIL_PENALTY = 0.20
BUDGET_VIOLATION_PENALTY = 0.20
DELIVERY_TIME_VIOLATION_PENALTY = 0.15

UNIT_OPTIONS = ["pack", "kg", "g", "L", "ml", "dozen", "pc"]
BRAND_RULES = ["Any brand", "Prefer selected brand", "Selected brand only"]

# ---------------------------------------------------------------------------
# 3. CATALOGUE SPEC — exactly 100 SKUs across 12 categories.
#    (generic_product, category, pack_size, unit, price_low, price_high, [brands])
# ---------------------------------------------------------------------------

CATALOGUE_SPEC = [
    # ---- Dairy (13 SKUs / 9 generic products) ----
    ("Milk", "Dairy", "500ml", "ml", 24, 38, ["Amul", "Mother Dairy", "Nandini", "Heritage"]),
    ("Lactose-free Milk", "Dairy", "1L", "L", 95, 125, ["Amul", "Mother Dairy"]),
    ("Curd", "Dairy", "400g", "g", 35, 50, ["Amul"]),
    ("Paneer", "Dairy", "200g", "g", 70, 95, ["Mother Dairy"]),
    ("Butter", "Dairy", "100g", "g", 48, 58, ["Amul"]),
    ("Cheese Slices", "Dairy", "200g", "g", 95, 120, ["Amul"]),
    ("Greek Yogurt", "Dairy", "200g", "g", 60, 85, ["Epigamia"]),
    ("Buttermilk", "Dairy", "500ml", "ml", 18, 28, ["Nandini"]),
    ("Ghee", "Dairy", "500ml", "ml", 280, 340, ["Amul"]),

    # ---- Bakery (8 SKUs / 6 generic products) ----
    ("Bread", "Bakery", "400g", "g", 35, 52, ["Britannia", "Modern", "Harvest Gold"]),
    ("White Bread", "Bakery", "400g", "g", 32, 45, ["English Oven"]),
    ("Burger Buns", "Bakery", "6-pack", "pack", 40, 55, ["Britannia"]),
    ("Rusk", "Bakery", "200g", "g", 30, 42, ["Britannia"]),
    ("Croissant Pack", "Bakery", "4pc", "pc", 70, 95, ["Modern"]),
    ("Pav", "Bakery", "6-pack", "pack", 22, 32, ["English Oven"]),

    # ---- Produce (10 SKUs / 10 generic products) ----
    ("Bananas", "Produce", "1 dozen", "dozen", 35, 55, ["Farm Fresh"]),
    ("Apples", "Produce", "1kg", "kg", 120, 180, ["Farm Fresh"]),
    ("Onions", "Produce", "1kg", "kg", 25, 45, ["Farm Fresh"]),
    ("Tomatoes", "Produce", "1kg", "kg", 25, 50, ["Farm Fresh"]),
    ("Potatoes", "Produce", "1kg", "kg", 20, 35, ["Farm Fresh"]),
    ("Spinach", "Produce", "250g", "g", 15, 28, ["Farm Fresh"]),
    ("Carrots", "Produce", "500g", "g", 22, 38, ["Farm Fresh"]),
    ("Cucumber", "Produce", "500g", "g", 18, 30, ["Farm Fresh"]),
    ("Avocado", "Produce", "2-pack", "pack", 90, 140, ["Farm Fresh"]),
    ("Broccoli", "Produce", "300g", "g", 45, 70, ["Farm Fresh"]),

    # ---- Beverages (11 SKUs / 9 generic products) ----
    ("Tea", "Beverages", "250g", "g", 120, 160, ["Tata Tea", "Red Label"]),
    ("Coffee", "Beverages", "100g", "g", 140, 190, ["Nescafé", "Bru"]),
    ("Orange Juice", "Beverages", "1L", "L", 95, 130, ["Real"]),
    ("Cola", "Beverages", "750ml", "ml", 40, 55, ["Coca-Cola"]),
    ("Mineral Water", "Beverages", "1L", "L", 18, 25, ["Bisleri"]),
    ("Energy Drink", "Beverages", "250ml", "ml", 95, 120, ["Red Bull"]),
    ("Coconut Water", "Beverages", "1L", "L", 85, 110, ["Real"]),
    ("Iced Tea", "Beverages", "500ml", "ml", 35, 48, ["Nestea"]),
    ("Lemonade", "Beverages", "500ml", "ml", 30, 42, ["Paper Boat"]),

    # ---- Snacks (10 SKUs / 6 generic products) ----
    ("Chips", "Snacks", "52g", "g", 20, 35, ["Lay's", "Bingo", "Kurkure"]),
    ("Biscuits", "Snacks", "200g", "g", 30, 50, ["Britannia", "Parle", "Sunfeast"]),
    ("Namkeen Mix", "Snacks", "200g", "g", 45, 65, ["Haldiram's"]),
    ("Popcorn", "Snacks", "100g", "g", 35, 55, ["Act II"]),
    ("Chocolate Bar", "Snacks", "40g", "g", 40, 60, ["Cadbury"]),
    ("Trail Mix", "Snacks", "150g", "g", 130, 180, ["Happilo"]),

    # ---- Staples (12 SKUs / 7 generic products) ----
    ("Basmati Rice", "Staples", "1kg", "kg", 95, 160, ["India Gate", "Daawat", "Fortune"]),
    ("Wheat Flour", "Staples", "1kg", "kg", 45, 65, ["Aashirvaad", "Fortune"]),
    ("Sunflower Oil", "Staples", "1L", "L", 120, 165, ["Fortune", "Saffola", "Dhara"]),
    ("Toor Dal", "Staples", "1kg", "kg", 110, 150, ["Tata Sampann"]),
    ("Sugar", "Staples", "1kg", "kg", 42, 52, ["Madhur"]),
    ("Salt", "Staples", "1kg", "kg", 20, 28, ["Tata"]),
    ("Poha", "Staples", "500g", "g", 38, 52, ["Tata Sampann"]),

    # ---- Household (9 SKUs / 6 generic products) ----
    ("Detergent", "Household", "1kg", "kg", 120, 175, ["Surf Excel", "Ariel", "Tide"]),
    ("Dishwashing", "Household", "500ml", "ml", 85, 110, ["Vim", "Exo"]),
    ("Toilet Cleaner", "Household", "500ml", "ml", 65, 90, ["Harpic"]),
    ("Floor Cleaner", "Household", "1L", "L", 130, 175, ["Lizol"]),
    ("Garbage Bags", "Household", "30pc", "pack", 95, 130, ["All Time"]),
    ("Toilet Paper", "Household", "4-roll", "pack", 140, 190, ["Origami"]),

    # ---- Personal Care (9 SKUs / 5 generic products) ----
    ("Toothpaste", "Personal Care", "150g", "g", 75, 110, ["Colgate", "Pepsodent", "Close-Up"]),
    ("Shampoo", "Personal Care", "340ml", "ml", 180, 260, ["Dove", "Head & Shoulders", "Clinic Plus"]),
    ("Hand Wash", "Personal Care", "250ml", "ml", 85, 115, ["Dettol"]),
    ("Body Wash", "Personal Care", "250ml", "ml", 190, 240, ["Dove"]),
    ("Deodorant", "Personal Care", "150ml", "ml", 180, 230, ["Nivea"]),

    # ---- Baby Care (6 SKUs / 3 generic products) ----
    ("Diapers", "Baby Care", "Pack (M)", "pack", 380, 460, ["Pampers", "Huggies", "MamyPoko"]),
    ("Baby Wipes", "Baby Care", "80-pack", "pack", 150, 190, ["Huggies", "Johnson's Baby"]),
    ("Baby Lotion", "Baby Care", "200ml", "ml", 170, 210, ["Himalaya Baby"]),

    # ---- Meat & Seafood (4 SKUs / 4 generic products) ----
    ("Eggs", "Meat & Seafood", "12-pack", "pack", 70, 95, ["Farm Fresh"]),
    ("Chicken Breast", "Meat & Seafood", "500g", "g", 190, 240, ["Licious"]),
    ("Fish Fillet", "Meat & Seafood", "500g", "g", 250, 320, ["FreshToHome"]),
    ("Mutton", "Meat & Seafood", "500g", "g", 380, 450, ["Licious"]),

    # ---- Frozen (4 SKUs / 4 generic products) ----
    ("Frozen Peas", "Frozen", "500g", "g", 70, 95, ["McCain"]),
    ("Ice Cream Tub", "Frozen", "500ml", "ml", 180, 230, ["Amul"]),
    ("Frozen Fries", "Frozen", "400g", "g", 110, 145, ["McCain"]),
    ("Frozen Corn", "Frozen", "400g", "g", 85, 115, ["McCain"]),

    # ---- Health and OTC Essentials (4 SKUs / 4 generic products, non-prescription only) ----
    ("Multivitamin", "Health and OTC Essentials", "30 tabs", "pack", 250, 320, ["Himalaya"]),
    ("Adhesive Bandages", "Health and OTC Essentials", "1 pack", "pack", 45, 65, ["Band-Aid"]),
    ("ORS Sachets", "Health and OTC Essentials", "5pc", "pack", 35, 50, ["Electral"]),
    ("Antiseptic Liquid", "Health and OTC Essentials", "100ml", "ml", 55, 75, ["Dettol"]),
]

CATEGORIES = sorted({row[1] for row in CATALOGUE_SPEC})
assert len(CATEGORIES) == 12

IMPORTANT_MULTI_BRAND_GENERICS = [
    "Milk", "Bread", "Basmati Rice", "Wheat Flour", "Sunflower Oil", "Tea", "Coffee",
    "Biscuits", "Chips", "Detergent", "Dishwashing", "Toothpaste", "Shampoo",
    "Diapers", "Baby Wipes",
]

GENERIC_ALIASES = {
    "Milk": ["milk"], "Lactose-free Milk": ["lactose-free milk", "lactose free milk"],
    "Bread": ["brown bread", "bread"], "White Bread": ["white bread"],
    "Basmati Rice": ["basmati rice", "rice"], "Wheat Flour": ["wheat flour", "flour", "atta"],
    "Sunflower Oil": ["sunflower oil", "cooking oil", "oil"],
    "Tea": ["tea"], "Coffee": ["coffee"], "Biscuits": ["biscuits", "biscuit"],
    "Chips": ["chips"], "Detergent": ["detergent"],
    "Dishwashing": ["dishwashing", "dish wash", "dishwash"],
    "Toothpaste": ["toothpaste"], "Shampoo": ["shampoo"],
    "Diapers": ["diapers", "diaper"], "Baby Wipes": ["baby wipes", "wipes"],
    "Eggs": ["eggs", "egg"], "Toor Dal": ["toor dal", "dal"],
    "Sugar": ["sugar"], "Salt": ["salt"], "Ghee": ["ghee"],
    "Orange Juice": ["orange juice"], "Mineral Water": ["mineral water", "water bottle"],
    "Frozen Fries": ["fries", "frozen fries"], "Ice Cream Tub": ["ice cream"],
    "Multivitamin": ["multivitamin", "vitamins"], "ORS Sachets": ["ors"],
    "Adhesive Bandages": ["band-aid", "bandage", "bandages"],
    "Antiseptic Liquid": ["antiseptic"],
}


def _make_product_name(brand: str, generic: str, pack_size: str) -> str:
    return f"{brand} {generic} {pack_size}".strip()


@st.cache_data(show_spinner=False)
def build_product_master(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Exactly 100 product SKUs across 12 categories, with a realistic
    illustrative brand assigned to every product. Deterministic given seed."""
    rows = []
    pid = 1
    for generic, category, pack_size, unit, lo, hi, brands in CATALOGUE_SPEC:
        for brand in brands:
            rows.append({
                "product_id": f"P{pid:03d}",
                "generic_product": generic,
                "category": category,
                "brand": brand,
                "product_name": _make_product_name(brand, generic, pack_size),
                "pack_size": pack_size,
                "unit": unit,
                "price_low": lo,
                "price_high": hi,
                "keyword": generic.lower(),
            })
            pid += 1
    df = pd.DataFrame(rows)

    assert len(df) == 100, f"Expected 100 SKUs, got {len(df)}"
    assert df["product_id"].is_unique
    assert df["brand"].notna().all() and (df["brand"].str.len() > 0).all()
    for g in IMPORTANT_MULTI_BRAND_GENERICS:
        n_brands = df.loc[df["generic_product"] == g, "brand"].nunique()
        assert n_brands >= 2, f"{g} must have >= 2 brands, has {n_brands}"
    return df


def brands_for_generic(catalog_df: pd.DataFrame, generic: str) -> list:
    return sorted(catalog_df.loc[catalog_df["generic_product"] == generic, "brand"].unique().tolist())


def product_for_brand(catalog_df: pd.DataFrame, generic: str, brand: str) -> Optional[pd.Series]:
    match = catalog_df[(catalog_df["generic_product"] == generic) & (catalog_df["brand"] == brand)]
    return match.iloc[0] if not match.empty else None


def default_product_for_generic(catalog_df: pd.DataFrame, generic: str) -> pd.Series:
    """Deterministic default SKU for a generic product (used when brand = Any brand)."""
    subset = catalog_df[catalog_df["generic_product"] == generic].sort_values("product_id")
    return subset.iloc[0]


# ---------------------------------------------------------------------------
# 4. PLATFORM x PINCODE FEES  +  PLATFORM x PINCODE x PRODUCT OFFERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_platform_pincode_fees() -> pd.DataFrame:
    """Deterministic delivery fee / handling fee / ETA / free-delivery threshold for
    every (platform, pincode) combination = platform base value x city multiplier x
    pincode/locality multiplier."""
    rows = []
    for pf in PLATFORMS:
        base = PLATFORM_BASE_FEES[pf]
        for pin in ALL_PINCODES:
            city = PINCODE_TO_CITY[pin]
            cm = CITY_MULTIPLIERS[city]
            pm = PINCODE_MULTIPLIERS[pin]
            rows.append({
                "platform": pf, "pincode": pin, "city": city,
                "delivery_fee": round(base["delivery_fee"] * cm["fee_mult"] * pm["fee_mult"], 2),
                "handling_fee": round(base["handling_fee"] * cm["fee_mult"] * pm["fee_mult"], 2),
                "free_delivery_threshold": round(base["free_delivery_threshold"] * cm["fee_mult"] / 10) * 10,
                "eta": max(6, int(round(base["eta"] * cm["eta_mult"] * pm["eta_mult"]))),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_offers(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """The full embedded 100 products x 5 platforms x 20 pincodes offer matrix.
    Price and rating are platform-level (pincode-invariant, as in real quick-commerce
    catalogues); stock probability, ETA and fees vary by city and pincode. Fully
    deterministic given the fixed seed."""
    catalog_df = build_product_master(seed)
    fees_df = build_platform_pincode_fees()
    rng = np.random.default_rng(seed)

    n = len(catalog_df)
    base_price = rng.uniform(catalog_df["price_low"].values, catalog_df["price_high"].values)
    rating_base_noise = rng.normal(0, 0.15, size=n)

    chunks = []
    for pf in PLATFORMS:
        bias = PLATFORM_BIAS[pf]
        price_noise = rng.normal(0, 0.04, size=n)
        price = np.maximum(10.0, np.round(base_price * bias * (1 + price_noise), 2))
        rating = np.clip(4.2 + rating_base_noise + rng.normal(0, 0.10, size=n), 3.0, 5.0).round(1)

        pf_fees = fees_df[fees_df["platform"] == pf].set_index("pincode")
        for pin in ALL_PINCODES:
            city = PINCODE_TO_CITY[pin]
            cm = CITY_MULTIPLIERS[city]
            pm = PINCODE_MULTIPLIERS[pin]
            stock_prob = float(np.clip(PLATFORM_STOCK_PROB[pf] * cm["stock_mult"] * pm["stock_mult"], 0.05, 0.99))
            stock_roll = rng.random(n)
            in_stock = stock_roll < stock_prob
            fee_row = pf_fees.loc[pin]

            chunk = pd.DataFrame({
                "platform": pf, "city": city, "pincode": pin, "locality": PINCODE_TO_LOCALITY[pin],
                "product_id": catalog_df["product_id"].values, "product_name": catalog_df["product_name"].values,
                "generic_product": catalog_df["generic_product"].values, "brand": catalog_df["brand"].values,
                "category": catalog_df["category"].values, "price": price, "in_stock": in_stock,
                "rating": rating, "eta": fee_row["eta"], "delivery_fee": fee_row["delivery_fee"],
                "handling_fee": fee_row["handling_fee"],
                "free_delivery_threshold": fee_row["free_delivery_threshold"],
            })
            chunks.append(chunk)

    offers = pd.concat(chunks, ignore_index=True)
    assert len(offers) == len(catalog_df) * len(PLATFORMS) * len(ALL_PINCODES)
    return offers


@st.cache_data(show_spinner=False)
def build_product_avg_price(seed: int = RANDOM_SEED) -> pd.Series:
    """Baseline average cost per product_id across all platforms/pincodes — used as
    the reference for the +/-30% substitution price band."""
    offers = build_offers(seed)
    return offers.groupby("product_id")["price"].mean()


def get_offers_for(offers_df: pd.DataFrame, platform: str, pincode: str) -> pd.DataFrame:
    return offers_df[(offers_df["platform"] == platform) & (offers_df["pincode"] == pincode)]


# ---------------------------------------------------------------------------
# 5. SIMULATED "LLM" EXTRACTION — deterministic keyword/regex rules, now with
#    embedded-brand detection. This only pre-populates the structured
#    interface; the customer must validate/edit every row (Step 3).
# ---------------------------------------------------------------------------

WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "dozen": 1, "half dozen": 1}

DEFAULT_REQUEST = (
    "I need Amul lactose-free milk, Britannia brown bread, eggs and Fortune oil. "
    "Do not replace the milk brand. Deliver within 40 minutes and keep it below ₹1,500."
)

GLOBAL_LOCK_PATTERNS = [r"exact brand", r"same brand only", r"no other brand",
                        r"no substitut", r"don.?t substitute"]
ITEM_LOCK_KEYWORD_PATTERNS = [
    r"(?:do not|don.?t) replace (?:the |my )?([a-z][a-z\-]*)(?:\s+brand)?",
    r"same brand only for (?:the |my )?([a-z][a-z\-]*)",
    r"exact brand for (?:the |my )?([a-z][a-z\-]*)",
]


def _generic_match_positions(text_l: str, catalog_df: pd.DataFrame):
    """Return list of (generic_product, start_index, alias) for every generic
    product whose alias appears in the text. Longer, more specific aliases
    (e.g. "lactose-free milk") claim their text span first so a shorter,
    overlapping alias for a different generic (e.g. "milk") cannot also match
    inside it — this prevents spurious duplicate basket rows."""
    generics = catalog_df["generic_product"].unique().tolist()
    candidates = []
    for g in generics:
        aliases = sorted(GENERIC_ALIASES.get(g, [g.lower()]), key=len, reverse=True)
        for alias in aliases:
            idx = text_l.find(alias)
            if idx != -1:
                candidates.append((idx, len(alias), g, alias))
                break

    candidates.sort(key=lambda c: -c[1])  # longest span first — it claims its text
    claimed: list = []
    seen_generics: set = set()
    results = []
    for idx, length, g, alias in candidates:
        if g in seen_generics:
            continue
        start, end = idx, idx + length
        if any(not (end <= cs or start >= ce) for cs, ce in claimed):
            continue  # overlaps a span a more specific generic already claimed
        claimed.append((start, end))
        seen_generics.add(g)
        results.append((g, idx, alias))

    results.sort(key=lambda r: r[1])  # restore left-to-right reading order
    return results


def _apply_item_scoped_locks(text_l: str, matched: list) -> bool:
    """Detect phrases like "do not replace the milk brand" and lock only the
    matching basket item(s) to Selected brand only / Not allowed. Returns True
    if at least one item was scoped-locked this way."""
    locked_any = False
    for pat in ITEM_LOCK_KEYWORD_PATTERNS:
        for m in re.finditer(pat, text_l):
            keyword = m.group(1)
            for item in matched:
                generic = item["product_category"]
                g_l = generic.lower()
                aliases = GENERIC_ALIASES.get(generic, [g_l])
                if keyword in g_l or any(keyword in a for a in aliases):
                    if item["preferred_brand"] != "Any brand":
                        item["brand_priority"] = "Selected brand only"
                        item["substitution_rule"] = "Not allowed"
                        locked_any = True
    return locked_any


def _detect_brand_for_generic(text_l: str, catalog_df: pd.DataFrame, generic: str) -> Optional[str]:
    """Only checks brands that are actually valid for this generic product, to
    avoid cross-contamination between unrelated items."""
    candidates = brands_for_generic(catalog_df, generic)
    best_brand, best_idx = None, None
    for b in candidates:
        idx = text_l.find(b.lower())
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_brand, best_idx = b, idx
    return best_brand


def extract_from_text(text: str, catalog_df: pd.DataFrame, default_substitution: str = "Allowed"):
    text_l = text.lower()
    matched = []
    seen_generics = set()

    for generic, idx, alias in _generic_match_positions(text_l, catalog_df):
        if generic in seen_generics:
            continue
        seen_generics.add(generic)

        qty = 1
        m = re.search(r"(\d+)\s*(?:x\s*)?" + re.escape(alias), text_l)
        if m:
            qty = int(m.group(1))
        else:
            for word, val in sorted(WORD_NUMBERS.items(), key=lambda x: -len(x[0])):
                if re.search(r"\b" + re.escape(word) + r"\b[^.]{0,12}\b" + re.escape(alias), text_l):
                    qty = val
                    break

        brand = _detect_brand_for_generic(text_l, catalog_df, generic)
        if brand:
            row = product_for_brand(catalog_df, generic, brand)
            brand_rule = "Prefer selected brand"
            sub_rule = default_substitution
        else:
            row = default_product_for_generic(catalog_df, generic)
            brand = "Any brand"
            brand_rule = "Any brand"
            sub_rule = default_substitution

        matched.append({
            "product_category": generic, "preferred_brand": brand, "product_name": row["product_name"],
            "quantity": qty, "unit": row["unit"], "brand_priority": brand_rule, "substitution_rule": sub_rule,
        })

    # Item-scoped locks first (e.g. "do not replace the milk brand" -> only Milk is
    # locked). If no specific item was targeted but a generic lock phrase exists
    # anywhere (e.g. a bare "exact brand"), fall back to locking every branded item.
    scoped = _apply_item_scoped_locks(text_l, matched)
    if not scoped and any(re.search(p, text_l) for p in GLOBAL_LOCK_PATTERNS):
        for item in matched:
            if item["preferred_brand"] != "Any brand":
                item["brand_priority"] = "Selected brand only"
                item["substitution_rule"] = "Not allowed"

    budget = None
    bmatch = re.search(r"(?:₹|rs\.?|inr)\s?(\d[\d,]*)", text_l) or \
             re.search(r"(?:budget|around|under|below)\D{0,6}(\d[\d,]*)", text_l)
    if bmatch:
        budget = int(bmatch.group(1).replace(",", ""))

    max_minutes = None
    tmatch = re.search(r"within\s+(\d{1,3})\s*min", text_l)
    if tmatch:
        max_minutes = int(tmatch.group(1))

    return matched, budget, max_minutes


# ---------------------------------------------------------------------------
# 6. SUBSTITUTION CANDIDATE ENGINE — up to 3 ranked candidates.
# ---------------------------------------------------------------------------

def find_candidates(generic_offers: pd.DataFrame, exclude_pid: Optional[str], min_rating: float,
                     baseline_price: Optional[float], brand_rule: str, preferred_brand: Optional[str],
                     max_n: int = 3) -> list:
    """generic_offers: offers already filtered to (platform, pincode, generic_product).
    Returns up to `max_n` ranked candidate dict records satisfying category, stock,
    rating, +/-30% price band and brand-rule constraints."""
    df = generic_offers
    if exclude_pid is not None:
        df = df[df["product_id"] != exclude_pid]
    df = df[df["in_stock"] & (df["rating"] >= min_rating)]

    if baseline_price and baseline_price > 0:
        lo = baseline_price * (1 - SUBSTITUTION_PRICE_BAND)
        hi = baseline_price * (1 + SUBSTITUTION_PRICE_BAND)
        df = df[(df["price"] >= lo) & (df["price"] <= hi)]

    if brand_rule == "Selected brand only" and preferred_brand and preferred_brand != "Any brand":
        df = df[df["brand"] == preferred_brand]

    if df.empty:
        return []

    df = df.drop_duplicates(subset=["product_id"]).copy()
    if preferred_brand and preferred_brand != "Any brand" and brand_rule == "Prefer selected brand":
        df["brand_rank"] = (df["brand"] != preferred_brand).astype(int)
    else:
        df["brand_rank"] = 0
    ref = baseline_price if baseline_price and baseline_price > 0 else df["price"].mean()
    df["price_diff_pct"] = (df["price"] - ref).abs() / ref if ref else 0.0
    df = df.sort_values(by=["brand_rank", "rating", "price_diff_pct", "price", "product_name"],
                         ascending=[True, False, True, True, True])
    return df.head(max_n).to_dict("records")


def get_top_substitutes(generic: str, exclude_product_name: str, platform: str, pincode: str,
                         offers_df: pd.DataFrame, catalog_df: pd.DataFrame, min_rating: float,
                         brand_rule: str, preferred_brand: str, avg_price: pd.Series, max_n: int = 3) -> list:
    """Convenience wrapper used by the Step 5 approval portal preview (independent of a
    single platform — searches across all platforms at the chosen pincode so the customer
    can see realistic backups regardless of which platform ultimately gets recommended)."""
    excl_row = catalog_df[catalog_df["product_name"] == exclude_product_name]
    exclude_pid = excl_row.iloc[0]["product_id"] if not excl_row.empty else None
    baseline = avg_price.get(exclude_pid) if exclude_pid else None

    pool = offers_df[(offers_df["pincode"] == pincode) & (offers_df["generic_product"] == generic)]
    return find_candidates(pool, exclude_pid, min_rating, baseline, brand_rule, preferred_brand, max_n)


# ---------------------------------------------------------------------------
# 7. DETERMINISTIC COMPARISON ENGINE
# ---------------------------------------------------------------------------

@dataclass
class ItemOutcome:
    generic: str
    requested_brand: str
    requested_product: str
    quantity: int
    status: str            # "exact" | "alt_brand" | "substitute" | "unavailable"
    unit_price: float = 0.0
    rating: float = 0.0
    used_brand: str = ""
    used_product: str = ""
    reason: str = ""        # for unavailable: out_of_stock | below_rating | brand_lock | no_substitute | approval_disabled
    note: str = ""


def resolve_item(item: dict, platform: str, pincode: str, offers_df: pd.DataFrame, catalog_df: pd.DataFrame,
                  min_rating: float, avg_price: pd.Series) -> ItemOutcome:
    generic = item["product_category"]
    brand_rule = item.get("brand_priority", "Any brand")
    preferred_brand = item.get("preferred_brand", "Any brand")
    substitution_flag_allowed = item.get("substitution_rule", "Allowed") == "Allowed"
    substitution_approved = item.get("substitution_approved", True)
    substitution_allowed = substitution_flag_allowed and substitution_approved
    requested_name = item["product_name"]
    qty = max(1, int(item.get("quantity", 1)))

    generic_offers = offers_df[(offers_df["platform"] == platform) & (offers_df["pincode"] == pincode) &
                                (offers_df["generic_product"] == generic)]
    req_row = catalog_df[catalog_df["product_name"] == requested_name]
    req_pid = req_row.iloc[0]["product_id"] if not req_row.empty else None
    baseline = float(avg_price.get(req_pid)) if req_pid is not None and req_pid in avg_price.index else None

    def offer_for(pid):
        r = generic_offers[generic_offers["product_id"] == pid]
        return r.iloc[0] if not r.empty else None

    out = ItemOutcome(generic=generic, requested_brand=preferred_brand, requested_product=requested_name,
                       quantity=qty, status="unavailable")

    if brand_rule == "Selected brand only":
        row = offer_for(req_pid) if req_pid is not None else None
        if row is not None and bool(row["in_stock"]) and float(row["rating"]) >= min_rating:
            out.status, out.unit_price, out.rating = "exact", float(row["price"]), float(row["rating"])
            out.used_brand, out.used_product = row["brand"], row["product_name"]
            out.note = f"{row['product_name']} — selected-brand-only lock satisfied."
            return out
        reason = "brand_lock"
        if row is None:
            reason = "brand_lock"
        elif not bool(row["in_stock"]):
            reason = "brand_lock"
        elif float(row["rating"]) < min_rating:
            reason = "brand_lock"
        out.reason = reason
        out.note = (f"{requested_name} — selected-brand-only lock was active and no qualifying "
                    f"{preferred_brand} offer was available.")
        return out

    if preferred_brand and preferred_brand != "Any brand":
        row = offer_for(req_pid) if req_pid is not None else None
        if row is not None and bool(row["in_stock"]) and float(row["rating"]) >= min_rating:
            out.status, out.unit_price, out.rating = "exact", float(row["price"]), float(row["rating"])
            out.used_brand, out.used_product = row["brand"], row["product_name"]
            out.note = f"Preferred brand {preferred_brand} matched exactly."
            return out
        cause = "out_of_stock"
        if row is not None and float(row["rating"]) < min_rating:
            cause = "below_rating"
        if not substitution_allowed:
            out.reason = "approval_disabled" if (substitution_flag_allowed and not substitution_approved) else cause
            out.note = f"{requested_name} unavailable ({cause.replace('_', ' ')}); substitution not permitted."
            return out
        candidates = find_candidates(generic_offers, req_pid, min_rating, baseline, brand_rule, preferred_brand)
        if candidates:
            best = candidates[0]
            out.status = "alt_brand"
            out.unit_price, out.rating = float(best["price"]), float(best["rating"])
            out.used_brand, out.used_product = best["brand"], best["product_name"]
            out.note = (f"Preferred brand {preferred_brand} was unavailable. {best['brand']} was selected "
                        f"because it met the rating, price and substitution requirements.")
            return out
        out.reason = "no_substitute"
        out.note = f"{requested_name} — preferred brand unavailable and no qualifying substitute was found."
        return out

    # Any brand, no specific brand requested — open resolution across all qualifying brands.
    candidates = find_candidates(generic_offers, None, min_rating, baseline, "Any brand", None)
    if candidates:
        best = candidates[0]
        out.status = "exact"
        out.unit_price, out.rating = float(best["price"]), float(best["rating"])
        out.used_brand, out.used_product = best["brand"], best["product_name"]
        out.note = f"No brand preference — {best['brand']} {generic} selected on rating/price/availability."
        return out
    out.reason = "out_of_stock"
    out.note = f"No qualifying {generic} offer (any brand) met the in-stock/rating requirement."
    return out


def _fees_for(fees_df: pd.DataFrame, platform: str, pincode: str) -> dict:
    row = fees_df[(fees_df["platform"] == platform) & (fees_df["pincode"] == pincode)]
    if row.empty:
        return {"delivery_fee": 0.0, "handling_fee": 0.0, "free_delivery_threshold": 0.0, "eta": 30}
    r = row.iloc[0]
    return {"delivery_fee": float(r["delivery_fee"]), "handling_fee": float(r["handling_fee"]),
            "free_delivery_threshold": float(r["free_delivery_threshold"]), "eta": int(r["eta"])}


def evaluate_single_platform(basket: list, platform: str, pincode: str, offers_df: pd.DataFrame,
                              catalog_df: pd.DataFrame, fees_df: pd.DataFrame, min_rating: float,
                              avg_price: pd.Series) -> dict:
    item_cost, ratings = 0.0, []
    exact_matches, brand_matches, substitutions, unavailable = [], [], [], []

    for item in basket:
        oc = resolve_item(item, platform, pincode, offers_df, catalog_df, min_rating, avg_price)
        if oc.status == "unavailable":
            unavailable.append({"product": oc.requested_product, "reason": oc.reason, "note": oc.note})
            continue
        item_cost += oc.unit_price * oc.quantity
        ratings.append(oc.rating)
        if oc.status == "exact":
            exact_matches.append(oc.note)
        elif oc.status == "alt_brand":
            brand_matches.append(oc.note)
        elif oc.status == "substitute":
            substitutions.append(oc.note)

    fees = _fees_for(fees_df, platform, pincode)
    delivery_fee = 0.0 if item_cost >= fees["free_delivery_threshold"] else fees["delivery_fee"]
    total_fees = delivery_fee + fees["handling_fee"]
    total_cost = item_cost + total_fees
    avg_rating = float(np.mean(ratings)) if ratings else 3.5
    n_requested = len(basket)
    completion_pct = round(100 * (n_requested - len(unavailable)) / n_requested, 1) if n_requested else 0.0

    return {
        "platform": platform, "item_cost": round(item_cost, 2), "fees": round(total_fees, 2),
        "total_cost": round(total_cost, 2), "eta": fees["eta"], "rating": round(avg_rating, 2),
        "exact_matches": exact_matches, "brand_matches": brand_matches, "substitutions": substitutions,
        "unavailable": unavailable, "num_deliveries": 1, "completion_pct": completion_pct,
    }


def evaluate_split_option(basket: list, pincode: str, offers_df: pd.DataFrame, catalog_df: pd.DataFrame,
                           fees_df: pd.DataFrame, min_rating: float, avg_price: pd.Series,
                           single_options: list, max_platforms: int = 2,
                           cft: float = CFT_PER_EXTRA_DELIVERY) -> Optional[dict]:
    if not basket or len(single_options) < 2 or max_platforms < 2:
        return None
    max_platforms = min(max_platforms, len(single_options))
    ranked = sorted(single_options, key=lambda o: o["total_cost"])
    eligible = [o["platform"] for o in ranked[:max_platforms]]
    best_single = ranked[0]

    total_item_cost, used = 0.0, set()
    exact_matches, brand_matches, substitutions, unavailable, ratings = [], [], [], [], []

    for item in basket:
        best_pf, best_oc = None, None
        for pf in eligible:
            oc = resolve_item(item, pf, pincode, offers_df, catalog_df, min_rating, avg_price)
            if oc.status != "unavailable":
                if best_oc is None or oc.unit_price < best_oc.unit_price:
                    best_pf, best_oc = pf, oc
        if best_oc is None:
            unavailable.append({"product": item["product_name"], "reason": "no_substitute",
                                 "note": f"{item['product_name']} unavailable on all eligible split platforms."})
            continue
        total_item_cost += best_oc.unit_price * best_oc.quantity
        ratings.append(best_oc.rating)
        used.add(best_pf)
        note = f"{best_oc.note} ({best_pf})"
        if best_oc.status == "exact":
            exact_matches.append(note)
        elif best_oc.status == "alt_brand":
            brand_matches.append(note)
        else:
            substitutions.append(note)

    if len(used) < 2:
        return None

    total_fees, max_eta = 0.0, 0
    for pf in used:
        fees = _fees_for(fees_df, pf, pincode)
        total_fees += fees["delivery_fee"] + fees["handling_fee"]
        max_eta = max(max_eta, fees["eta"])

    split_total = total_item_cost + total_fees
    savings = best_single["total_cost"] - split_total
    required_saving = cft * (len(used) - 1)
    if savings < required_saving:
        return None

    n_requested = len(basket)
    completion_pct = round(100 * (n_requested - len(unavailable)) / n_requested, 1) if n_requested else 0.0
    avg_rating = float(np.mean(ratings)) if ratings else 3.5

    return {
        "platform": " + ".join(sorted(used)), "item_cost": round(total_item_cost, 2),
        "fees": round(total_fees, 2), "total_cost": round(split_total, 2), "eta": max_eta,
        "rating": round(avg_rating, 2), "exact_matches": exact_matches, "brand_matches": brand_matches,
        "substitutions": substitutions, "unavailable": unavailable, "num_deliveries": len(used),
        "savings_vs_single": round(savings, 2), "completion_pct": completion_pct,
    }


def _apply_ranking_penalties(df: pd.DataFrame, max_budget: float, delivery_minutes: int) -> pd.Series:
    penalty = pd.Series(0.0, index=df.index)
    penalty += (100 - df["completion_pct"]) / 100 * INCOMPLETE_BASKET_PENALTY
    brand_lock_fail = df["unavailable"].apply(lambda u: any(x.get("reason") == "brand_lock" for x in u))
    penalty += brand_lock_fail.astype(float) * BRAND_LOCK_FAIL_PENALTY
    penalty += (df["total_cost"] > max_budget).astype(float) * BUDGET_VIOLATION_PENALTY
    penalty += (df["eta"] > delivery_minutes).astype(float) * DELIVERY_TIME_VIOLATION_PENALTY
    return penalty


def run_comparison(basket: list, pincode: str, offers_df: pd.DataFrame, catalog_df: pd.DataFrame,
                    fees_df: pd.DataFrame, min_rating: float, avg_price: pd.Series, max_budget: float,
                    delivery_minutes: int, priority: str = "Balanced", max_platforms: int = 2) -> dict:
    singles = [evaluate_single_platform(basket, pf, pincode, offers_df, catalog_df, fees_df, min_rating, avg_price)
               for pf in PLATFORMS]
    split = evaluate_split_option(basket, pincode, offers_df, catalog_df, fees_df, min_rating, avg_price,
                                   singles, max_platforms=max_platforms)
    candidates = singles + ([split] if split else [])

    df = pd.DataFrame(candidates)
    df["unavailable_count"] = df["unavailable"].apply(len)

    def norm(series, invert=False):
        lo, hi = series.min(), series.max()
        out = pd.Series(0.0, index=series.index) if hi == lo else (series - lo) / (hi - lo)
        return (1 - out) if invert else out

    w = RECOMMENDATION_WEIGHTS[priority]
    base_score = (
        w["cost"] * norm(df["total_cost"]) + w["delivery"] * norm(df["eta"]) +
        w["availability"] * norm(df["unavailable_count"]) + w["rating"] * norm(df["rating"], invert=True) +
        w["delivery_count"] * norm(df["num_deliveries"])
    )
    df["match_score"] = base_score + _apply_ranking_penalties(df, max_budget, delivery_minutes)

    return {
        "all_options": df,
        "best_match": df.sort_values("match_score").iloc[0].to_dict(),
        "cheapest": df.sort_values("total_cost").iloc[0].to_dict(),
        "fastest": df.sort_values("eta").iloc[0].to_dict(),
        "split": split,
        "priority": priority,
    }


SAMPLE_GENERICS = ["Milk", "Eggs", "Bread", "Basmati Rice", "Toor Dal", "Sunflower Oil",
                   "Onions", "Tomatoes", "Bananas", "Toothpaste", "Detergent", "Dishwashing"]

BASKET_COLUMNS = ["product_category", "preferred_brand", "product_name", "quantity",
                  "unit", "brand_priority", "substitution_rule"]


def _blank_basket_item(catalog_df: pd.DataFrame, generic: str, default_substitution: str = "Allowed") -> dict:
    row = default_product_for_generic(catalog_df, generic)
    return {"product_category": generic, "preferred_brand": "Any brand", "product_name": row["product_name"],
            "quantity": 1, "unit": row["unit"], "brand_priority": "Any brand",
            "substitution_rule": default_substitution}


# ---------------------------------------------------------------------------
# 8. CUSTOMER MVP PAGE
# ---------------------------------------------------------------------------

def _init_customer_state(catalog_df: pd.DataFrame) -> None:
    st.session_state.setdefault("free_text", DEFAULT_REQUEST)
    st.session_state.setdefault("extraction_tags", [])
    st.session_state.setdefault("city", CITIES[0])
    st.session_state.setdefault("pincode", list(CITY_PINCODES[CITIES[0]].keys())[0])
    st.session_state.setdefault("min_rating", 4.0)
    st.session_state.setdefault("default_substitution", "Allowed")
    st.session_state.setdefault("basket", [_blank_basket_item(catalog_df, "Milk")])


def render_customer_mvp(catalog_df: pd.DataFrame, offers_df: pd.DataFrame, fees_df: pd.DataFrame,
                         avg_price: pd.Series) -> None:
    render_page_header(
        "AI-assisted quick-commerce comparison", "🛍️ BasketBee Shopper",
        "Describe your shopping list once, validate the structured basket, set brand and "
        "substitution preferences, and compare deterministically across five platforms.",
    )
    render_stats_line(["🧺 100 products", "🏬 5 platforms", "📍 20 pincodes / 5 cities"])

    _init_customer_state(catalog_df)

    # ---- Step 1 — Free-text request ----
    with st.container(border=True):
        section_header("Step 1 — Free-text request",
                        "Type naturally. The extractor pulls out products, brands, quantities, budget "
                        "and delivery-time hints automatically (simulated, rule-based — not a live LLM).")
        free_text = st.text_area("Free-text request", value=st.session_state.free_text, height=110,
                                  label_visibility="collapsed")
        st.session_state.free_text = free_text

        c1, c2 = st.columns([1, 1])
        interpret_clicked = c1.button("✨ Interpret and populate", type="primary", use_container_width=True)
        clear_clicked = c2.button("Clear", type="secondary", use_container_width=True)

        if interpret_clicked:
            matched, budget, max_minutes = extract_from_text(
                free_text, catalog_df, default_substitution=st.session_state.default_substitution)
            if matched:
                st.session_state.basket = matched
                tags = [f"✓ {len(matched)} products extracted"]
                if budget:
                    tags.append(f"✓ Budget ₹{budget:,}")
                    st.session_state["extracted_budget"] = budget
                if max_minutes:
                    tags.append(f"✓ Within {max_minutes} min")
                    st.session_state["extracted_minutes"] = max_minutes
                brands_found = [m["preferred_brand"] for m in matched if m["preferred_brand"] != "Any brand"]
                if brands_found:
                    tags.append(f"✓ Brands detected: {', '.join(brands_found)}")
                st.session_state.extraction_tags = tags
            else:
                st.session_state.extraction_tags = []
                st.warning("No catalogue items recognized — add items manually in Step 3.")

        if clear_clicked:
            st.session_state.free_text = ""
            st.session_state.basket = []
            st.session_state.extraction_tags = []

        if st.session_state.extraction_tags:
            st.success(" · ".join(st.session_state.extraction_tags))

    # ---- Step 2 — Structured preferences (city, pincode, budget, rating, etc.) ----
    with st.container(border=True):
        section_header("Step 2 — Structured preferences",
                        "These parameters control geography, budget, delivery-time and the minimum "
                        "acceptable product rating used by the comparison engine.")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            city = st.selectbox("City", CITIES, index=CITIES.index(st.session_state.city))
            if city != st.session_state.city:
                st.session_state.city = city
                st.session_state.pincode = list(CITY_PINCODES[city].keys())[0]
            pincode_options = list(CITY_PINCODES[st.session_state.city].keys())
            pin_idx = pincode_options.index(st.session_state.pincode) if st.session_state.pincode in pincode_options else 0
            pincode = st.selectbox("Pincode", pincode_options, index=pin_idx, format_func=pincode_label)
            st.session_state.pincode = pincode
        with pc2:
            priority = st.selectbox("Recommendation priority", list(RECOMMENDATION_WEIGHTS.keys()), index=2)
            max_budget = st.number_input("Max budget (₹)", min_value=0,
                                          value=int(st.session_state.get("extracted_budget", 1200)), step=50)
            delivery_minutes = st.number_input("Delivery minutes (max)", min_value=5,
                                                value=int(st.session_state.get("extracted_minutes", 30)), step=5)
        with pc3:
            max_platforms = st.selectbox("Max platforms (smart split)", [1, 2, 3], index=1)
            min_rating = st.slider("Minimum acceptable product rating", min_value=3.0, max_value=5.0,
                                    value=st.session_state.min_rating, step=0.1)
            st.session_state.min_rating = min_rating
            default_substitution = st.selectbox("Default substitution (new rows)", ["Allowed", "Not allowed"])
            st.session_state["default_substitution"] = default_substitution

        st.caption(f"Serving location: {city} · {pincode_label(pincode)}. The minimum rating "
                   f"({min_rating:.1f}⭐) is a strict constraint — offers and substitutes below it are rejected.")

    # ---- Step 3 — Validated structured basket ----
    generic_options = sorted(catalog_df["generic_product"].unique().tolist())
    with st.container(border=True):
        section_header("Step 3 — Validated basket",
                        "Every row is editable. Preferred brand, resolved SKU and brand rule are set "
                        "in Step 4's Brand Preference Portal directly below — native st.data_editor cannot "
                        "reliably filter one column's options based on another row's value.")

        if st.button("📦 Load monthly essentials sample"):
            st.session_state.basket = [_blank_basket_item(catalog_df, g, default_substitution)
                                        for g in SAMPLE_GENERICS]

        basket_df = pd.DataFrame(st.session_state.basket, columns=BASKET_COLUMNS) if st.session_state.basket \
            else pd.DataFrame(columns=BASKET_COLUMNS)

        edited = st.data_editor(
            basket_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "product_category": st.column_config.SelectboxColumn("Generic product / category",
                                                                       options=generic_options, required=True, width="medium"),
                "preferred_brand": st.column_config.TextColumn("Preferred brand", disabled=True, width="small"),
                "product_name": st.column_config.TextColumn("Resolved SKU", disabled=True, width="large"),
                "quantity": st.column_config.NumberColumn("Qty", min_value=1, step=1, default=1, width="small"),
                "unit": st.column_config.SelectboxColumn("Unit", options=UNIT_OPTIONS, width="small"),
                "brand_priority": st.column_config.TextColumn("Brand rule", disabled=True, width="small"),
                "substitution_rule": st.column_config.SelectboxColumn("Substitution", options=["Allowed", "Not allowed"],
                                                                       width="small"),
            },
            key="basket_editor",
        )
        edited = edited.dropna(subset=["product_category"]).reset_index(drop=True)
        edited["quantity"] = edited["quantity"].fillna(1).astype(int).clip(lower=1)
        edited["unit"] = edited["unit"].fillna("pack")
        edited["substitution_rule"] = edited["substitution_rule"].fillna("Allowed")

        rebuilt = []
        for _, r in edited.iterrows():
            category = r["product_category"]
            existing_brand = r.get("preferred_brand")
            existing_name = r.get("product_name")
            if pd.isna(existing_brand) or pd.isna(existing_name):
                default_row = default_product_for_generic(catalog_df, category)
                brand = "Any brand"
                pname = default_row["product_name"]
                rule = "Any brand"
            else:
                brand = existing_brand
                pname = existing_name
                rule = r.get("brand_priority") if not pd.isna(r.get("brand_priority")) else "Any brand"
            rebuilt.append({
                "product_category": category, "preferred_brand": brand, "product_name": pname,
                "quantity": int(r["quantity"]), "unit": r["unit"], "brand_priority": rule,
                "substitution_rule": r["substitution_rule"],
            })
        st.session_state.basket = rebuilt

    # ---- Step 4 — Customer Brand Preference Portal ----
    with st.container(border=True):
        section_header("🏷️ Step 4 — Customer Brand Preference Portal",
                        "For every basket item, choose a preferred brand (only brands relevant to that "
                        "generic product are shown) and a brand rule that governs how strictly it is enforced.")
        if not st.session_state.basket:
            st.caption("Your basket is empty — add items in Step 3 first.")
        else:
            for i, item in enumerate(st.session_state.basket):
                category = item["product_category"]
                brand_options = ["Any brand"] + brands_for_generic(catalog_df, category)
                default_brand = item.get("preferred_brand", "Any brand")
                b_idx = brand_options.index(default_brand) if default_brand in brand_options else 0

                bc1, bc2, bc3 = st.columns([2, 1.2, 1.2])
                with bc1:
                    st.caption(f"Requested: **{category}** · currently resolved SKU: {item['product_name']}")
                with bc2:
                    sel_brand = st.selectbox("Preferred brand", brand_options, index=b_idx,
                                              key=f"brand_pref_{i}_{category}")
                with bc3:
                    rule_idx = BRAND_RULES.index(item.get("brand_priority", "Any brand")) \
                        if item.get("brand_priority", "Any brand") in BRAND_RULES else 0
                    sel_rule = st.selectbox("Brand rule", BRAND_RULES, index=rule_idx,
                                             key=f"brand_rule_{i}_{category}")

                if sel_brand == "Any brand":
                    resolved_row = default_product_for_generic(catalog_df, category)
                else:
                    resolved_row = product_for_brand(catalog_df, category, sel_brand)
                    if resolved_row is None:
                        resolved_row = default_product_for_generic(catalog_df, category)

                st.session_state.basket[i]["preferred_brand"] = sel_brand
                st.session_state.basket[i]["brand_priority"] = sel_rule if sel_brand != "Any brand" else "Any brand"
                st.session_state.basket[i]["product_name"] = resolved_row["product_name"]
                st.divider()

    # ---- Step 5 — Dynamic Customer Substitution Approval Portal ----
    with st.container(border=True):
        section_header("🔍 Step 5 — Dynamic Customer Substitution Approval Portal",
                        "Top 3 eligible substitutes for every substitution-enabled item at your selected "
                        "pincode, ranked by brand-rule compliance, rating and price. Uncheck approval to "
                        "apply a strict exact-product lock for this comparison run.")
        if not st.session_state.basket:
            st.caption("Your basket is empty — add items in Step 3 first.")
        else:
            for i, item in enumerate(st.session_state.basket):
                category = item["product_category"]
                if item["substitution_rule"] != "Allowed":
                    st.warning(f"⚠️ Lock Active: Substitutions prohibited for **{item['product_name']}** "
                               f"(substitution rule set to Not allowed in Step 3).")
                    st.session_state.basket[i]["substitution_approved"] = False
                    st.divider()
                    continue

                # Cross-platform pool for the preview so it isn't tied to one single platform.
                pool = offers_df[(offers_df["pincode"] == pincode) & (offers_df["generic_product"] == category)]
                req_row = catalog_df[catalog_df["product_name"] == item["product_name"]]
                req_pid = req_row.iloc[0]["product_id"] if not req_row.empty else None
                baseline = float(avg_price.get(req_pid)) if req_pid in avg_price.index else None
                candidates = find_candidates(pool, req_pid, min_rating, baseline, item["brand_priority"],
                                              item["preferred_brand"])

                approve_key = f"approve_sub_{i}_{category}"
                st.markdown(f"**{item['product_name']}**")
                if candidates:
                    for cand in candidates:
                        cand_avg = avg_price.get(cand["product_id"], cand["price"])
                        st.write(f"✨ {cand['brand']} {cand['product_name'].replace(cand['brand'] + ' ', '', 1)} "
                                 f"— Avg ₹{cand_avg:.0f}, ⭐ {cand['rating']:.1f}")
                else:
                    st.caption("No qualifying substitutes currently available for this item at this pincode.")

                approved = st.checkbox(f"Approve substitutes for {item['product_name']}", value=True, key=approve_key)
                st.session_state.basket[i]["substitution_approved"] = approved
                if not approved:
                    st.warning("⚠️ Lock Active: substitution approval withdrawn — exact-product lock applied "
                               "for this comparison run.")
                st.divider()

    with st.container(border=True):
        validated = st.checkbox("✅ I have reviewed and validated this basket, brand preferences and "
                                 "substitution settings")
        compare_clicked = st.button("⚖️ Compare across 5 platforms", type="primary",
                                     disabled=(not st.session_state.basket or not validated))

    if compare_clicked:
        results = run_comparison(st.session_state.basket, pincode, offers_df, catalog_df, fees_df,
                                  min_rating, avg_price, max_budget, delivery_minutes,
                                  priority=priority, max_platforms=max_platforms)
        render_comparison_output(results, max_budget, delivery_minutes)


_UNAVAILABLE_REASON_LABEL = {
    "out_of_stock": "Out of stock", "below_rating": "Rating below threshold",
    "brand_lock": "Brand lock (selected-brand-only)", "no_substitute": "No qualifying substitute",
    "approval_disabled": "Customer substitution approval disabled",
}


def _platform_card(label: str, opt: Optional[dict], delivery_minutes: int, max_budget: float) -> None:
    with st.container(border=True):
        st.markdown(f"**{label}**")
        if opt is None:
            st.markdown("#### —")
            st.caption("No qualifying option")
            st.info("Savings don't clear the Convenience Friction Threshold, fewer than 2 platforms "
                     "have items in stock, or no split satisfies the rating/brand constraints.")
            return
        st.markdown(f"#### {opt['platform']}")
        st.metric("Total (incl. fees)", f"₹{opt['total_cost']:.0f}")
        st.caption(f"Items ₹{opt['item_cost']:.0f} + fees ₹{opt['fees']:.0f}")
        st.progress(min(1.0, opt["completion_pct"] / 100), text=f"Basket completion: {opt['completion_pct']:.0f}%")
        eta_flag = "🟢" if opt["eta"] <= delivery_minutes else "🟠"
        budget_flag = "🟢" if opt["total_cost"] <= max_budget else "🟠"
        st.write(f"{eta_flag} ~{int(opt['eta'])} min (limit {delivery_minutes}) · {budget_flag} budget "
                 f"₹{max_budget:,.0f} · ⭐ {opt['rating']} · 📦 {opt['num_deliveries']} delivery(ies)")
        if opt["total_cost"] > max_budget:
            st.warning(f"Exceeds your ₹{max_budget:,.0f} budget")
        if opt["eta"] > delivery_minutes:
            st.warning(f"Exceeds your {delivery_minutes} min delivery-time limit")
        if opt.get("brand_matches"):
            st.info("Preferred-brand matches / alternates: " + "; ".join(opt["brand_matches"]))
        if opt.get("substitutions"):
            st.info("Substituted: " + "; ".join(opt["substitutions"]))
        if opt.get("unavailable"):
            lines = [f"{u['product']} ({_UNAVAILABLE_REASON_LABEL.get(u['reason'], u['reason'])})"
                     for u in opt["unavailable"]]
            st.warning("Unavailable: " + "; ".join(lines))
        if not opt.get("unavailable") and not opt.get("substitutions") and not opt.get("brand_matches"):
            st.success("All items confirmed in stock at preferred brand / rating")


def render_comparison_output(results: dict, max_budget: float, delivery_minutes: int) -> None:
    section_header("Comparison across 5 platforms")
    df = results["all_options"]

    cols = st.columns(4)
    with cols[0]:
        _platform_card("🎯 Best match", results["best_match"], delivery_minutes, max_budget)
    with cols[1]:
        _platform_card("💰 Cheapest", results["cheapest"], delivery_minutes, max_budget)
    with cols[2]:
        _platform_card("⚡ Fastest", results["fastest"], delivery_minutes, max_budget)
    with cols[3]:
        _platform_card("🔀 Smart split", results["split"], delivery_minutes, max_budget)

    w = RECOMMENDATION_WEIGHTS[results["priority"]]
    bm = results["best_match"]
    st.info(
        f"**ℹ️ Why {bm['platform']} is the Best match**\n\n"
        f"**{results['priority']}** weighs cost {w['cost']*100:.0f}%, delivery time {w['delivery']*100:.0f}%, "
        f"availability {w['availability']*100:.0f}%, rating {w['rating']*100:.0f}% and delivery count "
        f"{w['delivery_count']*100:.0f}%, with additional ranking penalties for an incomplete basket, a "
        f"strict brand-lock failure, a budget violation or a delivery-time violation. Among all five "
        f"single-platform options and the best qualifying multi-platform split, **{bm['platform']}** scores "
        f"lowest overall: ₹{bm['total_cost']:.0f} total, ~{int(bm['eta'])} min, {bm['completion_pct']:.0f}% "
        f"basket completion. This is a deterministic calculation — the free-text step only interprets your "
        f"request."
    )

    section_header("Master comparison matrix")
    display_df = df[["platform", "item_cost", "fees", "total_cost", "eta", "rating",
                      "num_deliveries", "completion_pct"]].rename(columns={
        "platform": "Platform", "item_cost": "Item cost (₹)", "fees": "Fees (₹)", "total_cost": "Total (₹)",
        "eta": "ETA (min)", "rating": "Rating", "num_deliveries": "Deliveries", "completion_pct": "Completion %",
    }).sort_values("Total (₹)")
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={"Completion %": st.column_config.ProgressColumn("Completion %", min_value=0, max_value=100, format="%.0f%%")},
    )


# ---------------------------------------------------------------------------
# 9. COMPANY DASHBOARD PAGE — deterministic, fictional, illustrative only.
# ---------------------------------------------------------------------------

CATEGORY_DEMAND_BASE = {
    "Dairy": 88, "Snacks": 78, "Produce": 70, "Beverages": 66, "Staples": 74,
    "Household": 60, "Personal Care": 64, "Baby Care": 58, "Bakery": 55,
    "Meat & Seafood": 50, "Frozen": 46, "Health and OTC Essentials": 42,
}


@st.cache_data(show_spinner=False)
def build_dashboard_opportunities(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fictional, deterministic (city, pincode, category, brand)-level demand and
    fulfilment-gap dataset that feeds the Company Dashboard's priority table."""
    catalog_df = build_product_master(seed)
    offers_df = build_offers(seed)
    avg_price = build_product_avg_price(seed)

    pairs = catalog_df[["category", "brand"]].drop_duplicates()
    rows = []
    for _, pr in pairs.iterrows():
        category, brand = pr["category"], pr["brand"]
        cat_products = catalog_df[(catalog_df["category"] == category) & (catalog_df["brand"] == brand)]["product_id"]
        cat_avg_price = float(avg_price.reindex(cat_products).mean())
        for pin in ALL_PINCODES:
            city = PINCODE_TO_CITY[pin]
            locality = PINCODE_TO_LOCALITY[pin]
            cm, pm = CITY_MULTIPLIERS[city], PINCODE_MULTIPLIERS[pin]

            subset = offers_df[(offers_df["category"] == category) & (offers_df["brand"] == brand) &
                                (offers_df["pincode"] == pin)]
            fill_rate = float(subset["in_stock"].mean()) if not subset.empty else 0.0
            avg_eta = float(subset["eta"].mean()) if not subset.empty else 0.0
            unfulfilled_rate = round((1 - fill_rate) * 100, 1)

            demand_index = round(np.clip(
                CATEGORY_DEMAND_BASE.get(category, 60) * cm["demand_mult"] * pm["demand_mult"], 0, 100), 1)
            availability_gap = round(max(0.0, demand_index - fill_rate * 100), 1)
            preferred_brand_gap = round(unfulfilled_rate * 0.6, 1)
            estimated_lost_value = round((1 - fill_rate) * cat_avg_price * demand_index / 1000, 2)

            if unfulfilled_rate > 25:
                rec = f"Add {category} micro-fulfilment slot in {locality}"
            elif preferred_brand_gap > 12:
                rec = f"Onboard a secondary {brand} supplier near {locality}"
            elif avg_eta > 20:
                rec = f"Add dark-store capacity to cut ETA in {locality}"
            else:
                rec = "Maintain current assortment and monitor"

            priority_score = round(np.clip(
                0.4 * (demand_index / 100) + 0.35 * (unfulfilled_rate / 100) +
                0.25 * (availability_gap / 100), 0, 1), 2)

            rows.append({
                "City": city, "Pincode": pin, "Locality": locality, "Category": category, "Brand": brand,
                "Demand index": demand_index, "Unfulfilled rate": unfulfilled_rate,
                "Preferred-brand gap": preferred_brand_gap, "Average ETA": round(avg_eta, 1),
                "Availability gap": availability_gap, "Estimated lost value (₹L/mo)": estimated_lost_value,
                "Operational recommendation": rec, "Priority score": priority_score,
            })
    return pd.DataFrame(rows)


def render_company_dashboard(catalog_df: pd.DataFrame, offers_df: pd.DataFrame) -> None:
    render_page_header(
        "Business Intelligence", "📊 BasketBee Intelligence",
        "Location intelligence for quick-commerce companies — aggregated, anonymized demand and "
        "fulfilment signals across 5 cities and 20 pincodes. All figures are static, fictional demo values.",
    )

    account = st.selectbox("Company account (platform view)", ["Market Overview"] + PLATFORMS)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        city_choice = st.selectbox("City", ["All Cities"] + CITIES)
    with f2:
        pin_universe = ALL_PINCODES if city_choice == "All Cities" else list(CITY_PINCODES[city_choice].keys())
        pincode_choice = st.selectbox("Pincode", ["All Pincodes"] + pin_universe,
                                       format_func=lambda p: p if p == "All Pincodes" else pincode_label(p))
    with f3:
        category_choice = st.multiselect("Category", CATEGORIES, default=CATEGORIES)
    with f4:
        brand_universe = sorted(catalog_df["brand"].unique().tolist())
        brand_choice = st.multiselect("Brand", brand_universe, default=brand_universe)

    filtered = offers_df.copy()
    if account != "Market Overview":
        filtered = filtered[filtered["platform"] == account]
    if city_choice != "All Cities":
        filtered = filtered[filtered["city"] == city_choice]
    if pincode_choice != "All Pincodes":
        filtered = filtered[filtered["pincode"] == pincode_choice]
    if category_choice:
        filtered = filtered[filtered["category"].isin(category_choice)]
    if brand_choice:
        filtered = filtered[filtered["brand"].isin(brand_choice)]

    if filtered.empty:
        st.warning("No offers match the current filter combination — widen city, pincode, category or brand.")
        fill_rate = avg_eta = avg_basket = conversion = lost_value = pref_brand_fulfilment = 0.0
    else:
        fill_rate = float(filtered["in_stock"].mean()) * 100
        avg_eta = float(filtered["eta"].mean())
        in_stock_prices = filtered.loc[filtered["in_stock"], "price"]
        avg_item_price = float(in_stock_prices.mean()) if not in_stock_prices.empty else 0.0
        avg_basket = avg_item_price * 8  # illustrative basket-size assumption
        conversion = float(np.clip(22 + fill_rate * 0.22 + max(0.0, 15 - avg_eta) * 1.1, 5, 65))
        lost_value = round((100 - fill_rate) / 100 * avg_basket * 0.08 / 1000, 2)
        important_mask = filtered["generic_product"].isin(IMPORTANT_MULTI_BRAND_GENERICS)
        pref_pool = filtered[important_mask]
        pref_brand_fulfilment = float(pref_pool["in_stock"].mean()) * 100 if not pref_pool.empty else fill_rate

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Basket Fill Rate", f"{fill_rate:.1f}%")
    k2.metric("Session Conversion", f"{conversion:.1f}%")
    k3.metric("Average ETA", f"{avg_eta:.0f} min")
    k4.metric("Average Basket", f"₹{avg_basket:,.0f}")
    k5.metric("Estimated Lost Value", f"₹{lost_value:.2f}L / mo")
    st.caption(f"Preferred-brand fulfilment rate (multi-brand generics proxy): {pref_brand_fulfilment:.1f}%")

    st.write("")
    opp = build_dashboard_opportunities()
    opp_f = opp.copy()
    if city_choice != "All Cities":
        opp_f = opp_f[opp_f["City"] == city_choice]
    if pincode_choice != "All Pincodes":
        opp_f = opp_f[opp_f["Pincode"] == pincode_choice]
    if category_choice:
        opp_f = opp_f[opp_f["Category"].isin(category_choice)]
    if brand_choice:
        opp_f = opp_f[opp_f["Brand"].isin(brand_choice)]

    cc1, cc2 = st.columns(2)
    with cc1:
        with st.container(border=True):
            section_header("Category demand", "Average demand index vs. unfulfilled-request rate.")
            if not opp_f.empty:
                cat_agg = opp_f.groupby("Category")[["Demand index", "Unfulfilled rate"]].mean()
                st.bar_chart(cat_agg)
            else:
                st.caption("No data for the current filters.")
    with cc2:
        with st.container(border=True):
            section_header("Brand demand", "Average demand index by brand (filtered categories only).")
            if not opp_f.empty:
                brand_agg = opp_f.groupby("Brand")["Demand index"].mean().sort_values(ascending=False).head(15)
                st.bar_chart(brand_agg)
            else:
                st.caption("No data for the current filters.")

    with st.container(border=True):
        section_header("Priority Business Opportunities",
                        "City / pincode / category / brand catchments ranked by demand, unfulfilled rate "
                        "and availability gap.")
        if opp_f.empty:
            st.caption("No opportunities match the current filters.")
        else:
            st.dataframe(
                opp_f.sort_values("Priority score", ascending=False).head(25),
                use_container_width=True, hide_index=True,
                column_config={
                    "Priority score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=1, format="%.2f"),
                    "Unfulfilled rate": st.column_config.ProgressColumn("Unfulfilled rate", min_value=0, max_value=100, format="%.0f%%"),
                },
            )


# ---------------------------------------------------------------------------
# 10. DEMO DATA EXPLORER PAGE
# ---------------------------------------------------------------------------

def render_demo_data(catalog_df: pd.DataFrame, offers_df: pd.DataFrame) -> None:
    render_page_header(
        "Embedded Dataset", "🗄️ Demo Data Explorer",
        "The complete offline product, brand and offer matrix powering this prototype — "
        "100 SKUs across 12 categories, priced on 5 platforms across 20 pincodes.",
    )

    with st.container(border=True):
        section_header("Filters")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            search = st.text_input("🔎 Product name contains", "")
        with r1c2:
            generic_filter = st.multiselect("Generic product", sorted(catalog_df["generic_product"].unique()))
        with r1c3:
            category_filter = st.multiselect("Category", CATEGORIES)
        with r1c4:
            brand_filter = st.multiselect("Brand", sorted(catalog_df["brand"].unique()))

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            platform_filter = st.multiselect("Platform", PLATFORMS)
        with r2c2:
            city_filter = st.multiselect("City", CITIES)
        with r2c3:
            pin_universe = ALL_PINCODES if not city_filter else \
                [p for c in city_filter for p in CITY_PINCODES[c].keys()]
            pincode_filter = st.multiselect("Pincode", pin_universe, format_func=pincode_label)
        with r2c4:
            stock_filter = st.selectbox("Stock status", ["Any", "In stock only", "Out of stock only"])

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            min_rating_filter = st.slider("Minimum rating", 3.0, 5.0, 3.0, 0.1, key="demo_min_rating")
        with r3c2:
            price_range = st.slider("Price range (₹)", 0, 500, (0, 500), 5, key="demo_price_range")

    matrix = offers_df.copy()
    if search.strip():
        matrix = matrix[matrix["product_name"].str.contains(search, case=False, na=False)]
    if generic_filter:
        matrix = matrix[matrix["generic_product"].isin(generic_filter)]
    if category_filter:
        matrix = matrix[matrix["category"].isin(category_filter)]
    if brand_filter:
        matrix = matrix[matrix["brand"].isin(brand_filter)]
    if platform_filter:
        matrix = matrix[matrix["platform"].isin(platform_filter)]
    if city_filter:
        matrix = matrix[matrix["city"].isin(city_filter)]
    if pincode_filter:
        matrix = matrix[matrix["pincode"].isin(pincode_filter)]
    if stock_filter == "In stock only":
        matrix = matrix[matrix["in_stock"]]
    elif stock_filter == "Out of stock only":
        matrix = matrix[~matrix["in_stock"]]
    matrix = matrix[matrix["rating"] >= min_rating_filter]
    matrix = matrix[(matrix["price"] >= price_range[0]) & (matrix["price"] <= price_range[1])]

    with st.container(border=True):
        section_header("Platform × Pincode Offer Matrix",
                        "One row per (platform, pincode, product) offer. Empty results are handled safely.")
        st.caption(f"Showing {len(matrix):,} of {len(offers_df):,} offers")
        if matrix.empty:
            st.info("No offers match the current filters — widen your search or clear a filter.")
        else:
            display_cols = ["platform", "city", "pincode", "locality", "product_id", "product_name", "brand",
                             "category", "price", "in_stock", "rating", "eta", "delivery_fee", "handling_fee",
                             "free_delivery_threshold"]
            st.dataframe(matrix[display_cols].rename(columns={
                "platform": "Platform", "city": "City", "pincode": "Pincode", "locality": "Locality",
                "product_id": "Product ID", "product_name": "Product name", "brand": "Brand",
                "category": "Category", "price": "Price (₹)", "in_stock": "In stock", "rating": "Rating",
                "eta": "ETA (min)", "delivery_fee": "Delivery fee (₹)", "handling_fee": "Handling fee (₹)",
                "free_delivery_threshold": "Free-delivery threshold (₹)",
            }), use_container_width=True, hide_index=True)

    with st.container(border=True):
        section_header("Downloads", "CSV exports of every embedded dataset used by this prototype.")
        product_brand_map = catalog_df[["generic_product", "category", "brand", "product_id", "product_name"]] \
            .sort_values(["category", "generic_product", "brand"])
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.download_button("⬇️ Product master (CSV)", catalog_df.to_csv(index=False).encode("utf-8"),
                                "basketbee_product_master.csv", "text/csv", use_container_width=True)
        with dc2:
            st.download_button("⬇️ Product–brand mapping (CSV)", product_brand_map.to_csv(index=False).encode("utf-8"),
                                "basketbee_product_brand_mapping.csv", "text/csv", use_container_width=True)
        with dc3:
            st.download_button("⬇️ Platform-product-pincode offers (CSV)", offers_df.to_csv(index=False).encode("utf-8"),
                                "basketbee_offers.csv", "text/csv", use_container_width=True)
        with dc4:
            opp_df = build_dashboard_opportunities()
            st.download_button("⬇️ Company-dashboard opportunities (CSV)", opp_df.to_csv(index=False).encode("utf-8"),
                                "basketbee_dashboard_opportunities.csv", "text/csv", use_container_width=True)


# ---------------------------------------------------------------------------
# 11. MAIN — top navigation
# ---------------------------------------------------------------------------

NAV_ITEMS = ["🛍️ Customer MVP", "📊 Company Dashboard", "🗄️ Demo Data"]


def render_top_nav() -> str:
    st.session_state.setdefault("page", NAV_ITEMS[0])
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Navigate", NAV_ITEMS, default=st.session_state.page,
                                       label_visibility="collapsed")
        if choice:
            st.session_state.page = choice
    else:
        cols = st.columns(len(NAV_ITEMS))
        for c, item in zip(cols, NAV_ITEMS):
            kind = "primary" if st.session_state.page == item else "secondary"
            if c.button(item, use_container_width=True, type=kind, key=f"nav_{item}"):
                st.session_state.page = item
    return st.session_state.page


def main() -> None:
    catalog_df = build_product_master()
    offers_df = build_offers()
    fees_df = build_platform_pincode_fees()
    avg_price = build_product_avg_price()

    page = render_top_nav()
    st.divider()

    with st.container(border=True):
        if page == "🛍️ Customer MVP":
            render_customer_mvp(catalog_df, offers_df, fees_df, avg_price)
        elif page == "📊 Company Dashboard":
            render_company_dashboard(catalog_df, offers_df)
        else:
            render_demo_data(catalog_df, offers_df)


if __name__ == "__main__":
    main()