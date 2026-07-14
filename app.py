"""
BasketBee — Executive MBA Working Prototype (single-file, self-contained)
==========================================================================
100% native-Streamlit styled demo covering the full 5-platform, 100-product,
3-region quick-commerce comparison concept, with three views reachable from
the top navigation bar:

  1. 🛍️ Customer MVP        — dual free-text / structured basket input,
     synced, with a deterministic Best match / Cheapest / Fastest / Smart
     split comparison across Swiggy Instamart, Zepto, Amazon Now, Flipkart
     Minutes and Blinkit — plus a live substitution & brand-logic preview.
  2. 📊 Company Dashboard    — a static illustrative geospatial demand and
     fulfilment dashboard with a per-account (per-platform) toggle.
  3. 🗄️ Demo Data           — a searchable, downloadable view of the full
     embedded 100-product x 5-platform pricing matrix.

NO external APIs, NO live scraping, NO database. Every price, fee, KPI and
dashboard number below is hardcoded or generated once from a fixed random
seed, so the app is 100% offline and reproducible. The "LLM interpretation"
step is simulated with transparent keyword/regex rules — swap
`extract_from_text()` for a real LLM call beyond the prototype stage.

DESIGN NOTE — why there is no custom CSS/HTML in this file:
    Earlier drafts injected custom `<div class="bb-hero">...</div>` gradient
    banners via `st.markdown(..., unsafe_allow_html=True)`. Those blocks hardcode
    their own text colors, so whenever the *rest* of the page followed Streamlit's
    native theme (light or dark, depending on the visitor's setting), the custom
    banner text could end up unreadable against its own background. The fix here
    is structural, not another color patch: every page header, card, badge and
    callout below uses only native Streamlit elements (st.title, st.caption,
    st.container(border=True), st.subheader, st.info/warning/success/error,
    st.metric). Native elements always pick up Streamlit's current theme
    automatically, so contrast is correct regardless of light/dark mode.

HOW TO RUN
    pip install streamlit pandas numpy
    streamlit run app.py
    (if 'streamlit' isn't recognized as a command, run:
     python -m streamlit run app.py)
"""

import re
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 0. PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BasketBee — Prototype", page_icon="🐝", layout="wide")


# ---------------------------------------------------------------------------
# NATIVE UI HELPERS
# No custom HTML/CSS anywhere in this file — every helper below composes
# plain Streamlit primitives only, so theme contrast is always correct.
# ---------------------------------------------------------------------------

def render_page_header(eyebrow: str, title: str, subtitle: str):
    """Native, theme-safe page header. Wrapped in a bordered container so it
    reads as a distinct header block, using only st.caption / st.title, which
    Streamlit renders with correct contrast in both light and dark mode."""
    with st.container(border=True):
        st.caption(eyebrow.upper())
        st.title(title)
        st.caption(subtitle)


def render_stats_line(stats: list):
    """Plain-text stat summary (e.g. '100 products • 5 platforms • 3 regions'),
    replacing the old custom HTML 'pill row'."""
    st.caption(" &nbsp;•&nbsp; ".join(stats))


def section_header(title: str, subtitle: str = ""):
    """Native replacement for the old custom '.bb-card-title / .bb-card-sub'
    HTML markup — just st.subheader + st.caption."""
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# PLATFORMS, REGIONS
# ---------------------------------------------------------------------------

PLATFORMS = ["Swiggy Instamart", "Zepto", "Amazon Now", "Flipkart Minutes", "Blinkit"]

REGIONS = ["Bengaluru", "Mumbai", "Delhi NCR"]

# Per-platform baseline fee/ETA structure (illustrative, hardcoded, offline).
# Roughly reflects known 2025-26 market positioning: Blinkit = deepest network /
# fastest ETA; Instamart/Zepto close #2-3; Amazon Now & Flipkart Minutes newer
# entrants with slightly slower ETA and higher variability while they scale.
PLATFORM_BASE_FEES = {
    "Blinkit":           {"delivery_fee": 22.0, "handling_fee": 6.0, "free_delivery_threshold": 199.0, "eta": 10},
    "Zepto":             {"delivery_fee": 19.0, "handling_fee": 7.0, "free_delivery_threshold": 149.0, "eta": 12},
    "Swiggy Instamart":  {"delivery_fee": 20.0, "handling_fee": 7.0, "free_delivery_threshold": 149.0, "eta": 14},
    "Amazon Now":        {"delivery_fee": 15.0, "handling_fee": 5.0, "free_delivery_threshold": 299.0, "eta": 22},
    "Flipkart Minutes":  {"delivery_fee": 18.0, "handling_fee": 6.0, "free_delivery_threshold": 249.0, "eta": 20},
}

# Region multipliers applied on top of the platform baseline (fee & ETA).
REGION_MULTIPLIERS = {
    "Bengaluru": {"fee_mult": 1.00, "eta_mult": 1.00},
    "Mumbai":    {"fee_mult": 1.08, "eta_mult": 1.12},
    "Delhi NCR": {"fee_mult": 1.05, "eta_mult": 1.05},
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

CFT_PER_EXTRA_DELIVERY = 50  # Convenience Friction Threshold, INR per extra delivery

# ---------------------------------------------------------------------------
# CATALOGUE — 100 products across 12 categories (hardcoded, offline)
# ---------------------------------------------------------------------------

CATEGORY_ITEMS = {
    "Dairy": ["Toned Milk 500ml", "Full Cream Milk 1L", "Curd 400g", "Paneer 200g", "Butter 100g",
              "Cheese Slices 200g", "Greek Yogurt 200g", "Buttermilk 500ml", "Ghee 500ml", "Lactose-free Milk 1L"],
    "Bakery": ["Whole Wheat Bread 400g", "White Bread 400g", "Multigrain Bread 400g", "Burger Buns 6-pack",
               "Rusk 200g", "Bread Sticks 100g", "Croissant Pack 4pc", "Pav 6-pack"],
    "Produce": ["Bananas 1 dozen", "Apples 1kg", "Onions 1kg", "Tomatoes 1kg", "Potatoes 1kg",
                "Spinach 250g", "Carrots 500g", "Cucumber 500g", "Avocado 2-pack", "Broccoli 300g"],
    "Beverages": ["Orange Juice 1L", "Cola 750ml", "Green Tea 25 bags", "Instant Coffee 100g",
                  "Mineral Water 1L", "Energy Drink 250ml", "Coconut Water 1L", "Iced Tea 500ml", "Lemonade 500ml",
                  "Buttermilk Drink 250ml"],
    "Snacks": ["Potato Chips 52g", "Chocolate Bar 40g", "Biscuits 200g", "Namkeen Mix 200g", "Popcorn 100g",
               "Trail Mix 150g", "Granola Bar 4pc", "Peanut Butter 250g", "Almond Butter 250g", "Cookies 150g"],
    "Staples": ["Basmati Rice 1kg", "Toor Dal 1kg", "Sunflower Oil 1L", "Wheat Flour 1kg", "Sugar 1kg",
                "Salt 1kg", "Besan 500g", "Poha 500g", "Oats 500g", "Quinoa 500g"],
    "Household": ["Dish Wash Gel 500ml", "Detergent Powder 1kg", "Toilet Cleaner 500ml", "Floor Cleaner 1L",
                  "Garbage Bags 30pc", "Air Freshener 300ml", "Mosquito Repellent", "Toilet Paper 4-roll",
                  "Dishwashing Bar 200g"],
    "Personal Care": ["Toothpaste 150g", "Shampoo 340ml", "Hand Wash 250ml", "Hand Sanitizer 200ml",
                       "Body Wash 250ml", "Face Wash 100g", "Deodorant 150ml", "Conditioner 340ml"],
    "Baby care": ["Diapers Pack (M)", "Baby Wipes 80-pack", "Baby Lotion 200ml", "Baby Shampoo 200ml",
                  "Baby Food Jar", "Baby Powder 200g"],
    "Meat & Seafood": ["Chicken Breast 500g", "Mutton 500g", "Fish Fillet 500g", "Eggs 12-pack",
                       "Prawns 250g", "Chicken Sausages 250g", "Chicken Drumsticks 500g"],
    "Frozen": ["Frozen Peas 500g", "Ice Cream Tub 500ml", "Frozen Paratha 5pc", "Frozen Fries 400g",
               "Frozen Corn 400g", "Frozen Veg Nuggets 400g"],
    "Pharmacy": ["Multivitamin 30 tabs", "Pain Relief Gel", "Digestive Tablets", "ORS Sachets 5pc",
                "Band-aid Pack", "Antiseptic Liquid 100ml"],
}

CATEGORY_PRICE_RANGE = {
    "Dairy": (25, 110), "Bakery": (30, 60), "Produce": (18, 90), "Beverages": (40, 140),
    "Snacks": (25, 260), "Staples": (60, 210), "Household": (50, 230), "Personal Care": (55, 260),
    "Baby care": (140, 450), "Meat & Seafood": (150, 340), "Frozen": (90, 220), "Pharmacy": (35, 180),
}


def _keyword_from_name(name: str) -> str:
    # Strip trailing pack-size / quantity tokens to get a search-friendly keyword,
    # e.g. "Whole Wheat Bread 400g" -> "whole wheat bread"
    cleaned = re.sub(r"\s*\(?\b[\d.]+\s*(g|kg|ml|l|pc|pack|tabs?|dozen|pk|-pack|roll)\b\)?\s*$", "",
                      name, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\d+-?pack$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.lower()


def build_catalog() -> pd.DataFrame:
    rows = []
    pid = 1
    for category, items in CATEGORY_ITEMS.items():
        for name in items:
            rows.append({
                "product_id": f"P{pid:03d}", "name": name, "category": category,
                "keyword": _keyword_from_name(name),
            })
            pid += 1
    df = pd.DataFrame(rows)
    assert len(df) == 100, f"Expected 100 products, got {len(df)}"
    assert df["product_id"].is_unique
    return df


def build_platform_catalog(catalog_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """5-platform price / stock / rating matrix for every SKU (seeded, fully offline)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, prod in catalog_df.iterrows():
        lo, hi = CATEGORY_PRICE_RANGE[prod.category]
        base_price = rng.uniform(lo, hi)
        for pf in PLATFORMS:
            noise = rng.normal(0, 0.05)
            price = max(10.0, round(base_price * PLATFORM_BIAS[pf] * (1 + noise), 2))
            in_stock = bool(rng.random() < PLATFORM_STOCK_PROB[pf])
            rating = float(np.clip(rng.normal(4.2, 0.3), 3.0, 5.0))
            rows.append({"platform": pf, "product_id": prod.product_id, "name": prod["name"],
                         "category": prod.category, "price": price, "in_stock": in_stock,
                         "rating": round(rating, 1)})
    out = pd.DataFrame(rows)
    assert len(out) == len(catalog_df) * len(PLATFORMS)
    return out


def get_platform_fees(platform: str, region: str) -> dict:
    base = PLATFORM_BASE_FEES[platform]
    mult = REGION_MULTIPLIERS[region]
    return {
        "delivery_fee": round(base["delivery_fee"] * mult["fee_mult"], 2),
        "handling_fee": round(base["handling_fee"] * mult["fee_mult"], 2),
        "free_delivery_threshold": base["free_delivery_threshold"],
        "eta": max(6, int(round(base["eta"] * mult["eta_mult"]))),
    }


# ---------------------------------------------------------------------------
# SIMULATED "LLM" EXTRACTION
# ---------------------------------------------------------------------------

WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "dozen": 1, "half dozen": 1}

DEFAULT_REQUEST = (
    "I need breakfast groceries for four people for three days. Deliver within 40 "
    "minutes, keep it below ₹1,500, and do not replace my lactose-free milk."
)


def extract_from_text(text: str, catalog_df: pd.DataFrame, default_substitution: str = "Allowed"):
    text_l = text.lower()
    matched, seen = [], set()

    for _, prod in catalog_df.sort_values("keyword", key=lambda s: -s.str.len()).iterrows():
        kw = prod.keyword
        if not kw or kw in seen or kw not in text_l:
            continue
        if any(kw in s for s in seen if kw != s):
            continue
        qty = 1
        m = re.search(r"(\d+)\s*(?:x\s*)?" + re.escape(kw), text_l)
        if m:
            qty = int(m.group(1))
        else:
            for word, val in sorted(WORD_NUMBERS.items(), key=lambda x: -len(x[0])):
                if re.search(r"\b" + re.escape(word) + r"\b[^.]{0,12}\b" + re.escape(kw), text_l):
                    qty = val
                    break
        matched.append({"product_name": prod["name"], "quantity": qty, "unit": "pack",
                         "brand_priority": "No preference", "substitution_rule": default_substitution})
        seen.add(kw)

    # Budget: look for an explicit currency/₹ cue near the number, or the words
    # "budget"/"around"/"under"/"below". Deliberately excludes "within", which
    # is reserved for the delivery-time clause below — otherwise a phrase like
    # "within 40 minutes ... below ₹1,500" incorrectly grabs the "40" as the
    # budget because it appears earlier in the sentence. Numbers may contain
    # thousands commas (e.g. "1,500").
    budget = None
    bmatch = re.search(r"(?:₹|rs\.?|inr)\s?(\d[\d,]*)", text_l) or \
             re.search(r"(?:budget|around|under|below)\D{0,6}(\d[\d,]*)", text_l)
    if bmatch:
        budget = int(bmatch.group(1).replace(",", ""))

    max_minutes = None
    tmatch = re.search(r"within\s+(\d{1,3})\s*min", text_l)
    if tmatch:
        max_minutes = int(tmatch.group(1))

    if re.search(r"no substitut|don.?t substitute|exact brand|same brand only", text_l):
        for m in matched:
            m["substitution_rule"] = "Not allowed"

    return matched, budget, max_minutes


# ---------------------------------------------------------------------------
# DETERMINISTIC COMPARISON ENGINE (generalized to 5 platforms + N-way split)
# ---------------------------------------------------------------------------

def evaluate_single_platform(basket_df, platform, region, platform_catalog_df):
    item_cost, unavailable, substitutions, ratings = 0.0, [], [], []
    pf_catalog = platform_catalog_df[platform_catalog_df.platform == platform]

    for row in basket_df.itertuples():
        pc = pf_catalog[pf_catalog.product_id == row.product_id]
        if pc.empty:
            unavailable.append(row.product_name)
            continue
        pc_row = pc.iloc[0]
        if pc_row.in_stock:
            item_cost += pc_row.price * row.quantity
            ratings.append(pc_row.rating)
        elif row.substitution_rule == "Allowed":
            subs = pf_catalog[(pf_catalog.category == pc_row.category) & (pf_catalog.in_stock) &
                               (pf_catalog.product_id != row.product_id)]
            if not subs.empty:
                sub = subs.sort_values("price").iloc[0]
                item_cost += sub.price * row.quantity
                substitutions.append(f"{row.product_name} -> {sub['name']}")
                ratings.append(sub.rating)
            else:
                unavailable.append(row.product_name)
        else:
            unavailable.append(row.product_name)

    fees = get_platform_fees(platform, region)
    delivery_fee = 0.0 if item_cost >= fees["free_delivery_threshold"] else fees["delivery_fee"]
    total_fees = delivery_fee + fees["handling_fee"]
    total_cost = item_cost + total_fees
    avg_rating = float(np.mean(ratings)) if ratings else 3.5
    n_requested = len(basket_df)
    completion_pct = round(100 * (n_requested - len(unavailable)) / n_requested, 1) if n_requested else 0.0

    return {"platform": platform, "item_cost": round(item_cost, 2), "fees": round(total_fees, 2),
            "total_cost": round(total_cost, 2), "eta": fees["eta"], "rating": round(avg_rating, 2),
            "unavailable": unavailable, "substitutions": substitutions, "num_deliveries": 1,
            "completion_pct": completion_pct}


def evaluate_split_option(basket_df, region, platform_catalog_df, single_options,
                           max_platforms=2, cft=CFT_PER_EXTRA_DELIVERY):
    if basket_df.empty or len(single_options) < 2 or max_platforms < 2:
        return None
    max_platforms = min(max_platforms, len(single_options))
    best_single = min(single_options, key=lambda o: o["total_cost"])
    ranked = sorted(single_options, key=lambda o: o["total_cost"])
    eligible = [o["platform"] for o in ranked[:max_platforms]]
    eligible_cats = {pf: platform_catalog_df[platform_catalog_df.platform == pf] for pf in eligible}

    total_item_cost, unavailable, substitutions, used = 0.0, [], [], set()
    for row in basket_df.itertuples():
        candidates = []
        for pf in eligible:
            pc = eligible_cats[pf][eligible_cats[pf].product_id == row.product_id]
            if not pc.empty and pc.iloc[0].in_stock:
                candidates.append((pf, pc.iloc[0].price))
        if candidates:
            candidates.sort(key=lambda c: c[1])
            pf, price = candidates[0]
            total_item_cost += price * row.quantity
            used.add(pf)
        elif row.substitution_rule == "Allowed":
            found = False
            for pf in eligible:
                pf_cat = eligible_cats[pf]
                pc_any = pf_cat[pf_cat.product_id == row.product_id]
                if pc_any.empty:
                    continue
                cat_name = pc_any.iloc[0].category
                subs = pf_cat[(pf_cat.category == cat_name) & (pf_cat.in_stock) &
                               (pf_cat.product_id != row.product_id)]
                if not subs.empty:
                    sub = subs.sort_values("price").iloc[0]
                    total_item_cost += sub.price * row.quantity
                    substitutions.append(f"{row.product_name} -> {sub['name']} ({pf})")
                    used.add(pf)
                    found = True
                    break
            if not found:
                unavailable.append(row.product_name)
        else:
            unavailable.append(row.product_name)

    if len(used) < 2:
        return None  # everything landed on one platform anyway — no genuine split

    total_fees, max_eta = 0.0, 0
    for pf in used:
        fees = get_platform_fees(pf, region)
        total_fees += fees["delivery_fee"] + fees["handling_fee"]
        max_eta = max(max_eta, fees["eta"])

    split_total = total_item_cost + total_fees
    savings = best_single["total_cost"] - split_total
    required_saving = cft * (len(used) - 1)
    if savings < required_saving:
        return None

    avg_rating = np.mean([o["rating"] for o in single_options if o["platform"] in used])
    n_requested = len(basket_df)
    completion_pct = round(100 * (n_requested - len(unavailable)) / n_requested, 1) if n_requested else 0.0

    return {"platform": " + ".join(sorted(used)), "item_cost": round(total_item_cost, 2),
            "fees": round(total_fees, 2), "total_cost": round(split_total, 2), "eta": max_eta,
            "rating": round(float(avg_rating), 2), "unavailable": unavailable, "substitutions": substitutions,
            "num_deliveries": len(used), "savings_vs_single": round(savings, 2), "completion_pct": completion_pct}


def run_comparison(basket_df, region, platform_catalog_df, priority="Balanced", max_platforms=2):
    singles = [evaluate_single_platform(basket_df, pf, region, platform_catalog_df) for pf in PLATFORMS]
    split = evaluate_split_option(basket_df, region, platform_catalog_df, singles, max_platforms=max_platforms)
    candidates = singles + ([split] if split else [])

    df = pd.DataFrame(candidates)
    df["unavailable_count"] = df["unavailable"].apply(len)

    def norm(series, invert=False):
        lo, hi = series.min(), series.max()
        out = pd.Series(0.0, index=series.index) if hi == lo else (series - lo) / (hi - lo)
        return (1 - out) if invert else out

    w = RECOMMENDATION_WEIGHTS[priority]
    df["match_score"] = (
        w["cost"] * norm(df["total_cost"]) + w["delivery"] * norm(df["eta"]) +
        w["availability"] * norm(df["unavailable_count"]) + w["rating"] * norm(df["rating"], invert=True) +
        w["delivery_count"] * norm(df["num_deliveries"])
    )

    return {
        "all_options": df,
        "best_match": df.sort_values("match_score").iloc[0].to_dict(),
        "cheapest": df.sort_values("total_cost").iloc[0].to_dict(),
        "fastest": df.sort_values("eta").iloc[0].to_dict(),
        "split": split,
        "priority": priority,
    }

MONTHLY_ESSENTIALS_SAMPLE = [
    "Full Cream Milk 1L", "Eggs 12-pack", "Whole Wheat Bread 400g", "Basmati Rice 1kg",
    "Toor Dal 1kg", "Sunflower Oil 1L", "Onions 1kg", "Tomatoes 1kg", "Bananas 1 dozen",
    "Toothpaste 150g", "Dish Wash Gel 500ml", "Detergent Powder 1kg",
]

UNIT_OPTIONS = ["pack", "kg", "g", "L", "ml", "dozen", "pc"]

# ---------------------------------------------------------------------------
# SUBSTITUTION & BRAND LOGIC PREVIEW HELPERS
# ---------------------------------------------------------------------------

def get_substitution_alternatives(product_name: str, catalog_df: pd.DataFrame, max_alts: int = 2):
    """Look up the product's category in the 100-item index and return up to
    `max_alts` other product names from that same category — these are the
    alternatives BasketBee would evaluate as backup choices if the requested
    item is out of stock on a given platform."""
    match = catalog_df.loc[catalog_df["name"] == product_name]
    if match.empty:
        return []
    category = match.iloc[0]["category"]
    alternatives = catalog_df.loc[
        (catalog_df["category"] == category) & (catalog_df["name"] != product_name), "name"
    ].tolist()
    return alternatives[:max_alts]


def render_substitution_preview(catalog_df: pd.DataFrame):
    """🔍 Live Substitution & Brand Logic Preview — shown right underneath
    Step 3's validated basket table. For every row currently in
    st.session_state.basket: if substitution_rule == "Allowed", show which
    alternative products from the 100-item index would be evaluated as
    backups; if "Not allowed", show a strict lock tag."""
    with st.container(border=True):
        section_header(
            "🔍 Live Substitution & Brand Logic Preview",
            "Based on the substitution rule set for each row in Step 3, here is exactly what "
            "BasketBee will do if a platform runs out of stock.",
        )

        basket = st.session_state.get("basket", [])
        if not basket:
            st.caption("Add at least one item to Step 3's basket to preview substitution logic.")
            return

        for item in basket:
            product_name = item.get("product_name")
            if not product_name:
                continue
            rule = item.get("substitution_rule", "Allowed")

            if rule == "Allowed":
                alternatives = get_substitution_alternatives(product_name, catalog_df)
                if alternatives:
                    alt_text = " or ".join(alternatives)
                    st.info(
                        f"If **{product_name}** is out of stock, BasketBee will evaluate "
                        f"**{alt_text}** as backup choices."
                    )
                else:
                    st.info(f"**{product_name}** allows substitution, but no alternative products "
                            f"exist in its category in the 100-item index.")
            else:
                st.warning(f"⚠️ Lock Active: Substitutions prohibited for **{product_name}**.")


# ---------------------------------------------------------------------------
# CUSTOMER MVP PAGE
# ---------------------------------------------------------------------------

def render_customer_mvp(catalog_df, platform_catalog_df):
    render_page_header(
        "Quick-commerce comparison, simplified", "🛍️ BasketBee Shopper",
        "Describe your shopping list once, validate the structured basket, and compare "
        "it deterministically across all five quick-commerce platforms.",
    )
    render_stats_line(["🧺 100 products", "🏬 5 platforms", "📍 3 regions"])

    st.session_state.setdefault("free_text", DEFAULT_REQUEST)
    st.session_state.setdefault("extraction_tags", [])
    st.session_state.setdefault(
        "basket",
        [{"product_name": n, "quantity": 1, "unit": "pack", "brand_priority": "No preference",
          "substitution_rule": "Allowed"} for n in [catalog_df.iloc[0]["name"]]],
    )

    left, right = st.columns([1.2, 0.8])

    with left:
        with st.container(border=True):
            section_header("Step 1 — Free-text request",
                            "Type naturally. The extractor pulls out products, quantities, budget "
                            "and delivery-time hints automatically.")
            free_text = st.text_area("Free-text request", value=st.session_state.free_text, height=130,
                                      label_visibility="collapsed")
            st.session_state.free_text = free_text

            c1, c2 = st.columns([1, 1])
            interpret_clicked = c1.button("✨ Interpret and populate", type="primary", use_container_width=True)
            clear_clicked = c2.button("Clear", type="secondary", use_container_width=True)

            if interpret_clicked:
                matched, budget, max_minutes = extract_from_text(
                    free_text, catalog_df, default_substitution=st.session_state.get("default_substitution", "Allowed"))
                if matched:
                    st.session_state.basket = matched
                    tags = [f"✓ {len(matched)} products extracted"]
                    if budget:
                        tags.append(f"✓ Budget ₹{budget:,}")
                        st.session_state["extracted_budget"] = budget
                    if max_minutes:
                        tags.append(f"✓ Within {max_minutes} min")
                        st.session_state["extracted_minutes"] = max_minutes
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

    with right:
        with st.container(border=True):
            section_header("Step 2 — Structured preferences",
                            "These parameters control how the comparison engine scores and "
                            "filters the five platforms.")
            pc1, pc2 = st.columns(2)
            with pc1:
                region = st.selectbox("Region", REGIONS)
                priority = st.selectbox("Recommendation priority", list(RECOMMENDATION_WEIGHTS.keys()), index=2)
                max_budget = st.number_input("Max budget (₹)", min_value=0,
                                              value=int(st.session_state.get("extracted_budget", 1200)), step=50)
            with pc2:
                delivery_minutes = st.number_input("Delivery minutes (max)", min_value=5,
                                                    value=int(st.session_state.get("extracted_minutes", 30)), step=5)
                max_platforms = st.selectbox("Max platforms (smart split)", [1, 2, 3], index=1)
                default_substitution = st.selectbox("Default substitution", ["Allowed", "Not allowed"])
                st.session_state["default_substitution"] = default_substitution

    with st.container(border=True):
        section_header("Step 3 — Validated basket",
                        "Every row is editable. Choose from the full 100-item catalogue, adjust "
                        "quantity, unit, brand priority and substitution rules.")

        if st.button("📦 Load monthly essentials sample"):
            st.session_state.basket = [
                {"product_name": n, "quantity": 1, "unit": "pack",
                 "brand_priority": "No preference", "substitution_rule": default_substitution if 'default_substitution' in dir() else "Allowed"}
                for n in MONTHLY_ESSENTIALS_SAMPLE
            ]

        columns = ["product_name", "quantity", "unit", "brand_priority", "substitution_rule"]
        basket_df = pd.DataFrame(st.session_state.basket, columns=columns) if st.session_state.basket \
            else pd.DataFrame(columns=columns)
        product_options = catalog_df["name"].tolist()

        edited = st.data_editor(
            basket_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "product_name": st.column_config.SelectboxColumn("Product (100-item catalogue)",
                                                                   options=product_options, required=True, width="large"),
                "quantity": st.column_config.NumberColumn("Qty", min_value=1, step=1, default=1, width="small"),
                "unit": st.column_config.SelectboxColumn("Unit", options=UNIT_OPTIONS, width="small"),
                "brand_priority": st.column_config.SelectboxColumn("Brand priority",
                                                                     options=["No preference", "Locked to listed brand"],
                                                                     width="medium"),
                "substitution_rule": st.column_config.SelectboxColumn("Substitution", options=["Allowed", "Not allowed"],
                                                                       width="medium"),
            },
            key="basket_editor",
        )
        edited = edited.dropna(subset=["product_name"]).copy()
        name_to_id = dict(zip(catalog_df["name"], catalog_df["product_id"]))
        edited["product_id"] = edited["product_name"].map(name_to_id)
        edited["quantity"] = edited["quantity"].fillna(1).astype(int).clip(lower=1)
        edited["unit"] = edited["unit"].fillna("pack")
        edited["brand_priority"] = edited["brand_priority"].fillna("No preference")
        edited["substitution_rule"] = edited["substitution_rule"].fillna("Allowed")
        st.session_state.basket = edited[columns].to_dict("records")

        st.divider()
        validated = st.checkbox("✅ I have reviewed and validated this basket")
        compare_clicked = st.button("⚖️ Compare across 5 platforms", type="primary",
                                     disabled=(edited.empty or not validated))

    # 🔍 Live Substitution & Brand Logic Preview — sits right underneath
    # Step 3's validated basket table, always reflecting its current state.
    render_substitution_preview(catalog_df)

    if compare_clicked:
        results = run_comparison(edited, region, platform_catalog_df, priority=priority, max_platforms=max_platforms)
        render_comparison_output(results, max_budget, delivery_minutes)


def _platform_card(label, opt, delivery_minutes, max_budget):
    with st.container(border=True):
        st.markdown(f"**{label}**")
        if opt is None:
            st.markdown("#### —")
            st.caption("No qualifying option")
            st.info("Savings don't clear the ₹50-per-extra-delivery Convenience Friction Threshold, "
                    "or fewer than 2 platforms have any items in stock.")
            return
        st.markdown(f"#### {opt['platform']}")
        st.metric("Total (incl. fees)", f"₹{opt['total_cost']:.0f}")
        st.caption(f"Items ₹{opt['item_cost']:.0f} + fees ₹{opt['fees']:.0f}")
        st.progress(min(1.0, opt["completion_pct"] / 100), text=f"Basket completion: {opt['completion_pct']:.0f}%")
        eta_flag = "🟢" if opt["eta"] <= delivery_minutes else "🟠"
        st.write(f"{eta_flag} ~{int(opt['eta'])} min (limit {delivery_minutes}) · ⭐ {opt['rating']} · "
                 f"📦 {opt['num_deliveries']} delivery(ies)")
        if opt["total_cost"] > max_budget:
            st.warning(f"Exceeds your ₹{max_budget:,.0f} budget")
        if opt.get("unavailable"):
            st.warning("Unavailable: " + ", ".join(opt["unavailable"]))
        if opt.get("substitutions"):
            st.info("Substituted: " + "; ".join(opt["substitutions"]))
        if not opt.get("unavailable") and not opt.get("substitutions"):
            st.success("All items confirmed in stock")


def render_comparison_output(results, max_budget, delivery_minutes):
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
        f"{w['delivery_count']*100:.0f}%. Among all five single-platform options and the best qualifying "
        f"multi-platform split, **{bm['platform']}** scores lowest overall: ₹{bm['total_cost']:.0f} total, "
        f"~{int(bm['eta'])} min, {bm['completion_pct']:.0f}% basket completion. This is a deterministic "
        f"calculation — the free-text step only interprets your request."
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
# COMPANY DASHBOARD PAGE
# ---------------------------------------------------------------------------

ACCOUNT_METRICS = {
    "Market Overview":  {"fill_rate": 82.4, "conversion": 34.1, "avg_eta": 14, "avg_basket": 1180, "lost_value": 4.8},
    "Blinkit":          {"fill_rate": 91.2, "conversion": 41.5, "avg_eta": 10, "avg_basket": 1340, "lost_value": 2.1},
    "Zepto":            {"fill_rate": 86.7, "conversion": 38.2, "avg_eta": 12, "avg_basket": 1260, "lost_value": 3.0},
    "Swiggy Instamart": {"fill_rate": 84.9, "conversion": 36.0, "avg_eta": 14, "avg_basket": 1210, "lost_value": 3.6},
    "Amazon Now":       {"fill_rate": 74.3, "conversion": 24.6, "avg_eta": 22, "avg_basket": 1050, "lost_value": 6.4},
    "Flipkart Minutes": {"fill_rate": 77.8, "conversion": 27.9, "avg_eta": 20, "avg_basket": 1090, "lost_value": 5.5},
}

DEMAND_BY_CATEGORY = pd.DataFrame({
    "Demand index": [92, 74, 68, 61, 58, 54, 49, 41],
    "Unfulfilled rate": [14, 22, 18, 27, 19, 24, 31, 21],
}, index=["Dairy", "Snacks", "Produce", "Baby care", "Beverages", "Household", "Personal Care", "Staples"])

PRIORITY_OPPORTUNITIES = pd.DataFrame([
    {"Geographic catchment": "Bengaluru — HSR / Koramangala", "Latent supply signal": "High demand, low dairy availability",
     "Operational remediation": "Add dairy micro-fulfilment slot", "Priority score": 0.89},
    {"Geographic catchment": "Mumbai — Andheri / Powai", "Latent supply signal": "Baby-care stockouts recurring weekly",
     "Operational remediation": "Onboard 2nd baby-care supplier", "Priority score": 0.81},
    {"Geographic catchment": "Delhi NCR — Gurugram", "Latent supply signal": "Delivery gap exceeds 18 min at peak",
     "Operational remediation": "Add dark-store capacity", "Priority score": 0.77},
    {"Geographic catchment": "Bengaluru — Whitefield", "Latent supply signal": "High fee sensitivity, order abandonment",
     "Operational remediation": "Test lower small-cart fee threshold", "Priority score": 0.68},
    {"Geographic catchment": "Mumbai — Thane", "Latent supply signal": "Produce substitution rate above 30%",
     "Operational remediation": "Expand produce SKU depth", "Priority score": 0.62},
    {"Geographic catchment": "Delhi NCR — Noida", "Latent supply signal": "Household category underserved vs demand",
     "Operational remediation": "Local assortment review", "Priority score": 0.55},
])


def render_company_dashboard():
    render_page_header(
        "Business Intelligence", "📊 BasketBee Intelligence",
        "Location intelligence for quick-commerce companies. Aggregated, anonymized demand and "
        "fulfilment signals — illustrative sample dashboard, all figures are static demo values.",
    )

    account = st.selectbox("Company account", list(ACCOUNT_METRICS.keys()))
    m = ACCOUNT_METRICS[account]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Basket Fill Rate", f"{m['fill_rate']:.1f}%")
    k2.metric("Session Conversion", f"{m['conversion']:.1f}%")
    k3.metric("Average ETA", f"{m['avg_eta']} min")
    k4.metric("Average Basket", f"₹{m['avg_basket']:,}")
    k5.metric("Estimated Lost Value", f"₹{m['lost_value']:.1f}L / mo")

    st.write("")

    with st.container(border=True):
        section_header("Demand by Category",
                        "Demand index vs. unfulfilled-request rate, aggregated across all pilot zones.")
        st.bar_chart(DEMAND_BY_CATEGORY)

    with st.container(border=True):
        section_header("Priority Business Opportunities",
                        "Geographic catchments ranked by latent supply signal and remediation priority.")
        st.dataframe(
            PRIORITY_OPPORTUNITIES.sort_values("Priority score", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={"Priority score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=1, format="%.2f")},
        )


# ---------------------------------------------------------------------------
# DEMO DATA PAGE
# ---------------------------------------------------------------------------

def render_demo_data(catalog_df, platform_catalog_df):
    render_page_header(
        "Embedded Dataset", "🗄️ Demo Data",
        "The complete offline product and pricing matrix powering this prototype — "
        "100 SKUs across 12 categories, priced on all 5 platforms.",
    )

    search = st.text_input("🔎 Search by product name or category", "")

    with st.container(border=True):
        section_header("Product × Platform Price Matrix",
                        "One row per SKU, one column per platform (₹). Blank cells indicate the "
                        "item is out of stock on that platform.")

        matrix = platform_catalog_df.pivot_table(index=["category", "name"], columns="platform",
                                                   values="price").reset_index()
        matrix = matrix[["category", "name"] + PLATFORMS]
        matrix.columns = ["Category", "Product"] + PLATFORMS

        if search.strip():
            mask = (matrix["Product"].str.contains(search, case=False, na=False) |
                    matrix["Category"].str.contains(search, case=False, na=False))
            matrix = matrix[mask]

        st.caption(f"Showing {len(matrix)} of {len(catalog_df)} products")
        st.dataframe(matrix, use_container_width=True, hide_index=True)

        csv = platform_catalog_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download full dataset (CSV)", csv, "basketbee_demo_dataset.csv", "text/csv")


# ---------------------------------------------------------------------------
# MAIN — top navigation
# ---------------------------------------------------------------------------

NAV_ITEMS = ["🛍️ Customer MVP", "📊 Company Dashboard", "🗄️ Demo Data"]


def render_top_nav():
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


def main():
    catalog_df = build_catalog()
    platform_catalog_df = build_platform_catalog(catalog_df)

    page = render_top_nav()
    st.divider()

    # Each page renders inside its own bordered pane so the three views are
    # visually separated instead of stacking in the same undivided area.
    with st.container(border=True):
        if page == "🛍️ Customer MVP":
            render_customer_mvp(catalog_df, platform_catalog_df)
        elif page == "📊 Company Dashboard":
            render_company_dashboard()
        else:
            render_demo_data(catalog_df, platform_catalog_df)


if __name__ == "__main__":
    main()