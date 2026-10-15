"""
M5-Style Retail Dataset Generator.
Generates realistic multi-SKU, multi-store hierarchical retail sales data,
calendar metadata (events, holidays, SNAP), and weekly sell prices.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


CATEGORIES = {
    "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
    "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
    "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
}

STORES = {
    "CA": ["CA_1", "CA_2"],
    "TX": ["TX_1", "TX_2"],
}


def generate_sku_metadata(num_skus: int = 20) -> pd.DataFrame:
    """Generate hierarchical SKU metadata matching M5 schema."""
    skus = []
    cat_keys = list(CATEGORIES.keys())
    
    for i in range(1, num_skus + 1):
        cat = cat_keys[(i - 1) % len(cat_keys)]
        depts = CATEGORIES[cat]
        dept = depts[(i - 1) % len(depts)]
        item_id = f"{dept}_{i:03d}"
        skus.append({
            "item_id": item_id,
            "dept_id": dept,
            "cat_id": cat,
            "base_price": round(float(np.random.uniform(1.99, 29.99)), 2),
            "base_demand": float(np.random.uniform(4.0, 35.0)),
        })
    return pd.DataFrame(skus)


def generate_calendar(start_date: str = "2024-01-01", days: int = 730) -> pd.DataFrame:
    """Generate calendar metadata including events, holidays, and SNAP eligibility."""
    dates = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i) for i in range(days)]
    
    records = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        wday = d.weekday()  # 0=Monday, 6=Sunday
        # wm_yr_wk: M5 week numbering
        wm_yr_wk = int(f"{d.year}{d.isocalendar()[1]:02d}")
        
        event_name_1 = None
        event_type_1 = None
        
        # Major retail events
        if d.month == 1 and d.day == 1:
            event_name_1, event_type_1 = "NewYear", "National"
        elif d.month == 2 and 7 <= d.day <= 14 and wday == 6:
            event_name_1, event_type_1 = "SuperBowl", "Sporting"
        elif d.month == 5 and 25 <= d.day <= 31 and wday == 0:
            event_name_1, event_type_1 = "MemorialDay", "National"
        elif d.month == 7 and d.day == 4:
            event_name_1, event_type_1 = "IndependenceDay", "National"
        elif d.month == 9 and 1 <= d.day <= 7 and wday == 0:
            event_name_1, event_type_1 = "LaborDay", "National"
        elif d.month == 11 and 22 <= d.day <= 28 and wday == 3:
            event_name_1, event_type_1 = "Thanksgiving", "National"
        elif d.month == 12 and d.day == 25:
            event_name_1, event_type_1 = "Christmas", "National"

        # SNAP benefits active during first 10 days of the month
        snap_CA = 1 if 1 <= d.day <= 10 else 0
        snap_TX = 1 if 1 <= d.day <= 10 and d.day % 2 == 1 else 0

        records.append({
            "date": date_str,
            "wm_yr_wk": wm_yr_wk,
            "weekday": d.strftime("%A"),
            "wday": wday,
            "month": d.month,
            "year": d.year,
            "event_name_1": event_name_1,
            "event_type_1": event_type_1,
            "snap_CA": snap_CA,
            "snap_TX": snap_TX,
        })
    return pd.DataFrame(records)


def generate_sell_prices(
    calendar_df: pd.DataFrame, sku_df: pd.DataFrame, stores_list: List[str]
) -> pd.DataFrame:
    """Generate weekly sell prices with realistic periodic promotional markdowns."""
    weeks = calendar_df["wm_yr_wk"].unique()
    records = []
    
    np.random.seed(42)
    for store_id in stores_list:
        for _, sku in sku_df.iterrows():
            base_p = sku["base_price"]
            # Store-specific price variance
            store_mult = 1.05 if "CA" in store_id else 0.95
            reg_price = round(base_p * store_mult, 2)
            
            for wk in weeks:
                # 15% probability of promotion/markdown
                is_promo = np.random.rand() < 0.15
                discount = np.random.choice([0.10, 0.20, 0.30]) if is_promo else 0.0
                price = round(reg_price * (1.0 - discount), 2)
                records.append({
                    "store_id": store_id,
                    "item_id": sku["item_id"],
                    "wm_yr_wk": wk,
                    "sell_price": price,
                })
    return pd.DataFrame(records)


def generate_hierarchical_sales(
    sku_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    stores_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Generate daily SKU-store sales with realistic demand signals:
    - Day-of-week seasonality (weekend surge)
    - Annual seasonality (sine wave)
    - Promotional price elasticity
    - Holiday spikes & Christmas closures
    - SNAP benefit effect on foods
    - Occasional supply chain stockouts (anomalies) and viral demand spikes
    """
    np.random.seed(42)
    all_series = []

    # Pre-index prices for fast lookup
    price_map = {
        (row.store_id, row.item_id, row.wm_yr_wk): row.sell_price
        for row in prices_df.itertuples(index=False)
    }

    dates = calendar_df["date"].tolist()
    wm_yr_wks = calendar_df["wm_yr_wk"].tolist()
    wdays = calendar_df["wday"].tolist()
    events = calendar_df["event_name_1"].tolist()
    months = calendar_df["month"].tolist()
    snap_cas = calendar_df["snap_CA"].tolist()
    snap_txs = calendar_df["snap_TX"].tolist()

    n_days = len(dates)

    for state_id, stores in stores_dict.items():
        for store_id in stores:
            for _, sku in sku_df.iterrows():
                item_id = sku["item_id"]
                dept_id = sku["dept_id"]
                cat_id = sku["cat_id"]
                base_demand = sku["base_demand"]
                base_price = sku["base_price"]

                series_id = f"{item_id}_{store_id}"
                
                # Precompute series trajectory
                sales_series = np.zeros(n_days, dtype=float)
                
                for t in range(n_days):
                    dt = dates[t]
                    wk = wm_yr_wks[t]
                    wday = wdays[t]
                    event = events[t]
                    month = months[t]
                    snap = snap_cas[t] if state_id == "CA" else snap_txs[t]

                    # 1. Day of week multiplier (weekend peak Friday-Sunday)
                    dow_mult = 1.35 if wday in [4, 5, 6] else 0.88

                    # 2. Annual seasonality (sinusoidal)
                    annual_mult = 1.0 + 0.15 * np.sin(2 * np.pi * t / 365.25)

                    # 3. Price elasticity (e.g. -1.5)
                    curr_price = price_map.get((store_id, item_id, wk), base_price)
                    price_ratio = curr_price / (base_price + 1e-6)
                    price_mult = (price_ratio) ** (-1.4)

                    # 4. Event spikes & Christmas closures
                    event_mult = 1.0
                    if event == "Christmas":
                        # Store is closed on Christmas
                        sales_series[t] = 0.0
                        continue
                    elif event == "Thanksgiving":
                        event_mult = 1.45
                    elif event == "SuperBowl" and cat_id == "FOODS":
                        event_mult = 1.60
                    elif event in ["MemorialDay", "LaborDay", "IndependenceDay"]:
                        event_mult = 1.25

                    # 5. SNAP effect on food
                    snap_mult = 1.20 if (snap == 1 and cat_id == "FOODS") else 1.0

                    # 6. Combined Poisson lambda
                    lam = base_demand * dow_mult * annual_mult * price_mult * event_mult * snap_mult
                    
                    # Generate daily sales with Poisson + slight dispersion
                    daily_sales = float(np.random.poisson(max(lam, 0.1)))

                    # 7. Inject synthetic realistic anomalies (for detection evaluation)
                    # a) Stockout anomaly: 5 consecutive days of zero sales despite high expected demand
                    if t == 410 and store_id == "CA_1" and item_id.endswith("001"):
                        daily_sales = 0.0
                    elif 410 < t <= 415 and store_id == "CA_1" and item_id.endswith("001"):
                        daily_sales = 0.0
                    # b) Viral demand spike anomaly (e.g. 4.5x surge)
                    elif t == 520 and store_id == "TX_1" and item_id.endswith("002"):
                        daily_sales = round(daily_sales * 4.5)

                    sales_series[t] = daily_sales

                for t in range(n_days):
                    all_series.append({
                        "id": series_id,
                        "item_id": item_id,
                        "dept_id": dept_id,
                        "cat_id": cat_id,
                        "store_id": store_id,
                        "state_id": state_id,
                        "date": dates[t],
                        "sales": sales_series[t],
                    })

    return pd.DataFrame(all_series)


def generate_and_save_all(
    output_dir: str = "data",
    num_skus: int = 20,
    days_history: int = 730,
    start_date: str = "2024-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates and writes raw parquet datasets to the target directory."""
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating SKU metadata for {num_skus} SKUs...")
    sku_df = generate_sku_metadata(num_skus=num_skus)

    print(f"Generating calendar for {days_history} days starting {start_date}...")
    cal_df = generate_calendar(start_date=start_date, days=days_history)

    all_stores = [s for sublist in STORES.values() for s in sublist]
    print(f"Generating sell prices for {len(all_stores)} stores...")
    prices_df = generate_sell_prices(cal_df, sku_df, all_stores)

    print("Generating hierarchical sales series...")
    sales_df = generate_hierarchical_sales(sku_df, cal_df, prices_df, STORES)

    # Save to Parquet
    sales_path = raw_dir / "sales_train.parquet"
    cal_path = raw_dir / "calendar.parquet"
    prices_path = raw_dir / "sell_prices.parquet"

    sales_df.to_parquet(sales_path, index=False)
    cal_df.to_parquet(cal_path, index=False)
    prices_df.to_parquet(prices_path, index=False)

    print(f"Datasets successfully written:")
    print(f" - Sales: {sales_path} ({len(sales_df):,} rows)")
    print(f" - Calendar: {cal_path} ({len(cal_df):,} rows)")
    print(f" - Prices: {prices_path} ({len(prices_df):,} rows)")

    return sales_df, cal_df, prices_df


if __name__ == "__main__":
    generate_and_save_all()
