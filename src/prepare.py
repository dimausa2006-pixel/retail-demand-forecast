"""Turn the raw Online Retail II workbook into one clean table of weekly sales.

Usage:
    python src/prepare.py data/raw/online_retail_II.xlsx

Writes data/weekly_sales.csv and reports/cleaning.md.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_raw(path):
    """Read both sheets and return one table with a 'sheet' column."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = []
    for name, frame in sheets.items():
        frame = frame.copy()
        frame["sheet"] = name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def clean(raw):
    """Apply the cleaning rules one at a time and record how many rows each removes.

    Returns (clean_table, list of (rule, rows_removed, rows_left)).
    """
    log = []
    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    def step(rule, keep_mask):
        nonlocal df
        before = len(df)
        df = df[keep_mask]
        log.append((rule, before - len(df), len(df)))

    log.append(("Rows in the raw workbook", 0, len(df)))

    # The two sheets overlap for the first days of December 2010.
    first_sheet = df["sheet"].iloc[0]
    invoices_in_first = set(df.loc[df["sheet"] == first_sheet, "Invoice"])
    step("Invoices repeated in the second sheet",
         (df["sheet"] == first_sheet) | ~df["Invoice"].isin(invoices_in_first))

    step("Cancelled invoices (number starts with C)", ~df["Invoice"].str.startswith("C"))
    step("Quantity zero or negative", df["Quantity"] > 0)
    step("Price zero or negative", df["Price"] > 0)
    # Real products have a five-digit code, sometimes followed by letters.
    # Postage, bank charges, manual adjustments and samples do not.
    step("Lines that are not products (postage, fees, adjustments)",
         df["StockCode"].str.match(r"^\d{5}"))

    df = df.assign(revenue=df["Quantity"] * df["Price"])
    return df, log


def to_weekly(df):
    """Sum sales by week. A week starts on Monday."""
    df = df.assign(week_start=df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time)
    weekly = df.groupby("week_start").agg(
        units=("Quantity", "sum"),
        revenue=("revenue", "sum"),
        orders=("Invoice", "nunique"),
        products=("StockCode", "nunique"),
    )
    # Fill weeks with no sales at all (the shop closes over Christmas).
    full_index = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-MON")
    weekly = weekly.reindex(full_index, fill_value=0)
    weekly.index.name = "week_start"

    # Drop the first and last week if the data does not cover all of them.
    first_day, last_day = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    if first_day.normalize() > weekly.index[0]:
        weekly = weekly.iloc[1:]
    if last_day.normalize() < weekly.index[-1] + pd.Timedelta(days=5):
        weekly = weekly.iloc[:-1]
    weekly["revenue"] = weekly["revenue"].round(2)
    return weekly.reset_index()


def main():
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "online_retail_II.xlsx"
    raw = load_raw(raw_path)
    df, log = clean(raw)
    weekly = to_weekly(df)

    out = ROOT / "data" / "weekly_sales.csv"
    weekly.to_csv(out, index=False)

    lines = ["# Cleaning log", "",
             f"Source rows cover {raw['InvoiceDate'].min():%d %b %Y} to {raw['InvoiceDate'].max():%d %b %Y}.", "",
             "| Rule | Rows removed | Rows left |", "|:--|--:|--:|"]
    for rule, removed, left in log:
        lines.append(f"| {rule} | {removed:,} | {left:,} |")
    lines += ["",
              f"Weekly table: {len(weekly)} full weeks, {weekly['week_start'].min():%d %b %Y} to "
              f"{weekly['week_start'].max():%d %b %Y}.",
              f"Weeks with no sales: {(weekly['units'] == 0).sum()}.",
              f"Total revenue kept: £{weekly['revenue'].sum():,.0f}."]
    (ROOT / "reports" / "cleaning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
