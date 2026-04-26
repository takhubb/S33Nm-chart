#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jquantsapi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from jquantsapi.enums import BulkEndpoint
from requests.exceptions import HTTPError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_CHART_DIR = ROOT / "output" / "charts"
DEFAULT_DATA_DIR = ROOT / "output" / "data"
DEFAULT_PRICE_COL = "AdjC"
MARKET_CODES = ["0111", "0112", "0113"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create weekly industry, market, TOPIX, PER and PBR charts from J-Quants API V2.",
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--price-column", default=DEFAULT_PRICE_COL, choices=["AdjC", "C"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chart-dir", type=Path, default=DEFAULT_CHART_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sleep", type=float, default=1.10, help="Seconds between uncached API calls.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for API calls.")
    parser.add_argument("--group-agg", default="median", choices=["median", "mean"], help="How to aggregate stock indexes by industry or market.")
    parser.add_argument("--min-valuation-count", type=int, default=3, help="Minimum stocks required to plot a sector PER or PBR point.")
    parser.add_argument("--refresh", action="store_true", help="Ignore combined CSV caches.")
    return parser.parse_args()


def setup_matplotlib() -> None:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Hiragino Sans",
        "YuGothic",
        "BIZ UDGothic",
        "AppleGothic",
        "Osaka",
        "DejaVu Sans",
    ]
    font_family = next((font for font in preferred_fonts if font in installed_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": [font_family],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 180,
        }
    )


def ymd(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)


def read_csv(path: Path, date_cols: list[str] | None = None) -> pd.DataFrame:
    dtype = {"Code": "string", "S17": "string", "S33": "string", "Mkt": "string"}
    kwargs = {"dtype": dtype, "low_memory": False}
    if date_cols:
        kwargs["parse_dates"] = date_cols
    return pd.read_csv(path, **kwargs)


def cached_csv(
    path: Path,
    fetcher,
    *,
    date_cols: list[str] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    if path.exists() and not refresh:
        return read_csv(path, date_cols=date_cols)

    path.parent.mkdir(parents=True, exist_ok=True)
    df = fetcher()
    df.to_csv(path, index=False)
    return df


def call_with_retry(fetcher, label: str, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return fetcher()
        except Exception as exc:
            if isinstance(exc, HTTPError):
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    raise
            if attempt == attempts:
                raise
            wait = min(180, 30 * attempt) if "429" in str(exc) or "too many" in str(exc) else min(60, 2**attempt)
            print(f"{label}: retry {attempt}/{attempts - 1} after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)


def fetch_topix(
    cli: jquantsapi.ClientV2,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache = cache_dir / f"topix_{ymd(start_date)}_{ymd(end_date)}.csv"

    def fetch() -> pd.DataFrame:
        return cli.get_idx_bars_daily_topix(from_yyyymmdd=ymd(start_date), to_yyyymmdd=ymd(end_date))

    df = cached_csv(cache, fetch, date_cols=["Date"], refresh=refresh)
    df["Date"] = pd.to_datetime(df["Date"])
    df["C"] = pd.to_numeric(df["C"], errors="coerce")
    return df.dropna(subset=["Date", "C"]).sort_values("Date")


def weekly_last_close(df: pd.DataFrame, value_col: str = "C") -> pd.DataFrame:
    work = df[["Date", value_col]].dropna().copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date")
    work["WeekEnd"] = work["Date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    weekly = (
        work.groupby("WeekEnd", as_index=False)
        .agg(Date=("Date", "last"), Close=(value_col, "last"))
        .dropna(subset=["Close"])
        .sort_values("Date")
    )
    return weekly[["Date", "Close"]]


def fetch_master(
    cli: jquantsapi.ClientV2,
    as_of: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache = cache_dir / f"eq_master_{ymd(as_of)}.csv"

    def fetch() -> pd.DataFrame:
        return cli.get_eq_master(date=ymd(as_of))

    df = cached_csv(cache, fetch, date_cols=["Date"], refresh=refresh)
    df["Code"] = normalize_code(df["Code"])
    for col in ["S17", "S33", "Mkt"]:
        df[col] = df[col].astype("string")
    keep = ["Code", "CoName", "S17", "S17Nm", "S33", "S33Nm", "Mkt", "MktNm"]
    return df[keep].drop_duplicates("Code")


def fetch_weekly_equity_bars(
    cli: jquantsapi.ClientV2,
    weekly_dates: list[pd.Timestamp],
    cache_dir: Path,
    price_col: str,
    sleep_seconds: float,
    workers: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    bars_dir = cache_dir / "eq_bars_daily"
    bars_dir.mkdir(parents=True, exist_ok=True)
    needed_cols = ["Date", "Code", "C", "AdjC"]

    def load_one(idx: int, dt: pd.Timestamp) -> pd.DataFrame:
        key = ymd(dt)
        cache = bars_dir / f"eq_bars_daily_{key}.csv.gz"
        if cache.exists():
            df = read_csv(cache, date_cols=["Date"])
        else:
            print(f"fetch weekly stock bars {idx}/{len(weekly_dates)}: {key}", flush=True)

            def fetch() -> pd.DataFrame:
                return cli.get_eq_bars_daily(date_yyyymmdd=key)

            df = call_with_retry(fetch, f"eq_bars_daily {key}")
            df.to_csv(cache, index=False)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        missing = [col for col in [price_col, "Date", "Code"] if col not in df.columns]
        if missing:
            raise ValueError(f"{cache} is missing columns: {missing}")
        df = df[[col for col in needed_cols if col in df.columns]].copy()
        df["Code"] = normalize_code(df["Code"])
        df["Date"] = pd.to_datetime(df["Date"])
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        return df.dropna(subset=[price_col])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(load_one, idx, dt) for idx, dt in enumerate(weekly_dates, start=1)]
        for done, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if done % 25 == 0 or done == len(futures):
                print(f"weekly stock bars loaded: {done}/{len(futures)}", flush=True)

    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Code"]).reset_index(drop=True)


def build_group_index(
    bars: pd.DataFrame,
    master: pd.DataFrame,
    *,
    group_col: str,
    name_col: str,
    price_col: str,
    group_agg: str,
    allowed_codes: list[str] | None = None,
) -> pd.DataFrame:
    work = bars[["Date", "Code", price_col]].merge(master, on="Code", how="inner")
    work = work.dropna(subset=[price_col, group_col, name_col])
    if allowed_codes is not None:
        work = work[work[group_col].isin(allowed_codes)]
    elif group_col == "S17":
        work = work[(work["S17"] != "99") & (work["S17Nm"] != "その他")]
    elif group_col == "S33":
        work = work[(work["S33"] != "9999") & (work["S33Nm"] != "その他")]

    work = work[work[price_col] > 0].sort_values(["Code", "Date"])
    first_price = work.groupby("Code", observed=True)[price_col].transform("first")
    work["StockIndex"] = work[price_col] / first_price * 100.0
    grouped = (
        work.groupby(["Date", group_col, name_col], observed=True)
        .agg(Index=("StockIndex", group_agg), Constituents=("Code", "nunique"))
        .reset_index()
        .sort_values(["Date", group_col])
    )
    grouped["Index"] = grouped.groupby(group_col, observed=True)["Index"].transform(
        lambda s: s / s.dropna().iloc[0] * 100.0 if s.notna().any() else s
    )
    return grouped


def add_topix_and_normalize(
    grouped: pd.DataFrame,
    topix_weekly: pd.DataFrame,
    *,
    name_col: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    pivot = grouped.pivot_table(index="Date", columns=name_col, values="Index", aggfunc="last")
    pivot = pivot.dropna(axis=1, how="all").sort_index()
    topix = topix_weekly.set_index("Date")["Close"].rename("TOPIX")
    combined = pivot.join(topix, how="inner")
    first_dates = [combined[col].first_valid_index() for col in combined.columns]
    first_dates = [dt for dt in first_dates if dt is not None]
    common_start = max(first_dates)
    combined = combined.loc[combined.index >= common_start].copy()

    for col in combined.columns:
        first = combined[col].dropna().iloc[0]
        combined[col] = combined[col] / first * 100.0
    return combined, pd.Timestamp(common_start)


def plot_combined_index(
    data: pd.DataFrame,
    output: Path,
    *,
    title: str,
    legend_columns: int = 1,
    wide: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_width = 18 if wide else 13
    fig_height = 10 if wide else 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    normal_cols = [col for col in data.columns if col != "TOPIX"]
    cmap = plt.get_cmap("tab20")
    for idx, col in enumerate(normal_cols):
        ax.plot(data.index, data[col], linewidth=1.05, alpha=0.88, color=cmap(idx % 20), label=col)

    if "TOPIX" in data.columns:
        ax.plot(data.index, data["TOPIX"], color="black", linewidth=2.6, label="TOPIX", zorder=5)

    ax.axhline(100, color="black", linewidth=0.8, alpha=0.25)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_ylabel("相対指数 (開始日=100)")
    ax.grid(True, alpha=0.28)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=legend_columns, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.78 if wide else 0.82, 1))
    fig.savefig(output)
    plt.close(fig)


def numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        if col in df.columns:
            values = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
            result = result.fillna(values)
    return result


def fetch_fin_summary(
    cli: jquantsapi.ClientV2,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    workers: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    cache = cache_dir / f"fin_summary_{ymd(start_date)}_{ymd(end_date)}.csv.gz"
    date_cols = ["DiscDate", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn", "NxtFYSt", "NxtFYEn"]

    if cache.exists() and not refresh:
        df = read_csv(cache, date_cols=date_cols)
    else:
        try:
            df = fetch_fin_summary_bulk(cli, start_date, end_date, cache_dir, refresh, date_cols, sleep_seconds)
        except Exception as exc:
            if "429" in str(exc) or "too many" in str(exc):
                raise RuntimeError("J-Quants rate limit reached while fetching bulk financial summaries. Re-run later; cached files will be reused.") from exc
            print(f"bulk financial summaries unavailable, falling back to daily API: {exc}", flush=True)
            df = fetch_fin_summary_daily(cli, start_date, end_date, cache_dir, refresh, workers, date_cols)
        df.to_csv(cache, index=False)

    if df.empty:
        return df
    df["Code"] = normalize_code(df["Code"])
    df["DiscDate"] = pd.to_datetime(df["DiscDate"])
    return df.sort_values(["Code", "DiscDate", "DiscTime"])


def bulk_key_period(key: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    match = re.search(r"fins_summary_(\d{6}|\d{8})\.csv\.gz$", key)
    if not match:
        return None
    token = match.group(1)
    if len(token) == 6:
        start = pd.Timestamp(f"{token}01")
        end = start + pd.offsets.MonthEnd(1)
    else:
        start = pd.Timestamp(token)
        end = start
    return start.normalize(), end.normalize()


def fetch_fin_summary_bulk(
    cli: jquantsapi.ClientV2,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    date_cols: list[str],
    sleep_seconds: float,
) -> pd.DataFrame:
    listing = call_with_retry(lambda: cli.get_bulk_list(BulkEndpoint.FIN_SUMMARY), "bulk_list fin_summary")
    keys: list[str] = []
    for key in listing["Key"].astype(str):
        period = bulk_key_period(key)
        if period is None:
            continue
        period_start, period_end = period
        if period_end >= start_date and period_start <= end_date:
            keys.append(key)

    if not keys:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    bulk_cache = cache_dir / "bulk"
    for idx, key in enumerate(keys, start=1):
        local_path = bulk_cache / key
        downloaded = False
        if local_path.exists() and not refresh:
            df_one = read_csv(local_path, date_cols=date_cols)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            url = call_with_retry(lambda key=key: cli.get_bulk(key), f"bulk_get {key}")
            df_one = pd.read_csv(url, dtype={"Code": "string"}, compression="gzip")
            df_one.to_csv(local_path, index=False)
            downloaded = True
        frames.append(df_one)
        if downloaded and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if idx % 10 == 0 or idx == len(keys):
            print(f"bulk financial summaries loaded: {idx}/{len(keys)}", flush=True)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    return df[(df["DiscDate"] >= start_date) & (df["DiscDate"] <= end_date)].copy()


def fetch_fin_summary_daily(
    cli: jquantsapi.ClientV2,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    workers: int,
    date_cols: list[str],
) -> pd.DataFrame:
    daily_cache = cache_dir / "fin_summary_daily"
    daily_cache.mkdir(parents=True, exist_ok=True)
    dates = [pd.Timestamp(d) for d in pd.bdate_range(start_date, end_date)]
    frames: list[pd.DataFrame] = []

    def load_one(idx: int, dt: pd.Timestamp) -> pd.DataFrame:
        key = ymd(dt)
        year_dir = daily_cache / key[:4]
        data_path = year_dir / f"fin_summary_{key}.csv.gz"
        empty_path = year_dir / f"fin_summary_{key}.empty"
        if data_path.exists() and not refresh:
            return read_csv(data_path, date_cols=date_cols)
        if empty_path.exists() and not refresh:
            return pd.DataFrame()

        def fetch() -> pd.DataFrame:
            return cli.get_fin_summary(date_yyyymmdd=key)

        df_one = call_with_retry(fetch, f"fin_summary {key}")
        year_dir.mkdir(parents=True, exist_ok=True)
        if df_one.empty:
            empty_path.touch()
            return df_one
        df_one.to_csv(data_path, index=False)
        return df_one

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(load_one, idx, dt) for idx, dt in enumerate(dates, start=1)]
        for done, future in enumerate(as_completed(futures), start=1):
            day_df = future.result()
            if not day_df.empty:
                frames.append(day_df)
            if done % 100 == 0 or done == len(futures):
                print(f"financial summaries loaded: {done}/{len(futures)}", flush=True)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_fundamentals(fin: pd.DataFrame) -> pd.DataFrame:
    if fin.empty:
        return pd.DataFrame(columns=["Code", "DiscDate", "EPSForPER", "BPSForPBR"])

    work = fin.copy()
    work["EPSForPER"] = numeric_series(work, ["FEPS", "NxFEPS", "EPS", "NCEPS"])
    work["BPSForPBR"] = numeric_series(work, ["BPS", "NCBPS"])
    work = work.sort_values(["Code", "DiscDate", "DiscTime"])
    work[["EPSForPER", "BPSForPBR"]] = work.groupby("Code", observed=True)[
        ["EPSForPER", "BPSForPBR"]
    ].ffill()
    work = work.drop_duplicates(["Code", "DiscDate"], keep="last")
    return work[["Code", "DiscDate", "EPSForPER", "BPSForPBR"]].dropna(
        subset=["EPSForPER", "BPSForPBR"], how="all"
    )


def attach_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fund_groups = {code: frame.sort_values("DiscDate") for code, frame in fundamentals.groupby("Code", observed=True)}
    for code, price_frame in prices.groupby("Code", observed=True):
        fund_frame = fund_groups.get(code)
        if fund_frame is None or fund_frame.empty:
            continue
        merged = pd.merge_asof(
            price_frame.sort_values("Date"),
            fund_frame.drop(columns=["Code"]).sort_values("DiscDate"),
            left_on="Date",
            right_on="DiscDate",
            direction="backward",
        )
        merged["Code"] = code
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_sector_per_pbr(
    bars: pd.DataFrame,
    master: pd.DataFrame,
    fin: pd.DataFrame,
    *,
    price_col: str,
    min_count: int,
) -> pd.DataFrame:
    prices = bars[["Date", "Code", price_col]].merge(master[["Code", "S33", "S33Nm"]], on="Code", how="inner")
    prices = prices[(prices["S33"] != "9999") & (prices["S33Nm"] != "その他")]
    prices = prices.dropna(subset=[price_col, "S33", "S33Nm"]).sort_values(["Code", "Date"])
    fundamentals = prepare_fundamentals(fin)
    merged = attach_fundamentals(prices, fundamentals)
    if merged.empty:
        return pd.DataFrame(columns=["Date", "S33", "S33Nm", "PER", "PBR", "PERCount", "PBRCount"])

    merged["PER"] = np.where(merged["EPSForPER"] > 0, merged[price_col] / merged["EPSForPER"], np.nan)
    merged["PBR"] = np.where(merged["BPSForPBR"] > 0, merged[price_col] / merged["BPSForPBR"], np.nan)
    merged.loc[(merged["PER"] <= 0) | (merged["PER"] > 500), "PER"] = np.nan
    merged.loc[(merged["PBR"] <= 0) | (merged["PBR"] > 100), "PBR"] = np.nan
    result = (
        merged.groupby(["Date", "S33", "S33Nm"], observed=True)
        .agg(
            PER=("PER", "median"),
            PBR=("PBR", "median"),
            PERCount=("PER", "count"),
            PBRCount=("PBR", "count"),
        )
        .reset_index()
        .sort_values(["S33", "Date"])
    )
    result.loc[result["PERCount"] < min_count, "PER"] = np.nan
    result.loc[result["PBRCount"] < min_count, "PBR"] = np.nan
    return result


def slugify(value: str) -> str:
    value = re.sub(r"[^\w一-龥ぁ-んァ-ンー]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sector"


def plot_sector_per_pbr(sector_data: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (s33, name), frame in sector_data.groupby(["S33", "S33Nm"], observed=True):
        frame = frame.sort_values("Date")
        if frame[["PER", "PBR"]].dropna(how="all").empty:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6.5))
        ax2 = ax1.twinx()
        ax1.plot(frame["Date"], frame["PER"], color="#1f77b4", linewidth=1.7, label="PER")
        ax2.plot(frame["Date"], frame["PBR"], color="#d95f02", linewidth=1.7, label="PBR")
        ax1.set_title(f"{name} PER / PBR 推移", fontsize=14, pad=12)
        ax1.set_ylabel("PER (倍)", color="#1f77b4")
        ax2.set_ylabel("PBR (倍)", color="#d95f02")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d95f02")
        ax1.grid(True, alpha=0.28)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=False)
        fig.tight_layout()

        output = output_dir / f"sector_per_pbr_{s33}_{slugify(str(name))}.png"
        fig.savefig(output)
        plt.close(fig)
        paths.append(output)
    return paths


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    load_dotenv(ROOT / ".env")
    if not os.getenv("JQUANTS_API_KEY"):
        raise RuntimeError("JQUANTS_API_KEY is not set. Put it in .env or export it before running.")

    end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today(tz="Asia/Tokyo").tz_localize(None)
    end_date = end_date.normalize()
    start_date = end_date - relativedelta(years=args.years)
    fund_start_date = start_date

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.chart_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    cli = jquantsapi.ClientV2()
    cli.MAX_WORKERS = max(1, args.workers)

    print(f"period: {start_date.date()} to {end_date.date()}", flush=True)
    print(f"api workers: {cli.MAX_WORKERS}", flush=True)
    topix_daily = fetch_topix(cli, start_date, end_date, args.cache_dir, args.refresh)
    topix_weekly = weekly_last_close(topix_daily, "C")
    weekly_dates = [pd.Timestamp(d) for d in topix_weekly["Date"].tolist()]
    if not weekly_dates:
        raise RuntimeError("No TOPIX weekly dates found.")

    master_as_of = weekly_dates[-1]
    print(f"classification master as of: {master_as_of.date()}", flush=True)
    master = fetch_master(cli, master_as_of, args.cache_dir, args.refresh)
    weekly_bars = fetch_weekly_equity_bars(
        cli,
        weekly_dates,
        args.cache_dir,
        args.price_column,
        args.sleep,
        args.workers,
    )

    industry = build_group_index(
        weekly_bars,
        master,
        group_col="S33",
        name_col="S33Nm",
        price_col=args.price_column,
        group_agg=args.group_agg,
    )
    topix17 = build_group_index(
        weekly_bars,
        master,
        group_col="S17",
        name_col="S17Nm",
        price_col=args.price_column,
        group_agg=args.group_agg,
    )
    market = build_group_index(
        weekly_bars,
        master,
        group_col="Mkt",
        name_col="MktNm",
        price_col=args.price_column,
        group_agg=args.group_agg,
        allowed_codes=MARKET_CODES,
    )

    industry_chart, industry_start = add_topix_and_normalize(industry, topix_weekly, name_col="S33Nm")
    topix17_chart, topix17_start = add_topix_and_normalize(topix17, topix_weekly, name_col="S17Nm")
    market_chart, market_start = add_topix_and_normalize(market, topix_weekly, name_col="MktNm")

    industry_chart.to_csv(args.data_dir / "industry_relative_weekly.csv", index_label="Date")
    topix17_chart.to_csv(args.data_dir / "topix17_relative_weekly.csv", index_label="Date")
    market_chart.to_csv(args.data_dir / "market_relative_weekly.csv", index_label="Date")
    industry.to_csv(args.data_dir / "industry_group_index_raw.csv", index=False)
    topix17.to_csv(args.data_dir / "topix17_group_index_raw.csv", index=False)
    market.to_csv(args.data_dir / "market_group_index_raw.csv", index=False)

    plot_combined_index(
        industry_chart,
        args.chart_dir / "industry_relative_weekly.png",
        title=f"業種別 週次相対チャート + TOPIX ({industry_start.date()}=100)",
        legend_columns=1,
        wide=True,
    )
    plot_combined_index(
        topix17_chart,
        args.chart_dir / "topix17_relative_weekly.png",
        title=f"TOPIX-17業種別 週次相対チャート + TOPIX ({topix17_start.date()}=100)",
        legend_columns=1,
        wide=True,
    )
    plot_combined_index(
        market_chart,
        args.chart_dir / "market_relative_weekly.png",
        title=f"市場別 週次相対チャート + TOPIX ({market_start.date()}=100)",
        legend_columns=1,
        wide=False,
    )

    fin = fetch_fin_summary(cli, fund_start_date, end_date, args.cache_dir, args.refresh, args.workers, args.sleep)
    sector_per_pbr = build_sector_per_pbr(
        weekly_bars,
        master,
        fin,
        price_col=args.price_column,
        min_count=args.min_valuation_count,
    )
    sector_per_pbr.to_csv(args.data_dir / "sector_per_pbr_weekly.csv", index=False)
    per_pbr_paths = plot_sector_per_pbr(sector_per_pbr, args.chart_dir / "sector_per_pbr")

    print("created:", flush=True)
    print(f"  {args.chart_dir / 'industry_relative_weekly.png'}", flush=True)
    print(f"  {args.chart_dir / 'topix17_relative_weekly.png'}", flush=True)
    print(f"  {args.chart_dir / 'market_relative_weekly.png'}", flush=True)
    print(f"  {len(per_pbr_paths)} PER/PBR charts under {args.chart_dir / 'sector_per_pbr'}", flush=True)
    print(f"  CSV files under {args.data_dir}", flush=True)


if __name__ == "__main__":
    main()
