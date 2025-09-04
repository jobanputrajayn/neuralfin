#!/usr/bin/env python3
"""
Script to validate Alpha Vantage data against Yahoo Finance data for all tickers present in the Alpha Vantage cache.
For each ticker, fetches Yahoo data for the same date range as the Alpha Vantage data (interval is always '1d').
Prints total absolute deviations and standard deviation per column for each ticker.
"""
import sys
from src.data.stock_data import get_stock_data, ALPHA_VANTAGE_CACHE_DIR
import pandas as pd
import numpy as np
from tabulate import tabulate
import os

def get_alpha_vantage_tickers_and_ranges():
    """
    Scans the alpha_vantage_cache directory for all <ticker>_alpha_vantage_ohlc.pkl files,
    loads each DataFrame, and returns a dict mapping ticker to (first_date, last_date).
    """
    result = {}
    cache_dir = ALPHA_VANTAGE_CACHE_DIR
    if not cache_dir.exists():
        print(f"Alpha Vantage cache directory {cache_dir} does not exist.")
        return result
    for file in os.listdir(str(cache_dir)):
        if file.endswith('_alpha_vantage_ohlc.pkl'):
            ticker = file.split('_alpha_vantage_ohlc.pkl')[0]
            file_path = cache_dir / file
            try:
                df = pd.read_pickle(file_path)
                if not df.empty:
                    first_date = df.index.min()
                    last_date = df.index.max()
                    result[ticker] = (first_date, last_date)
                else:
                    print(f"  Info: DataFrame for {ticker} is empty.")
            except Exception as e:
                print(f"  Warning: Could not read data for {ticker} from {file_path}: {e}")
    return result

def validate_alpha_vantage_vs_yahoo(ticker_ranges, clean_zeros=True):
    """
    Validate Alpha Vantage data against Yahoo Finance data for all tickers in the cache.
    For each ticker, fetch Yahoo data for the same date range as the Alpha Vantage data.
    Reports total absolute deviations and standard deviation per column for each ticker.
    Args:
        ticker_ranges (dict): {ticker: (first_date, last_date)}
        clean_zeros (bool): Whether to clean zeros in Yahoo data.
    """
    results = []
    for ticker, (first_date, last_date) in ticker_ranges.items():
        av_file = ALPHA_VANTAGE_CACHE_DIR / f"{ticker}_alpha_vantage_ohlc.pkl"
        if not av_file.exists():
            print(f"[AV] No Alpha Vantage data for {ticker}, skipping.")
            continue
        try:
            av_df = pd.read_pickle(av_file)
            av_df = av_df.sort_index()  # Ensure ascending date order
        except Exception as e:
            print(f"[AV] Error loading Alpha Vantage data for {ticker}: {e}")
            continue
        # Restrict Yahoo data to the same date range as Alpha Vantage
        try:
            # get_stock_data does not support None for period, so use start/end
            yh_df = get_stock_data([ticker], period="max", interval='1d', clean_zeros=clean_zeros)
            if isinstance(yh_df.columns, pd.MultiIndex):
                yh_df = yh_df[ticker]
            # Restrict both to the same date range
            av_df_range = av_df.loc[first_date:last_date]
            yh_df_range = yh_df.loc[first_date:last_date]
        except Exception as e:
            print(f"[YF] Error loading Yahoo Finance data for {ticker}: {e}")
            continue
        # Align by date
        common_dates = av_df_range.index.intersection(yh_df_range.index)
        if len(common_dates) == 0:
            print(f"[WARN] No overlapping dates for {ticker}.")
            continue
        av_aligned = av_df_range.loc[common_dates]
        yh_aligned = yh_df_range.loc[common_dates]
        # Compare columns
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        row = {'Ticker': ticker}
        for col in columns:
            if col not in av_aligned.columns or col not in yh_aligned.columns:
                row[f"{col} nonzero count"] = 'N/A'
                row[f"{col} std dev"] = 'N/A'
                continue
            diff = (av_aligned[col] - yh_aligned[col])
            nonzero_diff = diff[diff != 0]
            nonzero_count = int(nonzero_diff.count())
            row[f"{col} nonzero count"] = nonzero_count
            row[f"{col} std dev"] = float(nonzero_diff.std()) if not nonzero_diff.empty else 0.0
        results.append(row)
    # Print summary table
    if results:
        print(tabulate(results, headers="keys", tablefmt="github"))
    else:
        print("No results to display.")

if __name__ == "__main__":
    ticker_ranges = get_alpha_vantage_tickers_and_ranges()
    if not ticker_ranges:
        print("No Alpha Vantage data found in cache.")
        sys.exit(1)
    print(f"Validating {len(ticker_ranges)} tickers from Alpha Vantage cache...")
    validate_alpha_vantage_vs_yahoo(ticker_ranges) 