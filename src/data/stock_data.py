"""
Stock data loading and processing utilities for the Hyper framework.

Contains functions for downloading stock data, caching, and ticker management.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from etils import epath
from alpha_vantage.timeseries import TimeSeries


# --- Data Caching Configuration ---
DATA_CACHE_DIR = epath.Path(Path.cwd() / 'data_cache')
ALPHA_VANTAGE_CACHE_DIR = epath.Path(Path.cwd() / 'alpha_vantage_cache')


def clean_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean zero prices using forward fill and backward fill.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with zero prices replaced
    """
    # Replace zeros with NaN
    df_clean = df.replace(0, np.nan)
    
    # Forward fill (use previous valid price)
    df_clean = df_clean.ffill()
    
    # Backward fill (use next valid price for leading NaNs)
    df_clean = df_clean.bfill()
    
    # Validate no zeros or NaNs remain
    zero_count = (df_clean == 0).sum().sum()
    nan_count = df_clean.isna().sum().sum()
    
    if zero_count > 0 or nan_count > 0:
        raise ValueError(f"Data cleaning failed: {zero_count} zeros, {nan_count} NaNs remaining")
    
    return df_clean


def _get_cache_filename(tickers, period, interval, clean_zeros=True):
    """
    Generates a unique cache filename based on tickers, period, interval, and cleaning options.
    
    Args:
        tickers: List of ticker symbols
        period: Data period (e.g., "5y", "1y")
        interval: Data interval (e.g., "1d", "1wk")
        clean_zeros: Whether zero prices are cleaned
        
    Returns:
        Path: Cache file path
    """
    # Create a consistent string representation of the tickers list
    ticker_string = "_".join(sorted(tickers))
    # Combine all parameters into a single string
    unique_string = f"{ticker_string}_{period}_{interval}_clean{clean_zeros}"
    # Use SHA256 hash to create a short, unique filename
    hash_object = hashlib.sha256(unique_string.encode())
    return DATA_CACHE_DIR / f"{hash_object.hexdigest()}.pkl"


def get_stock_data(tickers, period="5y", interval="1d", clean_zeros=True):
    """
    Downloads historical stock data for a list of tickers, with caching and zero price cleaning.

    Args:
        tickers (list): A list of stock ticker symbols.
        period (str): The period for which to download data (e.g., "5y", "1y", "3mo").
        interval (str): The interval of data (e.g., "1d", "1wk", "1mo").
        clean_zeros (bool): Whether to clean zero prices using forward/backward fill.

    Returns:
        pd.DataFrame: A MultiIndex DataFrame with stock data, where the first level
                      of the column index is the ticker symbol.
    """
    cache_file = _get_cache_filename(tickers, period, interval, clean_zeros)

    if cache_file.exists():
        print(f"Loading data from cache: {cache_file}")
        try:
            # Ensure the cache file is not empty or corrupted
            cached_data = pd.read_pickle(cache_file)
            if not cached_data.empty:
                return cached_data
            else:
                print(f"  Warning: Cached file {cache_file} is empty. Re-downloading.")
        except Exception as e:
            print(f"  Error loading from cache {cache_file}: {e}. Re-downloading.")

    print(f"Downloading data for {len(tickers)} tickers for period {period}...")
    downloaded_data = {}
    for i in range(0, len(tickers), 50):  # Process in batches of 50 to avoid URL length issues
        batch_tickers = tickers[i:i+50]
        try:
            data = yf.download(batch_tickers, period=period, interval=interval, 
                             group_by='ticker', progress=False, timeout=100, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch_tickers:
                    if (ticker, 'Close') in data.columns:
                        downloaded_data[ticker] = data[ticker].copy()  # Ensure we're working on a copy
            elif not data.empty and len(batch_tickers) == 1:
                # If only one ticker in batch, yfinance might return a flat DataFrame
                # Ensure columns are properly capitalized if not already
                temp_df = data.copy()
                temp_df.columns = [col.capitalize() for col in temp_df.columns]
                downloaded_data[batch_tickers[0]] = temp_df
            else:
                print(f"  Warning: Data for batch {batch_tickers} might be incomplete or empty.")

        except Exception as e:
            print(f"  Error downloading data for batch {batch_tickers}: {e}")
            for ticker in batch_tickers:
                try:
                    single_data = yf.download(ticker, period=period, interval=interval, 
                                            progress=False, timeout=100)
                    if not single_data.empty and 'Close' in single_data.columns:
                        single_data.columns = [col.capitalize() for col in single_data.columns]
                        downloaded_data[ticker] = single_data.copy()
                    else:
                        print(f"  Warning: No valid data for individual ticker {ticker}.")
                except Exception as single_e:
                    print(f"  Error downloading data for individual ticker {ticker}: {single_e}")

    if downloaded_data:
        # Standardize column names (e.g., 'Adj Close' to 'Close', ensure 'Open', 'High', 'Low', 'Volume' are present)
        processed_dfs = {}
        for ticker in tickers:  # Iterate through original `tickers` list to preserve order
            if ticker in downloaded_data:
                df = downloaded_data[ticker]
                # Standardize columns using the existing function
                df_std = _standardize_ohlcv_columns(df)
                if df_std is None:
                    print(f"  Warning: Ticker {ticker} is missing expected OHLCV columns after standardization. Skipping.")
                    continue
                # Convert to float32 for consistency
                processed_dfs[ticker] = df_std.astype(np.float32).copy()
            else:
                print(f"  Info: No downloaded data for ticker {ticker}. Skipping processing.")

        if not processed_dfs:
            print("  No tickers with complete OHLCV data after processing. Returning empty DataFrame.")
            return pd.DataFrame()

        # Combine all processed dataframes into a single MultiIndex DataFrame
        # The keys of processed_dfs become the first level of the MultiIndex
        # The columns of the DataFrames in processed_dfs becomes the second level
        # Explicitly pass `tickers` to `pd.concat` to ensure column order based on original `TICKERS` list
        combined_data = pd.concat([processed_dfs[t] for t in tickers if t in processed_dfs], 
                                 axis=1, join='outer', keys=[t for t in tickers if t in processed_dfs])

        # Ensure the index is named 'Date'
        if combined_data.index.name != 'Date':
            combined_data.index.name = 'Date'

        # Clean zero prices if requested
        if clean_zeros:
            print("🧹 Cleaning zero prices using forward/backward fill...")
            try:
                combined_data = clean_zero_prices(combined_data)
                print("✅ Zero prices cleaned successfully")
            except Exception as e:
                print(f"⚠️  Warning: Zero price cleaning failed: {e}")
                print("   Proceeding with original data (may cause sequence generation issues)")

        # Drop rows with any NaN values to ensure synchronous time series for backtesting
        combined_data = combined_data.ffill().fillna(0).dropna()

        print(f"Successfully downloaded and processed data for {len(processed_dfs)} tickers with {len(combined_data)} rows.")

        # Save to cache before returning
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure cache directory exists
        combined_data.to_pickle(cache_file)
        print(f"Data saved to cache: {cache_file}")

        return combined_data
    else:
        print("  No valid data collected for any ticker. Returning empty DataFrame.")
        return pd.DataFrame()


def _standardize_ohlcv_columns(df):
    """
    Map Alpha Vantage columns like '1. open', '2. high', etc. to ['Open', 'High', 'Low', 'Close', 'Volume'].
    Returns a DataFrame with only those columns, or None if any are missing.
    """
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "open" in col_lower:
            col_map[col] = "Open"
        elif "high" in col_lower:
            col_map[col] = "High"
        elif "low" in col_lower:
            col_map[col] = "Low"
        elif "close" in col_lower and "adj" not in col_lower:
            col_map[col] = "Close"
        elif "volume" in col_lower:
            col_map[col] = "Volume"
    df = df.rename(columns=col_map)
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in ohlcv_cols if c not in df.columns]
    if missing:
        print(f"  Warning: Missing columns after standardization: {missing}")
        return None
    return df[ohlcv_cols]


def download_alpha_vantage_ohlc(tickers, clean_zeros=True):
    """
    Downloads daily OHLC data for each ticker from Alpha Vantage (outputsize='full') and dumps each DataFrame to a file in alpha_vantage_cache.
    The file is named as <ticker>_alpha_vantage_ohlc.pkl.
    Also saves a meta file with just the 'Last Refreshed' date as <ticker>_alpha_vantage_ohlc_meta.json.
    No combining or cleaning is performed; raw DataFrames are saved.
    """
    import time
    import json
    from src.data.news_data import ALPHA_VANTAGE_API_KEY  # Use the same key logic as news_data
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    ALPHA_VANTAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading Alpha Vantage OHLC for {ticker}...")
        try:
            data, meta = ts.get_daily(symbol=ticker, outputsize='full')
            if data is not None and not data.empty:
                data = _standardize_ohlcv_columns(data)
                if data is None:
                    print(f"  Warning: Ticker {ticker} is missing expected OHLCV columns after standardization. Skipping.")
                    continue
                # Optionally clean zeros if requested
                if clean_zeros:
                    try:
                        data = clean_zero_prices(data)
                    except Exception as e:
                        print(f"  Warning: Zero price cleaning failed for {ticker}: {e}")
                        print("   Proceeding with original data.")
                # Save to cache
                out_file = ALPHA_VANTAGE_CACHE_DIR / f"{ticker}_alpha_vantage_ohlc.pkl"
                data.to_pickle(out_file)
                print(f"  ✅ Saved {ticker} OHLC data to {out_file}")
                # Save meta file with just 'Last Refreshed'
                last_refreshed = meta.get('3. Last Refreshed', None)
                meta_file = ALPHA_VANTAGE_CACHE_DIR / f"{ticker}_alpha_vantage_ohlc_meta.json"
                with open(meta_file, 'w') as f:
                    json.dump({'last_refreshed': last_refreshed}, f)
                print(f"  📝 Saved meta for {ticker} to {meta_file}")
            else:
                print(f"  ⚠️  No data returned for {ticker}")
        except Exception as e:
            print(f"  ❌ Error downloading {ticker}: {e}")
        # Alpha Vantage free tier: 5 calls/minute
        time.sleep(12)


def download_alpha_vantage_ohlc_incremental(tickers, clean_zeros=True):
    """
    Downloads 'compact' (latest) OHLC data for each ticker from Alpha Vantage and appends only new rows to the existing full-data file if present.
    Updates the meta file with the new last refreshed date.
    If the file does not exist, prints a message and skips that ticker (does not attempt download).
    """
    import time
    import json
    from src.data.news_data import ALPHA_VANTAGE_API_KEY
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    ALPHA_VANTAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        out_file = ALPHA_VANTAGE_CACHE_DIR / f"{ticker}_alpha_vantage_ohlc.pkl"
        meta_file = ALPHA_VANTAGE_CACHE_DIR / f"{ticker}_alpha_vantage_ohlc_meta.json"
        # If file does not exist, skip and ask user to run full download first
        if not out_file.exists():
            print(f"  ❌ Full data file for {ticker} not found. Please run the full download first. Skipping download.")
            continue
        print(f"Incremental Alpha Vantage OHLC for {ticker} (compact)...")
        try:
            data, meta = ts.get_daily(symbol=ticker, outputsize='compact')
            if data is not None and not data.empty:
                data = _standardize_ohlcv_columns(data)
                if data is None:
                    print(f"  Warning: Ticker {ticker} is missing expected OHLCV columns after standardization. Skipping.")
                    continue
                # Optionally clean zeros if requested
                if clean_zeros:
                    try:
                        data = clean_zero_prices(data)
                    except Exception as e:
                        print(f"  Warning: Zero price cleaning failed for {ticker}: {e}")
                        print("   Proceeding with original data.")
                try:
                    existing = pd.read_pickle(out_file)
                    # Find new rows (by index)
                    new_rows = data[~data.index.isin(existing.index)]
                    if not new_rows.empty:
                        combined = pd.concat([existing, new_rows]).sort_index()
                        combined = combined[~combined.index.duplicated(keep='last')]
                        combined.to_pickle(out_file)
                        print(f"  ✅ Appended {len(new_rows)} new rows to {out_file}")
                    else:
                        print(f"  ℹ️  No new rows to append for {ticker}")
                except Exception as e:
                    print(f"  ⚠️  Error reading or updating {out_file}: {e}. Skipping update.")
                    continue
                # Save/update meta file with new last refreshed
                last_refreshed = meta.get('3. Last Refreshed', None)
                with open(meta_file, 'w') as f:
                    json.dump({'last_refreshed': last_refreshed}, f)
                print(f"  📝 Updated meta for {ticker} to {meta_file}")
            else:
                print(f"  ⚠️  No data returned for {ticker}")
        except Exception as e:
            print(f"  ❌ Error downloading {ticker}: {e}")
        time.sleep(12)


def get_large_cap_tickers():
    """
    Returns a list of S&P 500 stock tickers sourced from slickcharts.com.
    Duplicates are removed to ensure a clean list while preserving the original order.

    Note: Directly fetching a comprehensive, dynamic list of all US stocks
    above a specific market capitalization (e.g., $1 Billion) using only
    `yfinance`'s standard functions or `yfinance.EquityQuery` for general
    screening across an entire universe of stocks is not straightforward.
    `yfinance.EquityQuery` is primarily for constructing queries for specific
    Yahoo Finance screener endpoints, which may not expose the full universe
    with arbitrary market cap filters.
    """
    print("Providing S&P 500 tickers from slickcharts.com, with duplicates removed and order preserved...")
    # List of S&P 500 tickers extracted from from the latest browse of https://www.slickcharts.com/sp500
    sp500_tickers = [
        #'MSFT','NVDA','AAPL','AMZN','META','AVGO','GOOG','GOOGL','TSLA','WMT','JPM','LLY','V','ORCL','NFLX','MA','XOM','COST','PG','JNJ','HD','BAC','ABBV','PLTR','KO','PM','UNH','IBM','CSCO','CVX','CRM','GE','TMUS','WFC','ABT','LIN','INTU','DIS','MCD','MS','NOW','AMD','AXP','T','MRK','ACN','RTX','GS','ISRG','TXN','PEP','UBER','VZ','BKNG','QCOM','CAT','ADBE','SCHW','AMGN','PGR','SPGI','BA','BLK','BSX','TMO','NEE','C','HON','SYK','AMAT','DHR','DE','TJX','PFE','MU','GILD','PANW','UNP','GEV','ETN','CMCSA','ADP','COF','CRWD','LRCX','COP','KLAC','LOW','ANET','VRTX','ADI','CB','APH','MDT','LMT','KKR','MMC','SBUX','BX','ICE','AMT','MO','WELL','PLD','SO','BMY','CME','CEG','TT','WM','DASH','FI','INTC','MCK','NKE','CTAS','DUK','HCA','EQIX','MDLZ','CVS','MCO','UPS','ELV','PH','CI','SHW','ABNB','CDNS','AJG','TDG','FTNT','DELL','RSG','MMM','ORLY','APO','AON','GD','SNPS','ECL','ZTS','CL','NOC','EMR','RCL','WMB','ITW','MAR','PYPL','CMG','HWM','PNC','JCI','EOG','MSI','USB','WDAY','COIN','NEM','ADSK','BK','APD','MNST','KMI','CSX','ROP','AZO','AXON','VST','TRV','CBOE','CARR','DLR','FCX','HLT','COR','NSC','REGN','PAYX','AFL','NXPI','AEP','FDX','PWR','ALL','MET','CHTR','MPC','O','PSA','SPG','TFC','OKE','PSX','GWW','CTVA','NDAQ','SLB','BDX','TEL','AIG','PCAR','AMP','FAST','SRE','GM','CPRT','LHX','D','URI','OXY','FANG','KDP','HES','KR','VLO','TGT','CMI','FICO','TTWO','GLW','EW','KMB','CCI','VRSK','EXC','ROST','FIS','MSCI','IDXX','F','KVUE','AME','PEG','CTSH','CBRE','BKR','CAH','YUM','GRMN','XEL','OTIS','EA','DHI','TRGP','RMD','PRU','MCHP','ED','ROK','SYY','ETR','EBAY','HIG','BRO','EQT','HSY','WAB','VMC','CSGP','VICI','MPWR','ODFL','ACGL','LYV','A','WEC','GEHC','MLM','IR','DXCM','CCL','EFX','EXR','IT','DAL','KHC','XYL','IRM','PCG','ANSS','RJF','LVS','NRG','GIS','AVB','WTW','HUM','LEN','STZ','MTB','LULU','EXE','NUE','VTR','DD','BR','KEYS','K','STX','STT','WRB','CNC','AWK','DTE','ROL','TSCO','IQV','VRSN','EL','DRI','EQR','SMCI','ADM','WBD','AEE','GDDY','FITB','TYL','TPL','DG','PPL','UAL','SBAC','PPG','IP','DOV','VLTO','MTD','ATO','FTV','CHD','HPE','ES','STE','CPAY','CNP','SYF','HPQ','TDY','FE','CDW','SW','CINF','HBAN','ON','DVN','LH','JBL','NTRS','PODD','ULTA','DOW','AMCR','HUBB','EXPE','NTAP','CMS','WDC','DLTR','NVR','INVH','PTC','CTRA','WAT','TROW','PHM','DGX','HAL','MKC','STLD','TSN','WSM','LYB','RF','LII','IFF','LDOS','WY','BIIB','EIX','GPN','GEN','L','NI','ERIE','ESS','ZBH','LUV','CFG','MAA','KEY','TPR','TRMB','PFG','PKG','GPC','HRL','FFIV','CF','SNA','RL','FDS','PNR','MOH','WST','DPZ','EXPD','FSLR','J','DECK','BAX','LNT','BALL','EVRG','CLX','ZBRA','APTV','BBY','HOLX','EG','KIM','TER','JBHT','COO','TXT','PAYC','AVY','OMC','TKO','UDR','INCY','IEX','JKHY','ALGN','MAS','REG','SOLV','ARE','CPT','NDSN','BLDR','FOXA','JNPR','DOC','BEN','ALLE','BG','BXP','MOS','AKAM','CHRW','RVTY','FOX','UHS','HST','POOL','SWKS','PNW','CAG','VTRS','DVA','SJM','TAP','AIZ','MRNA','SWK','KMX','WBA','GL','EPAM','LKQ','HAS','CPB','DAY','WYNN','MGM','NWS','HII','AOS','HSIC','EMN','IPG','MKTX','FRT','NCLH','PARA','NWSA','TECH','LW','AES','APA','MTCH','GNRC','CRL','ALB','IVZ','MHK','CZR','ENPH'
        'MSFT','NVDA','AAPL','AMZN','META','AVGO','GOOGL','TSLA','WMT','JPM','LLY','V','ORCL','NFLX','MA','XOM','COST','PG','JNJ','HD','BAC','ABBV','PLTR','KO','PM','UNH','IBM','CSCO','CVX','CRM','GE','TMUS','WFC','ABT','LIN','INTU','DIS','MCD','MS','NOW','AMD','AXP','T','MRK','ACN','RTX','GS','ISRG','TXN','PEP','UBER','VZ','BKNG','QCOM','CAT','ADBE','SCHW','AMGN','PGR','SPGI','BA','BLK','BSX','TMO','NEE','C','HON','SYK','AMAT','DHR','DE','TJX','PFE','MU','GILD','PANW','UNP','GEV','ETN','CMCSA','ADP','COF','CRWD','LRCX','COP','KLAC','LOW','ANET','VRTX','ADI','CB','APH','MDT','LMT','KKR','MMC','SBUX','BX','ICE','AMT','MO','WELL','PLD','SO','BMY','CME','CEG','TT','WM','DASH','FI','INTC','MCK','NKE','CTAS','DUK','HCA','EQIX','MDLZ','CVS','MCO','UPS','ELV','PH','CI','SHW','ABNB','CDNS','AJG','TDG','FTNT','DELL','RSG','MMM','ORLY','APO','AON','GD','SNPS','ECL','ZTS','CL','NOC','EMR','RCL','WMB','ITW','MAR','PYPL','CMG','HWM','PNC','JCI','EOG','MSI','USB','WDAY','COIN','NEM','ADSK','BK','APD','MNST','KMI','CSX','ROP','AZO','AXON','VST','TRV','CBOE','CARR','DLR','FCX','HLT','COR','NSC','REGN','PAYX','AFL','NXPI','AEP','FDX','PWR','ALL','MET','CHTR','MPC','O','PSA','SPG','TFC','OKE','PSX','GWW','CTVA','NDAQ','SLB','BDX','TEL','AIG','PCAR','AMP','FAST','SRE','GM','CPRT','LHX','D','URI','OXY','FANG','KDP','HES','KR','VLO','TGT','CMI','FICO','TTWO','GLW','EW','KMB','CCI','VRSK','EXC','ROST','FIS','MSCI','IDXX','F','KVUE','AME','PEG','CTSH','CBRE','BKR','CAH','YUM','GRMN','XEL','OTIS','EA','DHI','TRGP','RMD','PRU','MCHP','ED','ROK','SYY','ETR','EBAY','HIG','BRO','EQT','HSY','WAB','VMC','CSGP','VICI','MPWR','ODFL','ACGL','LYV','A','WEC','GEHC','MLM','IR','DXCM','CCL','EFX','EXR','IT','DAL','KHC','XYL','IRM','PCG','ANSS','RJF','LVS','NRG','GIS','AVB','WTW','HUM','LEN','STZ','MTB','LULU','EXE','NUE','VTR','DD','BR','KEYS','K','STX','STT','WRB','CNC','AWK','DTE','ROL','TSCO','IQV','VRSN','EL','DRI','EQR','SMCI','ADM','WBD','AEE','GDDY','FITB','TYL','TPL','DG','PPL','UAL','SBAC','PPG','IP','DOV','VLTO','MTD','ATO','FTV','CHD','HPE','ES','STE','CPAY','CNP','SYF','HPQ','TDY','FE','CDW','SW','CINF','HBAN','ON','DVN','LH','JBL','NTRS','PODD','ULTA','DOW','AMCR','HUBB','EXPE','NTAP','CMS','WDC','DLTR','NVR','INVH','PTC','CTRA','WAT','TROW','PHM','DGX','HAL','MKC','STLD','TSN','WSM','LYB','RF','LII','IFF','LDOS','WY','BIIB','EIX','GPN','GEN','L','NI','ERIE','ESS','ZBH','LUV','CFG','MAA','KEY','TPR','TRMB','PFG','PKG','GPC','HRL','FFIV','CF','SNA','RL','FDS','PNR','MOH','WST','DPZ','EXPD','FSLR','J','DECK','BAX','LNT','BALL','EVRG','CLX','ZBRA','APTV','BBY','HOLX','EG','KIM','TER','JBHT','COO','TXT','PAYC','AVY','OMC','TKO','UDR','INCY','IEX','JKHY','ALGN','MAS','REG','SOLV','ARE','CPT','NDSN','BLDR','FOXA','JNPR','DOC','BEN','ALLE','BG','BXP','MOS','AKAM','CHRW','RVTY','FOX','UHS','HST','POOL','SWKS','PNW','CAG','VTRS','DVA','SJM','TAP','AIZ','MRNA','SWK','KMX','WBA','GL','EPAM','LKQ','HAS','CPB','DAY','WYNN','MGM','NWS','HII','AOS','HSIC','EMN','IPG','MKTX','FRT','NCLH','PARA','NWSA','TECH','LW','AES','APA','MTCH','GNRC','CRL','ALB','IVZ','MHK','CZR','ENPH'
    ]
    # Remove duplicates while preserving order
    # For Python 3.7+ (which is typical for most environments), dict.fromkeys() preserves insertion order
    unique_tickers = list(dict.fromkeys(sp500_tickers))
    return unique_tickers 


def get_alpha_vantage_ticker_date_ranges():
    """
    Scans the alpha_vantage_cache directory for all <ticker>_alpha_vantage_ohlc.pkl files,
    loads each DataFrame, and returns a dictionary mapping ticker to (first_date, last_date).
    Only includes tickers for which data is available and readable.
    
    Returns:
        dict: {ticker: (first_date, last_date)} where dates are pd.Timestamp or None if not available
    """
    import os
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