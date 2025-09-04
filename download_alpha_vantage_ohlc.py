import sys
from src.data.stock_data import download_alpha_vantage_ohlc, download_alpha_vantage_ohlc_incremental, get_large_cap_tickers

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Alpha Vantage OHLC data for large cap tickers.")
    parser.add_argument('--tickers', type=str, default=None, help='Comma-separated list of tickers to use (overrides --ticker-count)')
    parser.add_argument('--ticker-count', type=int, default=1, help='Number of tickers to use from get_large_cap_tickers (default: 1, -1 for all)')
    parser.add_argument('--full', action='store_true', help='Download full history (default: incremental)')
    parser.add_argument('--no-clean', action='store_true', help='Do not clean zero prices')
    args = parser.parse_args()
    
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    else:
        all_tickers = get_large_cap_tickers()
        if args.ticker_count == -1:
            tickers = all_tickers
        else:
            tickers = all_tickers[:args.ticker_count]

    if not tickers:
        print("No tickers specified or found from get_large_cap_tickers(). Exiting.")
        sys.exit(1)

    print(f"[INFO] Using tickers: {tickers}")

    if args.full:
        print(f"[INFO] Running full Alpha Vantage download for: {tickers}")
        download_alpha_vantage_ohlc(tickers, clean_zeros=not args.no_clean)
    else:
        print(f"[INFO] Running incremental Alpha Vantage download for: {tickers}")
        download_alpha_vantage_ohlc_incremental(tickers, clean_zeros=not args.no_clean) 