import sys
from src.data.polygon_news_data import (
    download_all_polygon_news_after_date,
    download_incremental_polygon_news,
    get_polygon_news_for_tickers
)
from src.data.stock_data import get_large_cap_tickers

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Polygon news data for large cap tickers.")
    parser.add_argument('--tickers', type=str, default=None, help='Comma-separated list of tickers to use (overrides --ticker-count)')
    parser.add_argument('--ticker-count', type=int, default=1, help='Number of tickers to use from get_large_cap_tickers (default: 1, -1 for all)')
    parser.add_argument('--full', action='store_true', help='Download full history (default: incremental)')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date for news download (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for news download (YYYY-MM-DD format, defaults to now)')
    parser.add_argument('--days-back', type=int, default=30, help='Number of days to look back for incremental download (default: 30)')
    parser.add_argument('--limit-per-ticker', type=int, default=1000, help='Maximum articles per ticker (default: 1000)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if cache exists')
    parser.add_argument('--market-news', action='store_true', help='Download general market news instead of ticker-specific news')
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

    if args.market_news:
        print(f"[INFO] Running Polygon market news download for the last {args.days_back} days")
        from src.data.polygon_news_data import get_polygon_market_news
        market_news = get_polygon_market_news(days_back=args.days_back, limit=args.limit_per_ticker)
        if not market_news.empty:
            print(f"[SUCCESS] Downloaded {len(market_news)} market news articles")
        else:
            print("[WARNING] No market news data downloaded")
    else:
        if args.full:
            print(f"[INFO] Running full Polygon news download for: {tickers}")
            print(f"[INFO] Date range: {args.start_date} to {args.end_date or 'now'}")
            results = download_all_polygon_news_after_date(
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                force_download=args.force
            )
            
            # Print summary
            total_articles = sum(result.get('articles', 0) for result in results.values())
            successful_downloads = sum(1 for result in results.values() if result.get('status') == 'downloaded')
            print(f"[SUCCESS] Downloaded {total_articles} total articles for {successful_downloads}/{len(tickers)} tickers")
            
        else:
            print(f"[INFO] Running incremental Polygon news download for: {tickers}")
            print(f"[INFO] Looking back {args.days_back} days")
            results = download_incremental_polygon_news(
                tickers=tickers,
                days_back=args.days_back,
                force_download=args.force
            )
            
            # Print summary
            total_new_articles = sum(result.get('articles', 0) for result in results.values())
            total_articles = sum(result.get('total', 0) for result in results.values() if 'total' in result)
            successful_updates = sum(1 for result in results.values() if result.get('status') in ['incremental', 'up_to_date'])
            print(f"[SUCCESS] Downloaded {total_new_articles} new articles, total {total_articles} articles for {successful_updates}/{len(tickers)} tickers") 