import os
from src.data.stock_data import get_large_cap_tickers

CACHE_DIR = 'alpha_vantage_cache'

# Get the list of all large cap tickers
all_tickers = get_large_cap_tickers()

# Get the set of tickers with a .pkl file in the cache
def get_cached_tickers():
    cached = set()
    if not os.path.exists(CACHE_DIR):
        print(f"Cache directory {CACHE_DIR} does not exist.")
        return cached
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith('_alpha_vantage_ohlc.pkl'):
            ticker = fname.split('_alpha_vantage_ohlc.pkl')[0]
            cached.add(ticker)
    return cached

cached_tickers = get_cached_tickers()
# Keep missing tickers in the same order as all_tickers
missing = [t for t in all_tickers if t not in cached_tickers]

print(f"Total tickers: {len(all_tickers)}")
print(f"Cached tickers: {len(cached_tickers)}")
print(f"Missing tickers: {len(missing)}")

if missing:
    print("\nMissing tickers:")
    print(",".join(missing))
else:
    print("All tickers are present in the cache.") 