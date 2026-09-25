import os
import urllib.request
import pandas as pd

# Direct public URLs for the actual Kaggle M5 Forecasting Dataset
REAL_M5_URLS = {
    "calendar.csv": "https://huggingface.co/datasets/kashif/M5/resolve/main/calendar.csv",
    "sales_train_validation.csv": "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_validation.csv",
    "sell_prices.csv": "https://huggingface.co/datasets/kashif/M5/resolve/main/sell_prices.csv"
}

def download_file_with_progress(url, destination_path):
    """Downloads a file from a URL to local disk with progress tracking."""
    file_name = os.path.basename(destination_path)
    print(f"[DOWNLOAD] Downloading real dataset file: '{file_name}'...")
    
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, int(downloaded * 100 / total_size))
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"   Downloading {file_name}: {pct}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end="\r")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(destination_path, 'wb') as out_file:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 1024 * 64
        downloaded = 0
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            out_file.write(buffer)
            downloaded += len(buffer)
            if total_size > 0:
                pct = min(100, int(downloaded * 100 / total_size))
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"   Downloading {file_name}: {pct}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end="\r")
                
    print(f"\n[SUCCESS] Downloaded '{file_name}' to {destination_path}")

def download_real_m5_dataset(raw_data_dir="data/raw"):
    """
    Downloads the actual Kaggle M5 Forecasting dataset files to data/raw/
    if they are not already present on disk.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    paths = {}
    
    for filename, url in REAL_M5_URLS.items():
        dest_path = os.path.join(raw_data_dir, filename)
        paths[filename] = dest_path
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            download_file_with_progress(url, dest_path)
        else:
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"[CACHE] Found existing real dataset file '{filename}' ({size_mb:.1f} MB) at {dest_path}")
            
    return paths

def run_data_ingestion(raw_data_dir="data/raw", use_real_data=True):
    """
    Executes Phase 1 Data Ingestion using the actual Kaggle M5 dataset.
    """
    print("[DATA PIPELINE] Starting Phase 1 Data Ingestion Pipeline...")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    if use_real_data:
        paths = download_real_m5_dataset(raw_data_dir)
        sales_path = paths["sales_train_validation.csv"]
        calendar_path = paths["calendar.csv"]
        prices_path = paths["sell_prices.csv"]
    else:
        # Fallback path if synthetic is requested
        sales_path = os.path.join(raw_data_dir, "sales_train_validation.csv")
        calendar_path = os.path.join(raw_data_dir, "calendar.csv")
        prices_path = os.path.join(raw_data_dir, "sell_prices.csv")
        
    print("[DATA PIPELINE] Loading raw datasets into memory...")
    df_sales = pd.read_csv(sales_path)
    df_calendar = pd.read_csv(calendar_path)
    df_prices = pd.read_csv(prices_path)
    
    total_series = len(df_sales)
    date_cols = [c for c in df_sales.columns if c.startswith('d_')]
    total_days = len(date_cols)
    total_sales_volume = df_sales[date_cols].sum().sum()
    zero_sales_pct = (df_sales[date_cols] == 0).sum().sum() / (total_series * total_days) * 100
    
    metrics = {
        'total_series': total_series,
        'total_days': total_days,
        'total_sales_volume': int(total_sales_volume),
        'zero_sales_pct': round(float(zero_sales_pct), 2),
        'sales_shape': list(df_sales.shape),
        'calendar_shape': list(df_calendar.shape),
        'prices_shape': list(df_prices.shape),
        'states': list(df_sales['state_id'].unique()),
        'stores': list(df_sales['store_id'].unique())
    }
    
    print("\n[SUCCESS] Real Data Ingestion complete. Summary Metrics:")
    print(f"   • Dataset Source: Real Kaggle Walmart M5 Dataset")
    print(f"   • Total Time Series (Store x Item): {metrics['total_series']:,}")
    print(f"   • Total Daily Observations: {metrics['total_days']} days (d_1 to d_{metrics['total_days']})")
    print(f"   • Total Units Sold: {metrics['total_sales_volume']:,} units")
    print(f"   • Zero Sales Ratio: {metrics['zero_sales_pct']}%")
    print(f"   • States Included: {metrics['states']}")
    print(f"   • Stores Included ({len(metrics['stores'])} stores): {metrics['stores']}")
    
    return df_sales, df_calendar, df_prices, metrics

if __name__ == "__main__":
    run_data_ingestion(use_real_data=True)
