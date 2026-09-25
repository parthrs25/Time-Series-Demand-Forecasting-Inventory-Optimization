import os
import gc
import numpy as np
import pandas as pd
from src.preprocessing import reduce_mem_usage

def create_lag_features(df, lag_days=[28, 29, 35, 42, 56, 364]):
    """
    Creates shifted lag features grouped by time series ID.
    All lags are >= 28 days to prevent data leakage over a 28-day forecast horizon.
    """
    print(f"[FEATURE ENG] 1/5 Creating shifted lag features: {lag_days}...")
    
    # Sort values by id and day index to ensure correct shifting
    df = df.sort_values(['id', 'd_int']).reset_index(drop=True)
    
    for lag in lag_days:
        col_name = f'sales_lag_{lag}'
        df[col_name] = df.groupby('id')['sales'].shift(lag).astype(np.float32)
        
    return df

def create_rolling_features(df, windows=[7, 14, 28, 56, 180]):
    """
    Creates rolling window statistics (mean, std) computed on the 28-day shifted lag.
    """
    print(f"[FEATURE ENG] 2/5 Creating rolling window statistics on lag_28: {windows}...")
    
    # Grouped rolling on lag_28
    grouped = df.groupby('id')['sales_lag_28']
    
    for w in windows:
        df[f'rolling_mean_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean()).astype(np.float32)
        df[f'rolling_std_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0).astype(np.float32)
        
    return df

def create_price_features(df):
    """
    Creates relative price and markdown discount features.
    """
    print("[FEATURE ENG] 3/5 Creating price elasticity & markdown features...")
    
    # Max, min, mean price per item-store
    price_stats = df.groupby(['store_id', 'item_id'], observed=True)['sell_price'].agg(['max', 'min', 'mean']).reset_index()
    price_stats.columns = ['store_id', 'item_id', 'max_price', 'min_price', 'mean_price']
    
    df = df.merge(price_stats, on=['store_id', 'item_id'], how='left')
    
    df['price_discount_ratio'] = np.round(
        (df['max_price'] - df['sell_price']) / (df['max_price'] + 1e-5), 3
    ).astype(np.float32)
    
    df['price_relative_to_mean'] = np.round(
        df['sell_price'] / (df['mean_price'] + 1e-5), 3
    ).astype(np.float32)
    
    return df

def create_calendar_cyclical_features(df):
    """
    Creates calendar cyclical (sine/cosine) and policy/payday features.
    """
    print("[FEATURE ENG] 4/5 Creating cyclical calendar & payday features...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    
    # Payday window (1st, 15th, month-end)
    df['is_payday'] = df['day_of_month'].isin([1, 2, 15, 16, 30, 31]).astype(np.int8)
    
    # SNAP Concurrency Index
    df['snap_concurrency'] = (df['snap_CA'] + df['snap_TX'] + df['snap_WI']).astype(np.int8)
    
    # Cyclical trigonometric transforms for Month and Day of Week
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0).astype(np.float32)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0).astype(np.float32)
    df['sin_wday'] = np.sin(2 * np.pi * df['wday'] / 7.0).astype(np.float32)
    df['cos_wday'] = np.cos(2 * np.pi * df['wday'] / 7.0).astype(np.float32)
    
    return df

def encode_categorical_features(df):
    """
    Encodes categorical columns for LightGBM training.
    """
    print("[FEATURE ENG] 5/5 Encoding categorical columns...")
    
    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df

def generate_features(processed_data_dir="data/processed"):
    """
    Loads preprocessed grid_part_1.parquet, computes all feature sets,
    optimizes memory, and saves grid_part_2_features.parquet.
    """
    print("[FEATURE PIPELINE] Starting Automated Feature Engineering Pipeline...")
    
    input_path = os.path.join(processed_data_dir, "grid_part_1.parquet")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Please run Phase 2 preprocessing first.")
        
    print(f"[FEATURE PIPELINE] Loading master grid from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 1. Shifted Lags
    df = create_lag_features(df, lag_days=[28, 29, 35, 42, 56, 364])
    
    # 2. Rolling Window Stats on lag_28
    df = create_rolling_features(df, windows=[7, 14, 28, 56, 180])
    
    # 3. Price Features
    df = create_price_features(df)
    
    # 4. Cyclical Calendar & Payday Features
    df = create_calendar_cyclical_features(df)
    
    # 5. Categorical Encodings
    df = encode_categorical_features(df)
    
    # 6. Drop unneeded date column and downcast memory
    df = reduce_mem_usage(df, verbose=True)
    
    output_path = os.path.join(processed_data_dir, "grid_part_2_features.parquet")
    print(f"[FEATURE PIPELINE] Persisting feature matrix to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Feature Engineering complete. Output saved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"   • Total Rows: {len(df):,}")
    print(f"   • Total Features: {len(df.columns)}")
    print(f"   • Feature Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    generate_features()
