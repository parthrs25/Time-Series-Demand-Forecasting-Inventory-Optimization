import os
import gc
import numpy as np
import pandas as pd

def reduce_mem_usage(df, verbose=True):
    """
    Iterates through DataFrame columns to downcast numerical data types, 
    reducing RAM memory footprint by 70-80%.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == object:
            df[col] = df[col].astype('category')
            
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"[MEMORY OPTIMIZER] Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
        
    return df

def melt_and_merge_data(raw_data_dir="data/raw", processed_data_dir="data/processed", start_day=1069):
    """
    Transforms wide daily sales matrix into long panel format, merges calendar 
    and price metadata, applies memory optimization, and saves output.
    
    Args:
        raw_data_dir: Path to directory containing raw CSV files
        processed_data_dir: Path to save processed parquet files
        start_day: Starting day index (default=1069 for recent ~2.5 years of data; use 1 for full history)
    """
    print(f"[PREPROCESSING] Starting Phase 2 Data Transformation Pipeline (start_day={start_day})...")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    sales_path = os.path.join(raw_data_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(raw_data_dir, "calendar.csv")
    prices_path = os.path.join(raw_data_dir, "sell_prices.csv")
    
    # 1. Load Raw Datasets
    print("[PREPROCESSING] 1/5 Loading raw datasets...")
    df_sales = pd.read_csv(sales_path)
    df_calendar = pd.read_csv(calendar_path)
    df_prices = pd.read_csv(prices_path)
    
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    # Filter day columns according to start_day
    all_d_cols = [c for c in df_sales.columns if c.startswith('d_')]
    if start_day > 1:
        keep_d_cols = [f'd_{d}' for d in range(start_day, len(all_d_cols) + 1)]
    else:
        keep_d_cols = all_d_cols
        
    print(f"[PREPROCESSING] Selected {len(keep_d_cols)} observation days (d_{start_day} to d_{len(all_d_cols)})...")
    
    # 2. Melt Wide to Long Panel Format
    print("[PREPROCESSING] 2/5 Melting wide sales matrix to long panel series...")
    df_grid = pd.melt(
        df_sales[id_cols + keep_d_cols],
        id_vars=id_cols,
        value_vars=keep_d_cols,
        var_name='d',
        value_name='sales'
    )
    
    # Free memory from wide sales dataframe
    del df_sales
    gc.collect()
    
    # Extract integer day index (e.g. 'd_1069' -> 1069)
    df_grid['d_int'] = df_grid['d'].str.replace('d_', '').astype(np.int16)
    
    # 3. Merge Calendar Data
    print("[PREPROCESSING] 3/5 Merging calendar events and SNAP policy indicators...")
    calendar_cols = ['d', 'date', 'wm_yr_wk', 'wday', 'month', 'year', 
                     'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']
    df_grid = df_grid.merge(df_calendar[calendar_cols], on='d', how='left')
    
    del df_calendar
    gc.collect()
    
    # 4. Merge Sell Price Data
    print("[PREPROCESSING] 4/5 Merging sell price trajectories...")
    df_grid = df_grid.merge(df_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    del df_prices
    gc.collect()
    
    # 5. Apply Memory Downcasting Engine
    print("[PREPROCESSING] 5/5 Applying memory downcasting engine...")
    df_grid = reduce_mem_usage(df_grid, verbose=True)
    
    # 6. Save Processed Master Grid Artifact
    output_path = os.path.join(processed_data_dir, "grid_part_1.parquet")
    df_grid.to_parquet(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Phase 2 complete. Master processed grid saved to: {output_path} ({file_size_mb:.1f} MB)")
    print(f"   • Total Panel Rows: {len(df_grid):,}")
    print(f"   • Total Columns: {len(df_grid.columns)}")
    print(f"   • Column Names: {list(df_grid.columns)}")
    print(f"   • Memory Usage in RAM: {df_grid.memory_usage().sum() / 1024**2:.2f} MB")
    
    return df_grid

if __name__ == "__main__":
    melt_and_merge_data(start_day=1069)
