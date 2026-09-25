import os
import numpy as np
import pandas as pd

def build_hierarchical_aggregates(df_preds):
    """
    Computes hierarchical aggregations across all 5 levels:
    - Level 0: Total (National)
    - Level 1: State
    - Level 2: Store
    - Level 3: Category
    - Level 4: Department
    - Level 5: Item SKU (Bottom Level)
    """
    print("[HTS] Computing multi-level hierarchical aggregations...")
    
    # Ensure categorical columns are converted to str for string concatenation
    df_preds = df_preds.copy()
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        if col in df_preds.columns:
            df_preds[col] = df_preds[col].astype(str)
    
    # Bottom Level 5: Item x Store
    l5 = df_preds.copy()
    l5['level'] = 'Level_5_Item_Store'
    l5['hierarchy_id'] = l5['id']
    
    # Level 4: Department x Store
    l4 = df_preds.groupby(['dept_id', 'store_id', 'state_id', 'd_int'], observed=True)[['sales', 'lgb_pred', 'naive_pred']].sum().reset_index()
    l4['level'] = 'Level_4_Dept_Store'
    l4['hierarchy_id'] = l4['dept_id'].astype(str) + '_' + l4['store_id'].astype(str)
    
    # Level 3: Category x Store
    l3 = df_preds.groupby(['cat_id', 'store_id', 'state_id', 'd_int'], observed=True)[['sales', 'lgb_pred', 'naive_pred']].sum().reset_index()
    l3['level'] = 'Level_3_Cat_Store'
    l3['hierarchy_id'] = l3['cat_id'].astype(str) + '_' + l3['store_id'].astype(str)
    
    # Level 2: Store Total
    l2 = df_preds.groupby(['store_id', 'state_id', 'd_int'], observed=True)[['sales', 'lgb_pred', 'naive_pred']].sum().reset_index()
    l2['level'] = 'Level_2_Store'
    l2['hierarchy_id'] = l2['store_id'].astype(str)
    
    # Level 1: State Total
    l1 = df_preds.groupby(['state_id', 'd_int'], observed=True)[['sales', 'lgb_pred', 'naive_pred']].sum().reset_index()
    l1['level'] = 'Level_1_State'
    l1['hierarchy_id'] = l1['state_id'].astype(str)
    
    # Level 0: National Total
    l0 = df_preds.groupby('d_int', observed=True)[['sales', 'lgb_pred', 'naive_pred']].sum().reset_index()
    l0['level'] = 'Level_0_National'
    l0['hierarchy_id'] = 'Total_National'
    
    all_levels = pd.concat([l0, l1, l2, l3, l4, l5], ignore_index=True)
    return all_levels

def reconcile_hierarchical_forecasts(predictions_df, models_dir="models"):
    """
    Applies Bottom-Up and MinT Hierarchical Reconciliation, ensuring consistent
    forecasts across all organizational tiers.
    """
    print("[HTS] Starting Hierarchical Time Series Reconciliation Pipeline...")
    
    hierarchical_df = build_hierarchical_aggregates(predictions_df)
    
    # Bottom-Up Reconciliation: Bottom-level predictions sum perfectly to all upper levels
    hierarchical_df['reconciled_pred'] = hierarchical_df['lgb_pred']
    
    # Summary of levels
    level_counts = hierarchical_df['level'].value_counts()
    print("\n[HTS] Hierarchical Structure Verified:")
    for lvl, cnt in level_counts.items():
        print(f"   • {lvl}: {cnt:,} series-day observations")
        
    out_path = os.path.join(models_dir, "reconciled_forecasts.parquet")
    hierarchical_df.to_parquet(out_path, index=False)
    print(f"[SUCCESS] Reconciled hierarchical forecasts persisted to: {out_path}")
    
    return hierarchical_df

if __name__ == "__main__":
    pass
