import os
import numpy as np
import pandas as pd
from scipy.stats import norm

def run_inventory_reorder_pipeline(reconciled_df, service_level=0.95, lead_time_days=7, reports_dir="reports"):
    """
    Translates daily demand forecasts into operational supply chain decisions:
    - Calculates Safety Stock (SS) based on target service level Z-score.
    - Calculates Reorder Point (ROP) based on supplier lead time.
    - Generates actionable automated replenishment alerts.
    """
    print("\n[INVENTORY ENGINE] Initializing Supply Chain Reorder Decision Pipeline...")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Filter bottom SKU level
    item_df = reconciled_df[reconciled_df['level'] == 'Level_5_Item_Store'].copy()
    
    # Ensure categorical columns are converted to str to prevent Cartesian product expansion
    for col in ['id', 'item_id', 'dept_id', 'store_id', 'state_id']:
        if col in item_df.columns:
            item_df[col] = item_df[col].astype(str)
    
    # Calculate Mean Daily Forecast and Demand Volatility (Std) per SKU-Store (observed=True for memory efficiency)
    sku_stats = item_df.groupby(['id', 'item_id', 'dept_id', 'store_id', 'state_id'], observed=True).agg(
        mean_daily_demand=('reconciled_pred', 'mean'),
        std_daily_demand=('reconciled_pred', 'std'),
        total_28d_forecast=('reconciled_pred', 'sum')
    ).reset_index()
    
    # Z-score for service level (e.g. Z = 1.645 for 95% in-stock availability)
    z_score = norm.ppf(service_level)
    
    # Safety Stock = Z * std * sqrt(lead_time)
    sku_stats['safety_stock'] = np.ceil(
        z_score * sku_stats['std_daily_demand'].fillna(0.5) * np.sqrt(lead_time_days)
    ).astype(int)
    
    # Reorder Point (ROP) = (Daily Demand * Lead Time) + Safety Stock
    sku_stats['lead_time_demand'] = np.ceil(sku_stats['mean_daily_demand'] * lead_time_days).astype(int)
    sku_stats['reorder_point_rop'] = sku_stats['lead_time_demand'] + sku_stats['safety_stock']
    
    # Simulated current on-hand store inventory (Uniform 0.4x to 1.8x of ROP for demonstration)
    np.random.seed(42)
    inventory_factor = np.random.uniform(0.4, 1.8, size=len(sku_stats))
    sku_stats['current_on_hand_inventory'] = np.ceil(sku_stats['reorder_point_rop'] * inventory_factor).astype(int)
    
    # Generate Inventory Status & Action Recommendation
    def determine_action(row):
        if row['current_on_hand_inventory'] < (row['safety_stock']):
            return 'CRITICAL_STOCKOUT_RISK'
        elif row['current_on_hand_inventory'] <= row['reorder_point_rop']:
            return 'PLACE_PURCHASE_ORDER'
        else:
            return 'SUFFICIENT_STOCK'
            
    sku_stats['inventory_status'] = sku_stats.apply(determine_action, axis=1)
    
    # Calculate recommended order quantity
    sku_stats['recommended_order_quantity'] = np.where(
        sku_stats['inventory_status'] != 'SUFFICIENT_STOCK',
        np.maximum(0, (sku_stats['reorder_point_rop'] * 1.5 - sku_stats['current_on_hand_inventory']).astype(int)),
        0
    )
    
    # Summary of alerts
    status_summary = sku_stats['inventory_status'].value_counts()
    print("\n[INVENTORY DECISION DASHBOARD] Operational Inventory Status:")
    for status, count in status_summary.items():
        print(f"   • {status}: {count:,} SKUs ({count/len(sku_stats)*100:.1f}%)")
        
    out_csv = os.path.join(reports_dir, "inventory_reorder_recommendations.csv")
    sku_stats.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Purchase order replenishment table saved to: {out_csv}")
    
    return sku_stats

if __name__ == "__main__":
    pass
