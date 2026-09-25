import os
import json
import numpy as np
import pandas as pd

def calculate_wape(actual, predicted):
    """Weighted Absolute Percentage Error (WAPE)."""
    return np.sum(np.abs(actual - predicted)) / (np.sum(actual) + 1e-5) * 100

def calculate_rmse(actual, predicted):
    """Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((actual - predicted)**2))

def evaluate_forecast_accuracy(reconciled_df, reports_dir="reports"):
    """
    Evaluates forecast accuracy across all 6 hierarchy tiers, comparing
    the Naive Baseline vs. LightGBM Reconciled model.
    """
    print("\n[EVALUATION] === Multi-Level Hierarchical Performance Audit ===")
    os.makedirs(reports_dir, exist_ok=True)
    
    levels = [
        'Level_0_National',
        'Level_1_State',
        'Level_2_Store',
        'Level_3_Cat_Store',
        'Level_4_Dept_Store',
        'Level_5_Item_Store'
    ]
    
    report_rows = []
    
    for lvl in levels:
        sub = reconciled_df[reconciled_df['level'] == lvl]
        if len(sub) == 0:
            continue
            
        actual = sub['sales'].values
        lgb_pred = sub['reconciled_pred'].values
        naive_pred = sub['naive_pred'].values
        
        lgb_wape = calculate_wape(actual, lgb_pred)
        naive_wape = calculate_wape(actual, naive_pred)
        
        lgb_rmse = calculate_rmse(actual, lgb_pred)
        naive_rmse = calculate_rmse(actual, naive_pred)
        
        wape_lift = naive_wape - lgb_wape
        
        report_rows.append({
            'Hierarchy Level': lvl,
            'Series Count': sub['hierarchy_id'].nunique() if 'hierarchy_id' in sub.columns else 1,
            'Total Volume': int(np.sum(actual)),
            'Naive WAPE (%)': round(naive_wape, 2),
            'LightGBM WAPE (%)': round(lgb_wape, 2),
            'WAPE Reduction (%)': round(wape_lift, 2),
            'LightGBM RMSE': round(lgb_rmse, 3)
        })
        
    metrics_df = pd.DataFrame(report_rows)
    
    try:
        print("\n" + metrics_df.to_markdown(index=False))
    except Exception:
        print("\n" + str(metrics_df))
        
    # Save Metrics Report Markdown & JSON
    md_path = os.path.join(reports_dir, "metrics_table.md")
    json_path = os.path.join(reports_dir, "metrics_summary.json")
    
    try:
        metrics_df.to_markdown(md_path, index=False)
    except Exception:
        metrics_df.to_csv(os.path.join(reports_dir, "metrics_table.csv"), index=False)
        
    metrics_df.to_json(json_path, orient='records', indent=2)
    
    print(f"\n[SUCCESS] Metrics audit persisted to {md_path} and {json_path}")
    return metrics_df

if __name__ == "__main__":
    pass
