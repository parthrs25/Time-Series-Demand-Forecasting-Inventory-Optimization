import os
import sys
import argparse

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import set_seed, init_project_dirs
from src.data_ingestion import run_data_ingestion
from src.preprocessing import melt_and_merge_data
from src.feature_engineering import generate_features
from src.train import train_lightgbm_model, load_config
from src.hts import reconcile_hierarchical_forecasts
from src.evaluate import evaluate_forecast_accuracy
from src.inference import run_inventory_reorder_pipeline

def run_mlops_pipeline(stage="all", config_path="config/config.yaml"):
    """Master MLOps Pipeline Execution Engine."""
    print("=" * 80)
    print("[PIPELINE] RETAIL MULTI-STORE HIERARCHICAL DEMAND FORECASTER - MLOPS PIPELINE")
    print("=" * 80)
    
    cfg = load_config(config_path)
    set_seed(42)
    dirs = init_project_dirs(base_dir=project_root)
    
    # Stage 1: Data Ingestion
    if stage in ["ingest", "all"]:
        print("\n--- PHASE 1: DATA INGESTION ---")
        run_data_ingestion(
            raw_data_dir=cfg["paths"]["raw_data_dir"],
            use_real_data=cfg["preprocessing"]["use_real_data"]
        )
        
    # Stage 2: Data Preprocessing
    if stage in ["preprocess", "all"]:
        print("\n--- PHASE 2: DATA PREPROCESSING & MEMORY DOWNCASTING ---")
        melt_and_merge_data(
            raw_data_dir=cfg["paths"]["raw_data_dir"],
            processed_data_dir=cfg["paths"]["processed_data_dir"],
            start_day=cfg["preprocessing"]["start_day"]
        )
        
    # Stage 3: Feature Engineering
    if stage in ["features", "all"]:
        print("\n--- PHASE 3: AUTOMATED FEATURE ENGINEERING ---")
        generate_features(processed_data_dir=cfg["paths"]["processed_data_dir"])
        
    # Stage 4: Model Training & MLflow Tracking
    if stage in ["train", "all"]:
        print("\n--- PHASE 4: MODEL TRAINING & MLFLOW EXPERIMENT TRACKING ---")
        model, val_preds, _ = train_lightgbm_model(config_path)
        
    # Stage 5: HTS Reconciliation
    if stage in ["hts", "all"]:
        print("\n--- PHASE 5: HTS RECONCILIATION ---")
        if 'val_preds' not in locals():
            import joblib
            model_path = os.path.join(cfg["paths"]["models_dir"], "lgb_tweedie_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model binary missing. Please run stage 'train' first.")
            model, val_preds, _ = train_lightgbm_model(config_path)
            
        reconciled_df = reconcile_hierarchical_forecasts(val_preds, models_dir=cfg["paths"]["models_dir"])
        
    # Stage 6: Evaluation Audit
    if stage in ["evaluate", "all"]:
        print("\n--- PHASE 6: MULTI-LEVEL EVALUATION AUDIT ---")
        if 'reconciled_df' not in locals():
            rec_path = os.path.join(cfg["paths"]["models_dir"], "reconciled_forecasts.parquet")
            import pandas as pd
            reconciled_df = pd.read_parquet(rec_path)
            
        evaluate_forecast_accuracy(reconciled_df, reports_dir=cfg["paths"]["reports_dir"])
        
    # Stage 7: Production Inventory Reorder Engine
    if stage in ["inference", "all"]:
        print("\n--- PHASE 7: PRODUCTION INVENTORY REORDER ENGINE ---")
        if 'reconciled_df' not in locals():
            rec_path = os.path.join(cfg["paths"]["models_dir"], "reconciled_forecasts.parquet")
            import pandas as pd
            reconciled_df = pd.read_parquet(rec_path)
            
        run_inventory_reorder_pipeline(
            reconciled_df,
            service_level=cfg["inventory"]["service_level"],
            lead_time_days=cfg["inventory"]["lead_time_days"],
            reports_dir=cfg["paths"]["reports_dir"]
        )
        
    print("\n" + "=" * 80)
    print("[SUCCESS] MLOPS PIPELINE STAGE COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Demand Forecaster MLOps Pipeline Orchestrator")
    parser.add_argument("--stage", type=str, default="all", choices=["ingest", "preprocess", "features", "train", "hts", "evaluate", "inference", "all"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    run_mlops_pipeline(stage=args.stage, config_path=args.config)
