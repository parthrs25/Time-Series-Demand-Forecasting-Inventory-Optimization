import os

# Limit CPU threads at OS/C++ level
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import gc
import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt

def load_config(config_path="config/config.yaml"):
    """Loads MLOps configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[CONFIG] Loaded pipeline configuration from {config_path}")
    return cfg

def load_feature_matrix(processed_data_dir="data/processed"):
    """Loads grid_part_2_features.parquet."""
    path = os.path.join(processed_data_dir, "grid_part_2_features.parquet")
    print(f"[TRAIN] Loading feature matrix from {path}...")
    df = pd.read_parquet(path)
    return df

def train_lightgbm_model(config_path="config/config.yaml"):
    """
    Trains LightGBM model using parameters from config/config.yaml.
    MLflow tracking logs parameters, metrics, and models using SQLite backend.
    """
    cfg = load_config(config_path)
    
    processed_data_dir = cfg["paths"]["processed_data_dir"]
    models_dir = cfg["paths"]["models_dir"]
    val_days = cfg["validation"]["val_days"]
    early_stopping = cfg["validation"]["early_stopping_rounds"]
    params = cfg["model"]
    
    print(f"[TRAIN] Initializing Model Training Pipeline (CPU Capped: {params.get('num_threads', 2)} Threads)...")
    os.makedirs(models_dir, exist_ok=True)
    
    # Configure MLflow SQLite tracking database & artifacts folder
    mlruns_dir = os.path.abspath(cfg["paths"]["mlruns_dir"])
    os.makedirs(mlruns_dir, exist_ok=True)
    
    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path.replace('\\', '/')}")
    print(f"[MLflow] SQLite Tracking URI set to: {mlflow.get_tracking_uri()}")
    
    df = load_feature_matrix(processed_data_dir)
    
    max_d = df['d_int'].max()
    val_cutoff = max_d - val_days + 1
    
    print(f"[TRAIN] Train/Val Split: Train [d_1500 to d_{val_cutoff-1}], Val [d_{val_cutoff} to d_{max_d}] ({val_days} days)...")
    
    ignore_cols = ['id', 'd', 'sales', 'date', 'wm_yr_wk']
    features = [c for c in df.columns if c not in ignore_cols]
    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
    
    train_mask = (df['d_int'] < val_cutoff) & (df['d_int'] >= 1500)
    val_mask = (df['d_int'] >= val_cutoff)
    
    X_train = df[train_mask][features]
    y_train = df[train_mask]['sales']
    
    X_val = df[val_mask][features]
    y_val = df[val_mask]['sales']
    
    val_df_meta = df[val_mask][['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd_int', 'sales', 'sales_lag_28']].copy()
    
    del df
    gc.collect()
    
    print(f"[TRAIN] Training samples: {len(X_train):,}, Validation samples: {len(X_val):,}")
    print(f"[TRAIN] Feature count: {len(features)}")
    
    # Baseline: Naive lag_28
    val_df_meta['naive_pred'] = val_df_meta['sales_lag_28'].fillna(0)
    
    mlflow.set_experiment("Retail_Hierarchical_Demand_Forecasting")
    
    with mlflow.start_run(run_name="LightGBM_Tweedie_SQLite_MLflow"):
        print("[TRAIN] Fitting LightGBM Regressor (MLflow logging to SQLite & mlruns/)...")
        mlflow.log_params(params)
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=True)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_features, free_raw_data=True)
        
        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
                lgb.log_evaluation(period=25),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        preds = model.predict(X_val)
        preds = np.clip(preds, 0, None)
        val_df_meta['lgb_pred'] = preds
        
        val_rmse = np.sqrt(np.mean((val_df_meta['sales'] - val_df_meta['lgb_pred'])**2))
        naive_rmse = np.sqrt(np.mean((val_df_meta['sales'] - val_df_meta['naive_pred'])**2))
        
        val_wape = np.sum(np.abs(val_df_meta['sales'] - val_df_meta['lgb_pred'])) / (np.sum(val_df_meta['sales']) + 1e-5) * 100
        naive_wape = np.sum(np.abs(val_df_meta['sales'] - val_df_meta['naive_pred'])) / (np.sum(val_df_meta['sales']) + 1e-5) * 100
        
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_wape", val_wape)
        mlflow.log_metric("naive_rmse", naive_rmse)
        mlflow.log_metric("naive_wape", naive_wape)
        
        print("\n[EVALUATION BENCHMARK]")
        print(f"   • Naive Baseline RMSE: {naive_rmse:.4f} | WAPE: {naive_wape:.2f}%")
        print(f"   • LightGBM Tweedie RMSE: {val_rmse:.4f} | WAPE: {val_wape:.2f}% (WAPE Improvement: {naive_wape - val_wape:+.2f}%)")
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:20][::-1], importance_df['importance'][:20][::-1], color='#1f77b4')
        plt.title("Top 20 Predictive Features by Gain (LightGBM Tweedie)")
        plt.xlabel("Gain Importance")
        plt.tight_layout()
        
        fi_path = os.path.join(cfg["paths"]["reports_dir"], "figures", "feature_importance.png")
        plt.savefig(fi_path, dpi=200)
        plt.close()
        
        mlflow.log_artifact(fi_path)
        mlflow.lightgbm.log_model(model, artifact_path="model")
        
        model_file = os.path.join(models_dir, "lgb_tweedie_model.joblib")
        joblib.dump(model, model_file)
        print(f"[SUCCESS] Model serialized to {model_file}")
        print(f"[SUCCESS] MLflow experiment run & model logged to {db_path} and {mlruns_dir}")
        
    return model, val_df_meta, importance_df

if __name__ == "__main__":
    train_lightgbm_model()
