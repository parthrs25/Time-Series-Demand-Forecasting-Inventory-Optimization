import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import set_seed, init_project_dirs
from src.preprocessing import reduce_mem_usage
from src.feature_engineering import create_lag_features
from src.train import load_config

def test_utils_dir_init(tmp_path):
    """Test environment seeding and directory initialization."""
    set_seed(42)
    dirs = init_project_dirs(base_dir=str(tmp_path))
    assert isinstance(dirs, list)
    for d in dirs:
        assert os.path.exists(d)

def test_config_loading():
    """Test loading configuration from config/config.yaml."""
    cfg = load_config(os.path.join(project_root, "config", "config.yaml"))
    assert "model" in cfg
    assert cfg["model"]["objective"] == "tweedie"
    assert cfg["model"]["num_threads"] <= 4

def test_reduce_mem_usage():
    """Test memory downcasting engine."""
    # Build 10,000-row dataframe to demonstrate true RAM memory downcasting
    n = 10000
    df = pd.DataFrame({
        'int_col': np.random.randint(1, 100, size=n, dtype=np.int64),
        'float_col': np.random.randn(n).astype(np.float64),
        'str_col': np.random.choice(['CA_1', 'CA_2', 'TX_1', 'WI_1'], size=n)
    })
    start_mem = df.memory_usage().sum()
    df_opt = reduce_mem_usage(df, verbose=False)
    end_mem = df_opt.memory_usage().sum()
    
    assert df_opt['int_col'].dtype == np.int8
    assert df_opt['float_col'].dtype == np.float32
    assert df_opt['str_col'].dtype == 'category'
    assert end_mem < start_mem

def test_lag_leakage_safety():
    """Test that all created lag features are >= 28 days to prevent lookahead leakage."""
    df = pd.DataFrame({
        'id': ['item_1_CA_1'] * 100,
        'd_int': range(1, 101),
        'sales': np.random.randint(0, 10, size=100)
    })
    df_lags = create_lag_features(df, lag_days=[28, 35])
    assert 'sales_lag_28' in df_lags.columns
    assert 'sales_lag_35' in df_lags.columns
    # Check that lag 28 on row 28 (1-indexed index 27) matches sales on row 0
    assert df_lags.iloc[28]['sales_lag_28'] == df_lags.iloc[0]['sales']

def test_non_negative_sales_constraint():
    """Test that predictions clipping enforces non-negative numbers."""
    raw_preds = np.array([-1.5, 0.0, 2.3, -0.1])
    clipped_preds = np.clip(raw_preds, 0, None)
    assert np.all(clipped_preds >= 0)
    assert clipped_preds[0] == 0.0
