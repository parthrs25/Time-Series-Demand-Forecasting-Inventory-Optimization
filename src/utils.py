import os
import random
import numpy as np

def set_seed(seed=42):
    """Set global random seed for 100% reproducible execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[INFO] Random seed set to {seed}")

def init_project_dirs(base_dir=None):
    """Initialize standardized MLOps project directory structure."""
    if base_dir is None:
        base_dir = os.getcwd()
        
    dirs = [
        os.path.join(base_dir, 'config'),
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data', 'processed'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'reports', 'figures'),
        os.path.join(base_dir, 'src'),
        os.path.join(base_dir, 'notebooks'),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    print("[INFO] Directory structure initialized:")
    for d in dirs:
        print(f"   - {d}")
    return dirs
