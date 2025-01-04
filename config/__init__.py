from .defaults import _C as cfg

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

import os
from config import get_cfg_defaults

base_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取當前腳本所在目錄
config_path = os.path.join(base_dir, "config.yml")  # 拼接路徑
cfg = get_cfg_defaults()
cfg.merge_from_file(config_path)
cfg.freeze()
