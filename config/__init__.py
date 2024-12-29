from .defaults import _C as cfg

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

from config import get_cfg_defaults
cfg = get_cfg_defaults()
cfg.merge_from_file("CNC_signal_process/config/config.yml")
cfg.freeze()
