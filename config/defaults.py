from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# SPLIT_DATA
# -----------------------------------------------------------------------------
_C.SPLIT_DATA = CN()
_C.SPLIT_DATA.DATA_PATHS = ""
_C.SPLIT_DATA.GCODE_PATH = ""
_C.SPLIT_DATA.SR = 11024
_C.SPLIT_DATA.SAVE_PATH = ""

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'sequential_angry_fox'
_C.MODEL.PRETRAINED_PATH = ''
_C.MODEL.FILTERS = 32
_C.MODEL.NUM_LAYERS = 2
_C.MODEL.SEQUENCE_LENGTH = 10240
_C.MODEL.LORA = False
_C.MODEL.DROPOUT_RATE = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.DATA = ''
_C.DATASETS.FILE_PATHS = ''
_C.DATASETS.MACHININGERROR_PATH = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0
_C.DATALOADER.MAX_QUEUE_SIZE = 0
_C.DATALOADER.WORKPIECE_LENGTH = 50.0
_C.DATALOADER.NORM_MEAN = ''
_C.DATALOADER.NORM_STD = ''

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.M0MENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-10
_C.SOLVER.NORM_WEIGHT_DECAY = 0.0
_C.SOLVER.LR_WARMUP_EPOCHS = 5
_C.SOLVER.LR_WARMUP_DECAY = 0.01

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32
_C.TEST.WEIGHT = ""
_C.TEST.EVALUATE_ONLY = 'off'
_C.TEST.EVALUATE_DATA = ''
_C.TEST.LIM_MIN = -0.014
_C.TEST.LIM_MAX = 0.048
_C.TEST.LOCATION = [10,30,50]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.DESCRIPTION = ""

