from glob import glob
# import pandas as pd
import numpy as np
import pickle
import pandas as pd
from scipy import interpolate
import tqdm
# from tqdm import tqdm
from config import cfg


filepaths = sorted(glob(cfg.DATASETS.FILE_PATHS))
if not filepaths:
    raise FileNotFoundError(f"No files matched the path: {cfg.DATASETS.FILE_PATHS}")
print(filepaths)

dataset = pd.DataFrame({'filepaths':filepaths})
dataset['name'] = dataset.filepaths.str.rsplit("/",n=1).str[1].str.split(".pickle").str[0]
machining_error = pd.concat([pd.read_csv(path) for path in glob(cfg.DATASETS.MACHININGERROR_PATH)])
dataset = dataset.merge(machining_error, how='inner', on=['name'])
dataset['No'] = dataset['No'].astype(np.int32)
dataset.sort_values(['name'], inplace=True)
# dataset['setting'] = dataset['filepaths'].str.extract(r'(DOE6-\d{1,2})')
dataset['date'] = pd.to_datetime(dataset['name'].str.extract(r'(\d{8})')[0], format='%Y%m%d').dt.strftime('%m%d')
dataset['time'] = dataset['filepaths'].str.extract(r'O\d{4}-\d{8}(\d{4})\d{5}')
dataset.to_csv(cfg.DATASETS.DATA, index=False)

dataset = pd.read_csv(cfg.DATASETS.DATA)
measurement_point_ratio = np.array([1/20]) #YOKE/17 ATRANS/20
for batch in tqdm(dataset[['filepaths', '1mm', 'target_size']].values):
    filepath = batch[0]
    measurement_value = batch[1:-1].astype(np.float32) - batch[-1]
    
    with open(filepath, 'rb') as f:
        signals = pickle.load(f)
    
    for idx, signal in enumerate(signals):
        # measurement_point_index = (np.floor(measurement_point_ratio * signal.shape[1]) - 1).astype('int')
        
        # f = interpolate.interp1d(measurement_point_index, measurement_value, fill_value="extrapolate")
        # xnew = np.arange(signal.shape[1])
        # ynew = f(xnew)

        ynew = np.concatenate([measurement_value] * signal.shape[1])
        
        signal = np.concatenate([signal[:9], ynew[np.newaxis, :]], axis=0)
        signals[idx] = signal
    
    with open(filepath, 'wb') as f:
        pickle.dump(signals, f)