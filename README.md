## 安裝python

可以參考此網頁
https://hackmd.io/@smallshawn95/vscode_write_py

## 取得專案

```bash
git clone git@github.com:FCUAI-RPPG/CNC_signal_process.git
```

#創建環境
```bash
conda create --name myenv python=3.10.16
```

#開啟環境
```bash
acivate myenv
```

關閉環境
```bash
conda deactivate
```

## 需求套件

```bash
pip install -r requirements.txt
```

## 運行專案

將 Gcode 轉為 csv 檔
```
python txt_to_csv.py
```

切割訊號
```
python signal_split.py
```

製作訓練用 dataset
```
python create_dataset.py
```

fine tune 模型
```
python main_manual.py
```

整理 DOE4 資料並合併成 signal_split 可讀取名稱
```
python DOE4_file_sorted.py
```

## 環境變數

存在下面檔案內(可改參數)，請盡量不要包含中文路徑以免報錯
```
config/config.yml
```

# 操作說明

## 1. 使用 `txt_to_csv_targetsize.py`
**功能**：  
將 G-code 的 TXT 檔案轉換為 CSV 格式，並新增讀取目標尺寸 (target size) 的功能。

### 使用方法：
1. 在 `config` 檔案內設定 `SPLIT_DATA` 相關參數：
   - `TXT_PATH`：G-code 的 TXT 檔案存放位置
   - `GCODE_PATH`：轉換後的 G-code CSV 檔案存放位置
2. 執行程式後，在命令列輸入目標尺寸（單位：mm）。


## 2. 使用 `signal_slit.py`
**功能**：  
根據 G-code 分割工件的震動與伺服訊號，並輸出對應的圖片與 pickle 檔案。

### 使用方法：
1. 在 `config` 檔案內設定 `SPLIT_DATA` 相關參數：
   - `DATA_PATHS`：工件訊號存放位置
   - `GCODE_PATH`：G-code 檔案存放位置
   - `SR`：震動訊號的採樣率 (預設 `11024`)
   - `SAVE_PATH`：儲存切割後訊號與圖片的資料夾
   - `COMBINE`：如果訊號檔案被分開存放，請設為 `True`（預設 `False`）
2. 執行程式後，輸出：
   - 切割後的訊號 (`pickle` 檔)
   - 對應的圖片

### 預設輸入訊號欄位：
- **震動訊號**：
  ```python
  ['time', 'spindle_front', 'turret']
- **servo 訊號**：
  ```python
  ['time', 'motor_x_rpm', 'motor_x_current', 'motor_z_rpm', 'motor_z_current', 'spindle_rpm', 'spindle_current']
⚠ **注意**：
- 若實際使用的訊號欄位不同，請修改 `signal_slit.py`（約第 **470** 行）。
- 目前版本**無法分析參數化的 G-code**。


## 3. 使用 `create_dataset.py`
**功能**：  
執行後會輸出用於訓練的資料集。

### 使用方法：
1. 在 `config` 檔案內設定 `DATASETS` 相關參數：
   - `DATA`：資料集存放位置
   - `FILE_PATHS`：切割訊號存放位置
   - `MACHININGERROR_PATH`：加工誤差 CSV 檔案
⚠ **注意**：
- 目前 servo 檔案必須包含以下欄位： ['time', 'motor_x_rpm', 'motor_x_current', 'motor_z_rpm', 'motor_z_current', 'spindle_rpm', 'spindle_current', 'torque']


【4】 使用 main_manual.py
需要在 config 檔內設定下列相關參數路徑：
參閱 PPT 中的參數配置說明，其中詳列了可能需要調整的參數或路徑
MODEL
DATALOADER
SOLVER
TEST
OUTPUT_DIR
【重要】在跑模型前，請先將 dataset 資料集切分為訓練集與測試集，並分別放入：
訓練資料：DATASETS 下的 TRAIN_DATA 
測試資料：TEST 下的 EVALUATE_DATA 

【注意】目前提供的 model 與 machining_error.csv 僅支援 DOE4 與 DOE6


【5】 使用 DOE4_file_sorted.py
修改 DOE4 檔案儲存方式與名稱，使 signal_slit.py 能正常運作
需要在 config 檔內設定下列相關參數路徑：
SERVO_GUIDE_DIR: DOE4 ServoGuide 存放位置資料夾
ACC_DIR: DOE4 震動資料存放位置資料夾
DEST_ROOT: 整理並合併後資料夾存放位置
USE_DOE4: 是否使用 DOE4 訊號，是的話為 True 
