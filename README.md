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

## 環境變數

存在下面檔案內(可改參數)
```
config/config.yml
```

## 操作說明
【1】 使用 txt_to_csv.py
需要在 config 檔內設定 SPLIT_DATA 相關參數路徑：
TXT_PATH：放置 gcode 的 txt 檔案
GCODE_PATH：放置轉換好的 gcode.csv


【2】 使用 signal_slit.py
需要在 config 檔內設定 SPLIT_DATA 相關參數路徑：
DATA_PATHS：工件訊號
GCODE_PATH：Gcode 檔
SAVE_PATH：儲存切割後的訊號與圖片

最終會輸出：
切割後的訊號 pickle 檔
對應的圖片
預設輸入訊號檔案：
震動訊號：['time', 'spindle_front', 'turret']
servo 訊號：['time', 'motor_x_rpm', 'motor_x_current', 'motor_z_rpm', 'motor_z_current', 'spindle_rpm', 'spindle_current']
【注意】若實際使用的訊號欄位不同，可至 signal_slit.py 中（大約第 470 行）修改讀取位置。

!! 目前版本無法對參數化的 Gcode 進行分析 !!


【3】 使用 create_dataset.py
需要在 config 檔內設定 DATASETS 相關參數路徑：
DATA：資料集
FILE_PATHS：切割訊號
MACHININGERROR_PATH：加工誤差 CSV 檔案
執行後會輸出用於訓練的資料集。

【注意】目前 servo 檔案必須包含以下欄位： ['time', 'motor_x_rpm', 'motor_x_current', 'motor_z_rpm', 'motor_z_current', 'spindle_rpm', 'spindle_current', 'torgue']


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
