### 取得專案

```bash
git clone git@github.com:FCUAI-RPPG/CNC_signal_process.git
```

## 需求套件

```bash
pip install -r requirements.txt
```

## 運行專案

將 Gcode 轉為 csv 檔
```
txt_to_csv.py
```

切割訊號
```
signal_slit.py
```

製作訓練用 dataset
```
create_dataset.py
```

fine tune 模型
```
main_manual.py
```

## 環境變數

存在下面檔案內(可改參數)
```
config/config.yaml
```

## 操作細項

使用 txt_to_csv.py 的話要到檔案內改 input_file 和 output_file 位置

使用 signal_slit.py 需要設置 config 檔內的 SPLIT_DATA 相關參數路徑

使用 create_dataset.py 需要設置 config 檔內的 DATASETS 相關參數路徑

使用 main_manual.py 需要設置 config 檔內的 MODEL、DATALOADER、SOLVER、TEST、OUTPUT_DIR 相關參數路徑
