### 取得專案

```bash
git clone git@github.com:hsiangfeng/README-Example-Template.git
```

### 需求套件

```bash
npm install
```

### 運行專案

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

存在下面檔案位置內
```
config/config.yaml
```

## 資料夾與檔案說明

- config - 參數放置處
  - defaults.py - 預設參數
  - config.yaml - 調整參數位置(可動)
- model - 模型放置處
  - choose_model.py - 選擇模型
  - net.py - 模型位置
- utils - 常用工具放置處
  - logger - 日誌設定
- modules - 模組放置處
- create_dataset.py - 製作dataset
- main.py - 適合 pretrain 用
- main_manual.py - 適合 fine tune 用
- signal_slit.py - 訊號自動切割
- txt_to_csv.py - 將 Gcode 轉為 csv 檔
...

## 專案技術

- Node.js v16.15.0
- Vue v3.2.20
- Vite v4.0.4
- Vue Router v4.0.11
- Axios v0.24.0
- Bootstrap v5.1.3
...


