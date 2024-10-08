# agn5
## 用來開啟或關閉某些功能，以便於用戶根據需要進行操作
1. `notebook_mode`
   當設定為 `True` 時，表示希望在 Jupyter Notebook環境中運行代碼。這種模式通常會啟用一些特定的功能，如更好的輸出顯示、互動式控制。
2. `viz_mode`
   當設定為 `True` 時，表示啟用視覺化模式。這涉及生成圖表、圖形或其他視覺化表示，以幫助分析數據或呈現結果。在許多數據科學庫中（如 Matplotlib、Seaborn、Plotly 等），視覺化是分析和解釋數據的重要部分。
```
notebook_mode = True 
viz_mode = True
```
# import
## 常見的數據處理、機器學習和深度學習所需的庫和模組。用於構建和訓練機器學習模型，特別是在處理圖形結構數據。
```
import os #用於與操作系統互動，如文件和目錄的管理
import json #用於解析和生成 JSON（JavaScript Object Notation）格式的數據，為常用的數據交換格式
import argparse #用於解析命令行參數，便於在運行程式時傳入不同的參數設置
import time #提供時間相關的功能，如計時和時間戳

import numpy as np #一個強大的數據處理庫，尤其適合進行數值計算和矩陣運算
import copy #用於複製物件，以便在不影響原始物件的情況下進行操作
import torch #PyTorch 深度學習框架的核心庫，提供張量運算和自動微分功能
from torch.autograd import Variable #用於創建可以跟踪其操作以進行自動微分的張量（新版本 PyTorch已被棄用）
import torch.nn.functional as F #提供神經網絡的各種函數和操作，如激活函數和損失計算
import torch.nn as nn #包含構建神經網絡所需的各種層和模型

import matplotlib #用於數據可視化的庫，通常用於創建靜態、動畫和交互式圖形
import matplotlib.pyplot as plt 

import networkx as nx #用於創建、操作和研究複雜網絡結構的庫，特別是在圖論和網絡科學中非常有用
from sklearn.utils.class_weight import compute_class_weight #用於計算類別權重，以處理不平衡數據集的問題

from tensorboardX import SummaryWriter #用於將 PyTorch 訓練過程中的數據（如損失、準確率）記錄到 TensorBoard 以便可視化
from fastprogress import master_bar, progress_bar #用於快速和簡便地顯示進度條的庫，通常在長時間運行的過程中使用
```
```
# Remove warning
import warnings #用於控制警告訊息的顯示，這裡用來忽略某些類型的警告
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning #一個特定於 SciPy 的警告，用於處理稀疏矩陣運算的效率
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import * # 從`config`模組導入配置參數
from utils.graph_utils import * #從`utils.graph_utils`模組導入圖形處理相關的工具
from utils.google_tsp_reader import GoogleTSPReader #導入 Google TSP 讀取器，可能用於讀取旅行推銷員問題的數據
from utils.plot_utils import * #從`utils.plot_utils`導入可視化工具
from models.gcn_model import ResidualGatedGCNModel #導入自定義的 GCN（圖卷積網絡）模型
from utils.model_utils import *從`utils.model_utils`導入模型相關的工具和函數
```
## 設定Jupyter Notebook環境中的特定行為，用於開發和測試數據科學或機器學習程式碼
* `%load_ext autoreload`:Jupyter Notebook 的magic command，用於加載 autoreload 擴展。這個擴展允許自動重新加載模組，使 Notebook 可以在編輯外部 Python 檔案後立即加載變更，而不需重新啟動內核
* `%autoreload 2`:指示 Notebook 自動重新加載所有導入的模組。這樣在更新模組後，就不必每次手動重新導入或重啟 Notebook
```
if notebook_mode == True:
    %load_ext autoreload 
    %autoreload 2
    %matplotlib inline #指示 Jupyter Notebook 直接在單元格中顯示 matplotlib 圖片，而不是在外部窗口中打開
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    #導入用於設置圖像顯示格式的函數。這在 Jupyter Notebook 中尤其有用，可以更清晰地控制圖像質量
    set_matplotlib_formats('png') #指定 matplotlib 繪製的圖像格式為 png，提高圖像顯示的清晰度
```
# Load configurations (讀取超參數設定)
## 根據運行環境設定不同的配置檔案路徑，並加載相應的配置
* 根據不同模式（`notebook_mode` 和 `viz_mode` 的值）來設置配置文件路徑並加載配置內容，這樣可以適應 Notebook 和命令行環境的需求
* 目的:靈活地根據運行環境來自動選擇適當的配置文件
```
if notebook_mode==False: #代表不是在 Jupyter Notebook 環境下運行
    parser = argparse.ArgumentParser(description='gcn_tsp_parser') #用 argparse 來從命令行接收參數
    parser.add_argument('-c','--config', type=str, default="configs/default.json") #指定配置文件的路徑
    args = parser.parse_args() #解析命令行參數
    config_path = args.config #將配置文件的路徑存儲到 config_path
elif viz_mode == True: #在 Notebook 環境中且 viz_mode 為 True
    config_path = "logs/tsp10/config.json" #為視覺化過程提供特定的配置文件
else: #在 Notebook 中，且 viz_mode 為 False
    config_path = "configs/default.json" #使用默認配置文件
config = get_config(config_path) #加載指定路徑的配置文件，從 JSON 文件中讀取設置，並轉換為 Python 字典或其他數據結構
print("Loaded {}:\n{}".format(config_path, config)) #打印出所加載的配置文件路徑及其內容，方便確認所加載的設置是否符合預期
```
## 針對視覺化模式 (viz_mode == True) 的特定配置
* 設定了一些 GPU 和模型運行的參數，以便在可視化時更便於檢查和調整
* 提供了一些可以開啟的選項來測試不同規模的 TSP 問題，適合在進行可視化或模型驗證時使用
```
if viz_mode==True:
    config.gpu_id = "0" #指定使用 GPU 編號為 0 的設備，指在有多個 GPU 的情況下，系統將在第 0 個 GPU 上運行模型
    config.batch_size = 1 #將批次大小設為 1。在可視化模式中，選擇較小的批次大小以便更快地進行計算，方便查看結果
    config.accumulation_steps = 1 #設梯度累積步驟數為 1。表示模型每次更新權重時，都會直接使用該步的梯度，而不需要累積多個步驟的梯度
    config.beam_size = 1280 #設置光束搜索大小為 1280。這在路徑搜尋或解決組合優化問題時尤其有用，更大的 beam size 可探索更多解，但會增加計算成本
    
    # Uncomment below to evaluate generalization to variable sizes in viz_mode
#     config.num_nodes = 50 #設置節點數目為 50。這是針對TSP圖數據，用於測試模型在不同圖規模下的泛化能力
#     config.num_neighbors = 20 #設置圖中每個節點的鄰居數量為 20。可以用於控制圖的稀疏性，即每個節點僅連接到 20 個其他節點
#     config.train_filepath = f"./data/tsp{config.num_nodes}_train_concorde.txt" #設置訓練數據的文件路徑
#     config.val_filepath = f"./data/tsp{config.num_nodes}_val_concorde.txt" #設置驗證數據的文件路徑
#     config.test_filepath = f"./data/tsp{config.num_nodes}_test_concorde.txt" #設置測試數據的文件路徑
#     以上三個路徑基於節點數目（num_nodes）生成
```
# Configure GPU options
