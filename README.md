# agn5
## 用來開啟或關閉某些功能，以便於用戶根據需要進行操作
1. `notebook_mode`：
   當設定為 `True` 時，表示希望在 Jupyter Notebook環境中運行代碼。這種模式通常會啟用一些特定的功能，如更好的輸出顯示、互動式控制。
2. `viz_mode`：
當設定為 `True` 時，表示啟用視覺化模式。這涉及生成圖表、圖形或其他視覺化表示，以幫助分析數據或呈現結果。在許多數據科學庫中（如 Matplotlib、Seaborn、Plotly 等），視覺化是分析和解釋數據的重要部分。
```
notebook_mode = True 
viz_mode = True
```
## 解決旅行推銷員問題（TSP）
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
