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
    # 導入用於設置圖像顯示格式的函數。這在 Jupyter Notebook 中尤其有用，可以更清晰地控制圖像質量
    set_matplotlib_formats('png')  # 指定 matplotlib 繪製的圖像格式為 png，提高圖像顯示的清晰度
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
結果:
```
Loaded logs/tsp10/config.json:
{'expt_name': 'tsp10', 'gpu_id': '0', 'train_filepath': './data/tsp10_train_concorde.txt', 'val_filepath': './data/tsp10_val_concorde.txt', 'test_filepath': './data/tsp10_test_concorde.txt', 'num_nodes': 12, 'num_neighbors': 1, 'node_dim': 3, 'voc_nodes_in': 2, 'voc_nodes_out': 2, 'voc_edges_in': 3, 'voc_edges_out': 2, 'beam_size': 1280, 'hidden_dim': 2, 'num_layers': 20, 'mlp_layers': 3, 'aggregation': 'mean', 'max_epochs': 1500, 'val_every': 5, 'test_every': 100, 'batch_size': 1, 'batches_per_epoch': 1000, 'accumulation_steps': 1, 'learning_rate': 0.001, 'decay_rate': 1.01, 'q_edges_out': 1}
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
## 設定運行深度學習模型時使用的 GPU
1. `os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"`

    指定 CUDA 裝置的選擇順序。PCI_BUS_ID 是一個設定值，表示系統將根據 GPU 在 PCI 線上的實際硬體順序來分配 CUDA 裝置。這樣可以確保裝置 ID 與實際硬體的排列一致
2. `os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)`

   設定了 `CUDA_VISIBLE_DEVICES` 環境變數，用來指定程式可見的 CUDA 裝置。這意味著程式將只會看到並使用 `config.gpu_id` 指定的 GPU 裝置。config.gpu_id 是一個字串變數，代表程式應使用的 GPU ID
```
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
```
## 檢查系統是否有可用的 CUDA 支援 (即 GPU 是否可用)，並根據結果選擇使用 GPU 或 CPU 進行運算
```
if torch.cuda.is_available(): #會返回 `True` 或 `False`，表示是否有可用的 CUDA 設備
    print("CUDA available, using GPU ID {}".format(config.gpu_id)) #告訴用戶 CUDA 是可用的，並且正在使用指定的 GPU ID
    
    dtypeFloat = torch.cuda.FloatTensor
    #將浮點型張量的數據類型設置為 `torch.cuda.FloatTensor`，這樣可以確保張量會在 GPU 上執行計算
    dtypeLong = torch.cuda.LongTensor
    # 將長整數型張量的數據類型設置為 `torch.cuda.LongTensor`，同樣確保這類張量會在 GPU 上運算
    torch.cuda.manual_seed(1) #設置 CUDA 隨機數生成器的種子值，以確保運行過程中的隨機數生成是可重複的，有助於實驗的可再現性
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor #將浮點型張量的數據類型設置為 `torch.FloatTensor`，這樣張量會在 CPU 上進行計算
    dtypeLong = torch.LongTensor #將長整數型張量的數據類型設置為 `torch.LongTensor`，同樣這些張量會在 CPU 上運行
    torch.manual_seed(1) #設置 CPU 上的隨機數生成器的種子值，以確保 CPU 運算中的隨機性也是可重現的
```
結果:
```
CUDA not available
```
# Test data loading (測試讀取一筆資料)
## 主要在 Notebook 模式下運行，負責讀取 TSP 資料集，生成一個批次的資料，並打印出該批次中各種資料的形狀和內容。最後，使用 `plot_tsp` 函數將其中一個樣本的 TSP 圖像繪製出來，以便進行視覺化檢查和分析
```
if notebook_mode: #查是否處於 Notebook 模式
    num_nodes = config.num_nodes #圖中節點的數量，從配置檔讀取
    num_neighbors = config.num_neighbors #每個節點的鄰居數量，從配置檔讀取
    batch_size = config.batch_size #每批次處理的資料量，從配置檔讀取
    #train_filepath = config.train_filepath
    train_filepath = "./data/tsp10_train_concorde_data1.txt"
    #訓練資料的檔案路徑。原本是從配置檔讀取，但被註解掉，改為固定路徑 "./data/tsp10_train_concorde_data1.txt"

    dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size, train_filepath)
    #GoogleTSPReader:自定義的資料讀取器，用於讀取 TSP（旅行推銷員問題）的資料
    print("Number of batches of size {}: {}".format(batch_size, dataset.max_iter))
    #dataset.max_iter：表示資料集中可以生成的批次數量。打印出來以確認資料集的大小

    t = time.time() #記錄當前時間，用於計算批次生成所需的時間
    batch = next(iter(dataset))  # Generate a batch of TSPs #從TSP資料集中取得下一個批次的資料
    print("Batch generation took: {:.3f} sec".format(time.time() - t))
    print(batch)
    #打印批次生成所需的時間和批次內容：這有助於了解資料讀取和處理的效率
    print("edges:", batch.edges.shape) #圖的邊資訊，形狀（例如：[batch_size, num_edges, ...]）
    print("edges_values:", batch.edges_values.shape) #邊的權重或距離值
    print("edges_targets:", batch.edges_target.shape) #目標邊，指示在 TSP 解中應該選擇哪些邊
    print("nodes:", batch.nodes.shape) #圖中的節點
    print("nodes_target:", batch.nodes_target.shape) #節點的目標標籤，根據 TSP 最優解設定
    print("nodes_coord:", batch.nodes_coord.shape) #節點的座標，通常表示城市的位置
    print("tour_nodes:", batch.tour_nodes.shape) #旅行路線中的節點順序
    print("tour_len:", batch.tour_len.shape) #行程的總長度
    print("ET",batch.edges_target) #打印 edges_target 的內容，方便檢查目標邊的具體值
    #edges：圖的邊的資訊（即節點之間的連接）
    #edges_values：邊的權重或距離值。
    #edges_targets：目標邊，表示解決 TSP 時最優解中哪條邊應該被選擇。
    #nodes：圖中的節點。
    #nodes_target：節點的目標標籤（根據 TSP 最優解）。
    #nodes_coord：節點的座標（可能代表城市的地理位置）。
    #tour_nodes：旅行推銷員的完整行程路線（節點順序）。
    #tour_len：行程的總長度。
    
    #繪製 TSP 圖像
    idx = 0 #選擇第一個批次中的第一個樣本進行繪圖
    f = plt.figure(figsize=(5, 5)) #創建一個 5x5 英吋的繪圖窗口
    a = f.add_subplot(111) #添加一個子圖，111 表示1行1列中的第1個子圖
    plot_tsp(a, batch.nodes_coord[idx], batch.edges[idx], batch.edges_values[idx], batch.edges_target[idx])
    #a：繪圖的子圖對象
    #batch.nodes_coord[idx]：選定樣本的節點座標
    #batch.edges[idx]：選定樣本的邊資訊
    #batch.edges_values[idx]：選定樣本的邊權重或距離值
    #batch.edges_target[idx]：選定樣本的目標邊
```
結果:
```
Number of batches of size 1: 1
Batch generation took: 0.016 sec
{'edges': array([[[2., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 2.]]]), 'edges_values': array([[[0.        , 1.0379906 , 0.95073006, 0.25415555, 0.82213467,
         0.92997991, 0.86077359, 0.97946329, 0.88625403, 0.12109495,
         0.80872302, 0.95922917],
        [1.0379906 , 0.        , 0.49241492, 0.86685076, 0.50213688,
         0.54382257, 0.2891628 , 0.09809758, 0.97190759, 1.11399972,
         0.66674291, 0.47356286],
        [0.95073006, 0.49241492, 0.        , 0.70681574, 0.84870253,
         0.0570735 , 0.24325279, 0.54422872, 0.50742419, 1.06135757,
         0.99331855, 0.02129917],
        [0.25415555, 0.86685076, 0.70681574, 0.        , 0.77592814,
         0.68195519, 0.64846005, 0.82620508, 0.6489607 , 0.37361956,
         0.81415639, 0.71703718],
        [0.82213467, 0.50213688, 0.84870253, 0.77592814, 0.        ,
         0.8795263 , 0.60819743, 0.40403929, 1.19265581, 0.84733734,
         0.16460603, 0.83790531],
        [0.92997991, 0.54382257, 0.0570735 , 0.68195519, 0.8795263 ,
         0.        , 0.28224877, 0.59114755, 0.45035069, 1.04310132,
         1.0197805 , 0.07837267],
        [0.86077359, 0.2891628 , 0.24325279, 0.64846005, 0.60819743,
         0.28224877, 0.        , 0.31608438, 0.68548034, 0.95696164,
         0.75729963, 0.23063747],
        [0.97946329, 0.09809758, 0.54422872, 0.82620508, 0.40403929,
         0.59114755, 0.31608438, 0.        , 0.99915804, 1.04841967,
         0.56864533, 0.52723261],
        [0.88625403, 0.97190759, 0.50742419, 0.6489607 , 1.19265581,
         0.45035069, 0.68548034, 0.99915804, 0.        , 1.00668848,
         1.29954362, 0.52872337],
        [0.12109495, 1.11399972, 1.06135757, 0.37361956, 0.84733734,
         1.04310132, 0.95696164, 1.04841967, 1.00668848, 0.        ,
         0.81011136, 1.06887178],
        [0.80872302, 0.66674291, 0.99331855, 0.81415639, 0.16460603,
         1.0197805 , 0.75729963, 0.56864533, 1.29954362, 0.81011136,
         0.        , 0.98410927],
        [0.95922917, 0.47356286, 0.02129917, 0.71703718, 0.83790531,
         0.07837267, 0.23063747, 0.52723261, 0.52872337, 1.06887178,
         0.98410927, 0.        ]]]), 'edges_target': array([[[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]]), 'nodes': array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]), 'nodes_target': array([[ 0.,  3.,  9.,  8.,  4.,  2.,  1.,  7.,  6.,  5., 10., 11.]]), 'nodes_coord': array([[[0.19143259, 0.12439951],
        [1.        , 0.77527832],
        [0.56185319, 1.        ],
        [0.23085024, 0.37547976],
        [1.        , 0.27314145],
        [0.50477969, 1.        ],
        [0.71284329, 0.80928055],
        [1.        , 0.67718074],
        [0.054429  , 1.        ],
        [0.19673707, 0.0034208 ],
        [1.        , 0.10853541],
        [0.58315237, 1.        ]]]), 'tour_nodes': array([[ 0,  6,  5,  1,  4,  9,  8,  7,  3,  2, 10, 11]]), 'tour_len': array([9.51184347])}
edges: (1, 12, 12)
edges_values: (1, 12, 12)
edges_targets: (1, 12, 12)
nodes: (1, 12)
nodes_target: (1, 12)
nodes_coord: (1, 12, 2)
tour_nodes: (1, 12)
tour_len: (1,)
ET [[[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]]
```
![image](https://github.com/Yuu-Hsuan/agn5-12nodes-success/blob/main/img/2.png)
# Instantiate model (建立模組)
`ResidualGatedGCNModel(config, dtypeFloat, dtypeLong)`

這是自定義的模型類，基於殘差門控圖卷積網絡（Residual Gated Graph Convolutional Network）。它接收三個參數：

1. config：包含模型的超參數設定。
2. dtypeFloat：浮點數資料型別，根據之前的設定可能是 torch.cuda.FloatTensor 或 torch.FloatTensor。
3. dtypeLong：長整數資料型別，根據之前的設定可能是 torch.cuda.LongTensor 或 torch.LongTensor。
```
if notebook_mode == True: #檢查是否處於 Notebook 模式
    # Instantiate the network
    net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
    #用 PyTorch 的 `DataParallel` 將模型包裝起來，以支持多 GPU 並行運算。這樣可以在多塊 GPU 上同時進行計算，加速訓練過程
    if torch.cuda.is_available(): #檢查系統是否有可用的 CUDA（GPU）設備
        net.cuda() #將模型移動到 GPU 上，以利用 GPU 的計算能力進行加速
    print(net) #輸出模型的結構，包括各層的名稱、參數數量等資訊，幫助開發者了解模型的組成和規模

    # Compute number of network parameters
    nb_param = 0 #初始化參數計數器為 0
    for param in net.parameters(): #遍歷模型的所有參數
        nb_param += np.prod(list(param.data.size())) #計算每個參數張量的元素總數
        #param.data.size():返回張量的形狀，list():將其轉換為列表，np.prod():計算列表中所有元素的乘積，即張量的總元素數量
    print('Number of parameters:', nb_param)
    #將每個參數張量的元素數量累加到 nb_param 中，以獲得模型的總參數數量

    # Define optimizer(定義優化器)
    learning_rate = config.learning_rate #從配置檔中讀取學習率（learning rate）參數，這是優化器用來更新模型權重的步伐大小
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #使用 Adam 優化器來優化模型的參數。Adam 是一種基於一階和二階矩估計的自適應學習率優化算法，常用於訓練深度學習模型
    print(optimizer) #輸出優化器的配置，包括學習率和參數等資訊，幫助確認優化器是否正確設置
```
結果:
```
self.in_channels 300 #模型的輸入維度是 300。是輸入圖的特徵數量，例如每個節點可能有 300 維的特徵
DataParallel(
  (module): ResidualGatedGCNModel(  #模型的主體:GCN
#g:將輸入的特徵從 300 維降至 150 維。卷積核大小為 (1,1)，步幅也為 (1,1)，表這是一個逐元素的線性變換
    (g): Conv2d(300, 150, kernel_size=(1, 1), stride=(1, 1))
#W是一個兩層的卷積結構，首先將特徵從 150 維升至 300 維，再經過 BatchNorm2d 進行批量正規化，以幫助模型收斂和穩定訓練過程
    (W): Sequential(
      (0): Conv2d(150, 300, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
#theta 和 phi:這兩個卷積層也將輸入特徵從 300 維降至 150 維。通常在 GCN 中，這些是用來生成圖中的鄰接矩陣或權重矩陣的特徵表示
    (theta): Conv2d(300, 150, kernel_size=(1, 1), stride=(1, 1))
    (phi): Conv2d(300, 150, kernel_size=(1, 1), stride=(1, 1))
#nodes_coord_embedding 和 edges_values_embedding：Linear 層將節點的特徵從 3 維嵌入到 2 維，邊的特徵也嵌入到 1 維。這些嵌入是用來處理節點之間的空間座標或邊的值
    (nodes_coord_embedding): Linear(in_features=3, out_features=2, bias=False)
    (edges_values_embedding): Linear(in_features=1, out_features=1, bias=False)
#edges_embedding：Embedding 層將邊的特徵進行嵌入，嵌入 3 個類別的邊並將它們表示為 1 維特徵(F)
    (edges_embedding): Embedding(3, 1)
#圖卷積層的核心部分:定義 20 個 ResidualGatedGCNLayer 層，每一層都包含 node_feat（節點特徵）和 edge_feat（邊特徵）的更新機制。每層使用線性層 U 和 V 來對節點和邊進行特徵變換，並使用批量正規化來穩定訓練過程
    (gcn_layers): ModuleList(
      (0-19): 20 x ResidualGatedGCNLayer(
        (node_feat): NodeFeatures(
          (U): Linear(in_features=2, out_features=2, bias=True)
          (V): Linear(in_features=2, out_features=2, bias=True)
        )
        (edge_feat): EdgeFeatures(
          (U): Linear(in_features=2, out_features=2, bias=True)
          (V): Linear(in_features=2, out_features=2, bias=True)
        )
        (bn_node): BatchNormNode(
          (batch_norm): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
        (bn_edge): BatchNormEdge(
          (batch_norm): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
      )
    )
#mlp_edges 和 mlp_edges_Q：用來對邊進行多層感知器（MLP）操作的層。mlp_edges 包含 2 層線性層，而 mlp_edges_Q 是一個用來計算 Q 值的卷積層，將邊的特徵從 2 維降到 1 維(greedy)
    (mlp_edges): MLP(
      (U): ModuleList(
        (0-1): 2 x Linear(in_features=2, out_features=2, bias=True)
      )
      (V): Linear(in_features=2, out_features=2, bias=True)
    )
    (mlp_edges_Q): MLP_Q(
      (V): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
#模型總共有 182,021 個參數需要訓練。這些參數包括了模型中的所有卷積層、線性層以及嵌入層的權重
Number of parameters: 182021
#Adam 優化器配置：
Adam (
Parameter Group 0
    amsgrad: False #有啟用 AMSGrad，這是一種變體，用來改善收斂
    betas: (0.9, 0.999) #Adam 的兩個超參數，控制一階與二階動量的衰減率
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001 #學習率設置為 0.001
    maximize: False
    weight_decay: 0 #沒有進行權重衰減，通常用於防止過擬合
)
```
# 測試Train一筆資料
```
#定義機台座標
# def Trans_cor(location):
#     if location == 0 : return [3,6]
#     if location == 1 : return [2,5]
#     if location == 2 : return [4,5]
#     if location == 3 : return [1,4]
#     if location == 4 : return [3,4]
#     if location == 5 : return [5,4]
#     if location == 6 : return [1,3]
#     if location == 7 : return [3,3]
#     if location == 8 : return [5,3]
#     if location == 9 : return [2,2]
#     if location == 10: return [4,2]
#     if location == 11: return [3,1]

#每個 location 對應一組座標，座標值被正規化（例如，3/5 和 6/6），這可能是為了將座標標準化到某個範圍內（如 [0,1]）
def Trans_cor(location):
    if location == 0 : return [3/5,6/6]
    if location == 1 : return [2/5,5/6]
    if location == 2 : return [4/5,5/6]
    if location == 3 : return [1/5,4/6]
    if location == 4 : return [3/5,4/6]
    if location == 5 : return [5/5,4/6]
    if location == 6 : return [1/5,3/6]
    if location == 7 : return [3/5,3/6]
    if location == 8 : return [5/5,3/6]
    if location == 9 : return [2/5,2/6]
    if location == 10: return [4/5,2/6]
    if location == 11: return [3/5,1/6]
" between start and end"
"車輛速率 = 1 (m/min) 所以Manhattan distance為移動時間"
#從座標算出曼哈頓距離
def DS(start,end):
    return sum(map(lambda i ,j : abs(i-j),Trans_cor(start),Trans_cor(end)))
    # Trans_cor(start) 和 Trans_cor(end) 取得 start 和 end 位置的座標
    # map 和 lambda 計算各坐標軸的絕對差值，sum 將所有差值相加得到總距離
```
## 初始化節點和邊權重 (batch.edges_values)
* `batch.edges_values`

   初始化一個三維列表 batch.edges_values，用來存儲每個批次中每個節點對之間的邊權重

   結構: `[batch_size][n][n]`，每個批次有 `n` 個節點，`n` 個節點之間的邊權重初始為 0
```
Node=[0,1,2,3,4,5,6,7,8,9,10,11]
n = 12
batch.edges_values = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]
#batch.edges_values=np.zeros((12,12))

# 自訂屬性 (buffer, idle, car)(個節點定義三個屬性)
buffer=[2, 1, 2, 0, 1, 2, 1, 1, 2, 0, 2, 0] #代表每個節點的緩衝區大小或容量
idle=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] #表示節點是否閒置（1 表示閒置，0 表示不閒置）
car=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] #表示節點是否有車輛（1 表示有，0 表示無）

# 計算邊的權重（曼哈頓距離）
#  功能：填充 batch.edges_values，計算每對節點之間的曼哈頓距離
for i in range(len(Node)):
    for j in range(len(Node)):
        if i==j:
            batch.edges_values[i][j]=0  #如果 i == j，即同一節點，邊權重設為 0
        else:
            first=Node[i]
            second=Node[j]
            batch.edges_values[i][j]=DS(first,second)
            #否則，計算 Node[i] 和 Node[j] 之間的曼哈頓距離，並賦值給 batch.edges_values[i][j]
batch.edges_values=[batch.edges_values] #最後，將 batch.edges_values 包裝成一個列表，形成批次結構

#print(batch.edges_values)

#準備節點座標 (batch.nodes_coord)
#  功能：為每個節點組合 buffer、idle 和 car 屬性，形成節點的座標資訊
batch.nodes_coord=[]

for i in range(len(Node)):
    temp=[]
    temp.insert(0,buffer[i])
    temp.insert(1,idle[i])
    temp.insert(2,car[i])  #創建一個臨時列表 temp，依次插入 buffer[i]、idle[i] 和 car[i]
    batch.nodes_coord.append(temp) #將 temp 添加到 batch.nodes_coord 中
batch.nodes_coord=[batch.nodes_coord]  #最後，將 batch.nodes_coord 包裝成一個列表，形成批次結構
print(batch.nodes_coord) #打印 batch.nodes_coord 以確認內容


#定義邊和目標邊
#  功能：手動設置邊的連接關係和目標邊

#定義圖中節點之間的連接關係，使用二維列表表示。每個子列表代表一個節點的連接情況（1 表示有連接，0 表示無連接）
batch.edges=[[0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
batch.edges=[batch.edges]
batch.nodes=[[1,1,1,1,1,1,1,1,1,1,1,1]]

#義目標邊，用於訓練模型識別哪些邊應該被選擇。用於有監督學習中作為標籤
batch.edges_target=[[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]

#原本是全為 0 的目標邊，後來修改為有特定的目標邊
#batch.edges_target=[[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
batch.edges_target=[batch.edges_target]


# 輸入變數轉換為 PyTorch 變量
#  功能：將資料轉換為 PyTorch 的 Variable，以便於模型處理
#Input variables
x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
#邊的連接信息，轉換為長整數型張量
x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
#邊的權重（曼哈頓距離），轉換為浮點數型張量
x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
#節點的標識，轉換為長整數型張量
x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
#節點的座標資訊（buffer、idle、car），轉換為浮點數型張量
y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
#目標邊，轉換為長整數型張量
#requires_grad=False：這些變量在訓練過程中不需要計算梯度，因為它們是輸入數據
##

#print(batch)

#計算類別權重
#  功能：計算類別權重，用於處理類別不平衡問題
# Compute class weights
edge_labels = y_edges.cpu().numpy().flatten() #將 y_edges 從 GPU 移回 CPU，轉換為 NumPy 陣列並展平成一維
edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
#compute_class_weight：來自 sklearn 的函數，用於計算每個類別的權重。"balanced" 模式會根據類別出現的頻率自動調整權重，較少出現的類別權重較大
#dge_cw：計算得到的類別權重，用於後續的損失函數計算

#模型前向傳播
y_preds, q = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
#y_preds：模型的預測輸出
#q：可能是模型的中間結果或附加輸出（具體取決於模型實現）
#調用模型的前向傳播方法，輸入包括邊的連接信息、邊的權重、節點標識、節點座標、目標邊和類別權重
#loss = loss.mean()
print("Output size: {}".format(y_preds.size())) #打印預測結果的尺寸，幫助檢查模型輸出的正確性
#x_edges:指示函數，x_edges_value:distance matrix，
```
結果:
```
[[[2, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 1], [1, 0, 0], [2, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0]]]
Output size: torch.Size([1, 12, 12, 2])
```
## 定義連接函數 (connect 函數)
`if location == 1 : return [0,2]`表示`location == 1`，指編號0跟2的機台路徑
```
#功能：根據給定的位置編號，返回對應的節點連接關係
#每個 location 對應一對節點 [a, b]，表示這對節點之間存在連接
#幾號工單 回傳所對應的節點
def connect(location):
    if location == 0 : return [0,1]
    if location == 1 : return [0,2]
    if location == 2 : return [1,3]
    if location == 3 : return [1,4]
    if location == 4 : return [1,5]
    if location == 5 : return [2,3]
    if location == 6 : return [2,4]
    if location == 7 : return [2,5]
    if location == 8 : return [3,6]
    if location == 9 : return [3,7]
    if location == 10: return [3,8]
    if location == 11 : return [4,6]
    if location == 12 : return [4,7]
    if location == 13 : return [4,8]
    if location == 14 : return [5,6]
    if location == 15 : return [5,7]
    if location == 16 : return [5,8]
    if location == 17 : return [6,9]
    if location == 18 : return [6,10]
    if location == 19 : return [7,9]
    if location == 20 : return [7,10]
    if location == 21 : return [8,9]
    if location == 22: return [8,10]
    if location == 23 : return [9,11]
    if location == 24 : return [10,11]
```
# Reward Function
```
#選擇Goal則給獎勵10，其餘皆-1
#功能：這個函數根據當前狀態 (state)、可行動集合 (actionset) 和選擇的動作 (action) 來返回獎勵
#  獎勵矩陣 reward 是一個 12x12 的二維列表，預設所有動作的獎勵為 -1
#  當選擇到達目標 (Goal) 的動作時，給予獎勵 10
#  具體來說，第 10 和第 11 行的最後一個元素被設為 10，表示這些動作是達成目標的動作
def Find_reward(state,actionset,action):

    reward=[[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    
    return reward
```
# TEST ONE DATA (TEST Function)
```
#初始化距離矩陣 (edges_values)
#  功能：初始化並計算節點之間的曼哈頓距離，存入 batch.edges_values
global edges_value
Node=[0,1,2,3,4,5,6,7,8,9,10,11]
n=12
#創建一個 12x12x12 的三維列表 batch.edges_values，初始值全部為 0
batch.edges_values = [[[0 for k in range(n)] for j in range(n)] for i in range(n)] #Distance matrix固定不變

#用雙層迴圈遍歷所有節點對 (i, j)，若 i == j，距離設為 0
#否則，計算節點 i 和節點 j 之間的曼哈頓距離 DS(first, second)，並存入 batch.edges_values[i][j]
for i in range(len(Node)):
    for j in range(len(Node)):
        if i==j:
            batch.edges_values[i][j]=0
        else:
            first=Node[i]
            second=Node[j]
            batch.edges_values[i][j]=DS(first,second)
            
edges_value = batch.edges_values
```
```
#測試單筆資料函數 (test_one_data_w)
#  功能：測試單筆資料，根據輸入的座標、鄰接矩陣和可行動集合，輸出選擇的動作和對應的 Q 值(Q-Learning)
def test_one_data_w(coord,adjacency_w,action_s):
#  coord：節點的座標資訊
#  adjacency_w：當前狀態下的鄰接矩陣
#  action_s：可行動集合，即當前可以選擇的動作
    global edges_value
    # 資料前處理 (Encoding)
    batch.edges=[[0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
    #batch.edges 為預設的鄰接矩陣

    #batch.edges=[[2,1,1,0,0,0,0,0,0,0,0,0],[0,2,0,1,1,1,0,0,0,0,0,0],[0,0,2,1,1,1,0,0,0,0,0,0],[0,0,0,2,0,0,1,1,1,0,0,0],[0,0,0,0,2,0,1,1,1,0,0,0],[0,0,0,0,0,2,1,1,1,0,0,0],[0,0,0,0,0,0,2,0,0,1,1,0],[0,0,0,0,0,0,0,2,0,1,1,0],[0,0,0,0,0,0,0,0,2,1,1,0],[0,0,0,0,0,0,0,0,0,2,0,1],[0,0,0,0,0,0,0,0,0,0,2,1],[0,0,0,0,0,0,0,0,0,0,0,2]]
    batch.edges=[batch.edges] #指示函數不變，將 batch.edges 包裝成列表
    #batch.edges=adjacency_w  #目前的state_input_edges
    #設定 batch.edges_values、batch.nodes、batch.nodes_coord 和 batch.edges_target
    batch.edges_values=[edges_value] #Distance matrix
    batch.nodes=[[1,1,1,1,1,1,1,1,1,1,1,1]]
    batch.nodes_coord=[coord] #目前的state_input_nodes
    batch.edges_target=[adjacency_w]#目前的state_input_edges

#將 batch.edges、batch.edges_values、batch.nodes、batch.nodes_coord 和 batch.edges_target 轉換為 PyTorch 的 Variable，並指定資料型別
    x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
    x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
    x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

#前向傳播：將處理好的資料輸入模型 net，獲得預測結果 y_preds 和 Q 值 q
    # Forward pass (輸入網路架構，輸出Q值)
    edge_cw=1
    y_preds, q = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)

#選擇最大 Q 值的動作
    q=q.reshape(12,12) #將 Q 值 q 重塑為 12x12 的矩陣
    max_num=torch.argmax(q) #找出 Q 值中最大的元素位置 max_num
    max_num=max_num.item() #濾出 action_set 中的 Q 值，並選擇其中最大的 Q 值對應的動作 Max_choose
    
    #Filter (只採用action_set的Q值)
    in_action_set=[]
    for i in range(len(action_s)):
        position=connect(action_s[i])
        in_action_set.append(q[position[0]][position[1]])

    Max_q=max(in_action_set)
    Max_choose=action_s[in_action_set.index(max(in_action_set))]    

    return [Max_choose,Max_q] #返回選擇的動作 Max_choose 和對應的 Q 值 Max_q
```
# Train One Data (Train Function)
```
#函數定義和參數
#  功能：訓練單筆資料，根據當前狀態和動作集合，更新模型的 Q 值
#  參數：
#  state_now：當前狀態
#  action_set_now：當前狀態下的可行動集合
#  choose_action：選擇的動作
#  state_next：執行動作後的下一個狀態
#  action_set_next：下一個狀態下的可行動集合
def Train_One_Data(state_now,action_set_now,choose_action,state_next,action_set_next):
    global edges_value, loss
    edge_cw = None
#檢查下一個動作集合是否為空
#  功能：如果下一個狀態下沒有可行動集合，則不進行訓練，直接返回
#  原因：避免在沒有可行動的情況下進行更新，防止錯誤
    if len(action_set_next)==0:#若下一個action_set為空，則不學此筆資料
        return;
    #讀取現在的state

    #重新定義機台0和機台11的值
    #  功能：限制機台0的值不超過2，並將機台11的值設為0
    if state_now[0][0] > 2: 
        state_now[0][0]=2
    state_now[11][0]=0
    
    #將action_set_now轉成鄰接矩陣 (adjacency_now)
    #  功能：根據當前的可行動集合 action_set_now，生成當前狀態下的鄰接矩陣 adjacency_now
    adjacency_now=[]    
    temp=np.zeros((12,12)) #創建一個 12x12 的零矩陣 temp
    for i in range(len(action_set_now)):
        num=int(action_set_now[i])
        nodes=connect(num)
        #遍歷 action_set_now 中的每個動作編號 num，使用 connect(num) 函數獲取對應的節點對 [a, b]
        temp[nodes[0]][nodes[1]]=1
        #將 temp[a][b] 設為1，表示節點 a 和節點 b 之間有連接
    adjacency_now.append(temp) #將 temp 添加到 adjacency_now 列表中
    
    #處理下一個狀態 (state_next) 和下一個動作集合 (action_set_next)
    #重新定義機台0和機台11的值
    if state_next[0][0] > 2: 
        state_next[0][0]=2
    state_next[11][0]=0
    #限制 state_next[0][0] 不超過2，並將 state_next[11][0] 設為0

    #將action_set_next轉成鄰接矩陣 adjacency_next
    adjacency_next=[]    
    temp=np.zeros((12,12))
    for i in range(len(action_set_next)):
        num=int(action_set_next[i])
        nodes=connect(num)
        temp[nodes[0]][nodes[1]]=1
    adjacency_next.append(temp) #根據 action_set_next 中的動作編號，生成下一個狀態下的鄰接矩陣 adjacency_next
##################Input Now #######################################################################################
#功能：設置當前狀態下的鄰接矩陣、距離矩陣、節點標識、節點座標和目標鄰接矩陣
    batch.edges=[[0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]] #設置為預設的鄰接矩陣，並包裝成列表
    batch.edges=[batch.edges] #指示函數不變
    #batch.edges=adjacency_now
    batch.edges_values=[edges_value] #Distance matrix，batch.edges_values：設置為全局變數 ；edges_value，即距離矩陣
    batch.nodes=[[1,1,1,1,1,1,1,1,1,1,1,1]] #設置節點標識為全1列表
    batch.nodes_coord=[state_now] #目前的state_input_nodes #設置節點座標為當前狀態 state_now
    batch.edges_target=[adjacency_now]#目前的state_input_edges #設置目標鄰接矩陣為當前的鄰接矩陣 adjacency_now
####################AGN############AGN###############AGN##################AGN#############AGN##################AGN#############
# 轉換為 PyTorch 變量並計算類別權重，並指定資料型別和是否需要計算梯度
#  將批次資料轉換為 PyTorch 的 Variable，以便於模型處理
    # Convert batch to torch Variables
    x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
    x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
    x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

    #使用 compute_class_weight 計算類別權重 edge_cw，使得不同類別的權重平衡
    if type(edge_cw) != torch.Tensor: #如果類別權重 edge_cw 尚未計算，則計算類別權重以處理類別不平衡問題
        edge_labels = y_edges.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
    # Forward pass前向傳播
    #  功能：將處理好的資料輸入模型 net，進行前向傳播，獲得預測結果 y_preds 和 Q 值 q
    y_preds, q = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
############################################################################################################################    
#Next State Data處理下一個狀態的資料
#  功能：處理下一個狀態 state_next，並進行前向傳播，獲得下一個 Q 值 q2
    #batch.edges=adjacency_next:更新 batch.nodes_coord 和 batch.edges_target 為下一個狀態的資料
    batch.nodes_coord=[state_next] #下一筆的state_input_nodes
    batch.edges_target=[adjacency_next]#下一筆的state_input_edges

    #轉換為 PyTorch 的變量 x_nodes_coord 和 y_edges
    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    #進行前向傳播，獲得 y_preds2 和 q2
    y_preds2, q2 = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw) #下一筆的 Q
    #複製 q2 為 q2_original，並重塑為 12x12 的矩陣
    q2_original=q2.clone()
    q2=q2.reshape(12,12)
    q2_original=q2_original.reshape(12,12)

    #選擇下一個動作並計算獎勵
    #  功能：從下一個狀態的 Q 值中選擇可行動集合中最大的 Q 值對應的動作，並計算該動作的獎勵 reward2
    #Filter 找出q2的 a'(q2max_choose)
    in_action_set=[]
    for i in range(len(action_set_next)):
    #遍歷 action_set_next，獲取每個動作在 Q 值矩陣中的位置，並將對應的 Q 值加入 in_action_set
        position=connect(action_set_next[i])
        in_action_set.append(q2[position[0]][position[1]])

    q2max_choose=action_set_next[in_action_set.index(max(in_action_set))]
    #選擇 in_action_set 中最大的 Q 值 Max_q 對應的動作 Max_choose
    
    reward2=Find_reward(state_next,action_set_next,q2max_choose) #根據目前State找出Reward
    #使用 Find_reward 函數計算該動作的獎勵 reward2
    #print("Q2",q2)


    #找出Q2的max值
    #  功能：找出 q2 中最大的 Q 值及其位置            
    max_num=torch.argmax(q2) #使用 torch.argmax(q2) 找出 q2 中最大的元素的位置
    max_num=max_num.item() #將位置轉換為索引，並獲取對應的 Q 值 max_q2_values
    max_q2_values=q2[max_num//12][max_num%12].item()
    
####Update Q#############Update Q############Update Q#############Update Q############Update Q##################Update Q###
#更新 Q 值
#  根據 Q-Learning 的更新規則，更新目標 Q 值 target_q
    original_q=q.clone() #複製並重塑當前的 Q 值 q 為 original_q
    original_q=original_q.reshape(12,12) #12*12

    #計算當前動作 choose_action 的獎勵 reward1
    reward1=Find_reward(state_now,action_set_next,choose_action) #根據目前State找出Reward

    #複製並重塑 q 為 target_q，並分離計算圖 (detach)
    target_q=q.clone().detach()

    target_q=target_q.reshape(12,12) #12*12
    #print("target_q:",target_q)
    #print("action_choose:",action_choose[batch_num+epoch* batches_per_epoch])
    #print("action_choose:",action_choose[1377])

    #使用 connect(choose_action) 獲取選擇動作的節點對 [i, j]
    action_ij=connect(choose_action) #找出action_ij

    #print("action_ij",action_ij)
    #計算目標 Q 值 target_q_values，根據獎勵和下一個狀態的最大 Q 值
    #  Q target(i,j)=reward1(i,j)+γ×max_q2_values，其中 γ=0.9 是折扣因子
    target_q_values=reward1[action_ij[0]][action_ij[1]]+ 0.9 * max_q2_values
    target_q[action_ij[0]][action_ij[1]] = target_q_values #更新 target_q[i][j] 為 target_q_values
###################################################################################################################################
#計算損失並反向傳播
#Compute Loss and backward net
#  計算損失函數，進行反向傳播並更新模型參數
    #print("QQ",original_q)
    #使用 F.smooth_l1_loss 計算 original_q 與 target_q 之間的損失，這是一種結合了 L1 和 L2 損失的平滑損失函數
    loss = F.smooth_l1_loss(original_q, target_q)
    loss = loss.mean() #計算損失的平均值 loss.mean()
    Loss_plt.append(loss.item()) #將損失值添加到 Loss_plt 列表中，用於後續的損失可視化
    optimizer.zero_grad() #重置優化器的梯度 optimizer.zero_grad()
    #print("loss:",loss)
    loss.backward() #執行反向傳播 loss.backward()，計算梯度
    optimizer.step() #更新模型參數 optimizer.step()
```
# 畫圖Initial
```
import pygame as pg #用於創建視覺化環境，顯示車輛和機台的位置及狀態
import pandas as pd #用於處理事件列表和數據框架

#定義顏色:使用RGB值定義了多種顏色
white  = (255, 255, 255)
black  = (  0,   0,   0)
red    = (255,   0,   0)
green  = (  0, 255,   0)
blue   = (  0,   0, 255)
yellow = (255, 255,   0)
pink = (186,217,232)

#畫布字型初始，初始化Pygame庫
pg.init()
clock = pg.time.Clock() #TIME CLOCK設置時鐘
pg.display.set_caption("env") #標題名
screen = pg.display.set_mode((1280,900)) #設定視窗
bg = pg.Surface(screen.get_size()).convert() #創建一個與視窗同尺寸的背景畫布bg
#bg =pg.image.load("girl.jpg")
bg.fill(pink) #填充為粉紅色
font = pg.font.SysFont("simhei", 24)#字樣
font1 = pg.font.SysFont("simhei", 40)#字樣

#定義暫存區圖樣
def rect(bg, color, x, y): #在背景bg上繪製一個帶有黑色邊框和十字線的矩形，用於表示暫存區
    pg.draw.rect(bg, color,[x, y, 100, 100], 0)
    pg.draw.rect(bg, (0,0,0),[x, y, 100, 100], 2)
    pg.draw.line(bg, (0,0,0),(x,y+50), (x+100, y+50), 3)
    pg.draw.line(bg, (0,0,0),(x+50,y), (x+50, y+100), 3)
def car(bg, color, x, y): #car 函數：在背景bg上繪製一個小矩形，用於表示車輛
    pg.draw.rect(bg, color,[x, y, 25, 25], 0)

#畫暫存區 (背景)
#  繪製文字：在畫布上顯示「Begin」和「Goal」標籤，分別位於不同的位置，使用藍色文字和粉紅色背景
text1 = font.render("Begin", True, (0,0,255), pink)
bg.blit(text1, (100,80))

text10 = font.render("Goal", True, (0,0,255), pink)
bg.blit(text10, (600,730))

#載入和設置車輛圖像
#  載入三張相同的車輛圖片car.png，並將其縮放到100x100像素
#V1
truck1 = pg.image.load("car.png")
truck1 = pg.transform.scale(truck1,(100,100))
truck2 = pg.image.load("car.png")
truck2 = pg.transform.scale(truck2,(100,100))
truck3 = pg.image.load("car.png")
truck3 = pg.transform.scale(truck3,(100,100))
#  在每輛車上添加文字標籤「A」、「B」、「C」，用於區分不同的車輛
textA = font1.render("A", True, (0,0,255), pink)
textB = font1.render("B", True, (0,0,255), pink)
textC = font1.render("C", True, (0,0,255), pink)
#pg.draw.circle(truck1, (255,0,0), (25,25), 20, 0)  
truck1.blit(textA, (10,5))
truck2.blit(textB, (30,5))
truck3.blit(textC, (50,5))
#  設置車輛位置：獲取車輛圖像的矩形區塊並設置其起始位置
rect = truck1.get_rect()         #取得球矩形區塊
rect = truck2.get_rect()         #取得球矩形區塊
rect = truck3.get_rect()         #取得球矩形區塊
rect.center = (50,150)        #球起始位置
x, y = rect.topleft            #球左上角坐標

#定義環境資訊顯示函數
# env_info 函數：在畫布上顯示當前的時間資訊，格式為「Time」，使用藍色文字和粉紅色背景
def env_info(Time_info):
    Time_info = str(Time_info)
    text0 = font1.render("Time:"+Time_info+'s', True, (0,0,255), pink)
    bg.blit(text0, (10,10))

#定義車輛位置函數
#  truck_location 函數：根據給定的位置編號s，返回對應的座標，用於更新車輛在畫布上的位置
def truck_location(s):
    if  s == 1  :return (50,150)
    if  s == 2  :return (400,100)
    if  s == 3  :return (700,100)
    if  s == 4  :return (250,250)
    if  s == 5  :return (550,250)
    if  s == 6  :return (850,250)
    if  s == 7  :return (250,450)
    if  s == 8  :return (550,450)
    if  s == 9  :return (850,450)
    if  s == 10 :return (400,650)  #Q_2400_out
    if  s == 11 :return (700, 650)
    if  s == 12 :return (550, 800)

#定義車輛載貨函數
#  truckV1_load、truckV2_load、truckV3_load 函數：
#     分別用於更新和繪製三輛車的位置。根據傳入的truck_loc列表，計算每輛車的新位置並繪製在畫布上
def truckV1_load(truck_loc):    #有貨物紅色
    rect.center = (truck_location(truck_loc[0]+1)[0],truck_location(truck_loc[0]+1)[1]) #更新卡車位置
    screen.blit(truck1, rect.topleft)  #繪製卡車位置
    pg.display.update()
def truckV2_load(truck_loc):    #有貨物紅色
    rect.center = (truck_location(truck_loc[1]+1)[0],truck_location(truck_loc[1]+1)[1]) #更新卡車位置
    screen.blit(truck2, rect.topleft)  #繪製卡車位置
    pg.display.update()
def truckV3_load(truck_loc):    #有貨物紅色
    rect.center = (truck_location(truck_loc[2]+1)[0],truck_location(truck_loc[2]+1)[1]) #更新卡車位置
    screen.blit(truck3, rect.topleft)  #繪製卡車位置
    pg.display.update()

#定義更新數字和工作狀態函數
def update_number(x,y,a): #根據參數a的值，更新特定位置的矩形顏色和顯示的數字
    if   a==0 : #blue #0：綠色矩形，顯示數字「0」
        pg.draw.rect(bg,black,[x-2, y-2, 84, 84], 0)
        pg.draw.rect(bg,green,[x, y, 80, 80], 0)
        text1 = font1.render("0", True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
    elif a==1 : #yellow #1：黃色矩形，顯示數字「1」
        pg.draw.rect(bg,yellow,[x, y, 80, 80], 0)
        text1 = font1.render("1", True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
    elif a==2 : #green #2：紅色矩形，顯示數字「2」
        pg.draw.rect(bg,red,[x, y, 80, 80], 0)
        text1 = font1.render("2", True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
    elif a>=3 : # red #a >= 3：紅色矩形，顯示對應的數字
        pg.draw.rect(bg,red,[x, y, 80, 80], 0)
        a=str(a)
        text1 = font1.render(a, True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
def update_work(x,y,a): #根據參數a的值，更新特定位置的工作狀態矩形
    if   a==0 : #blue #0：綠色矩形，表示工作空閒
        pg.draw.rect(bg,black,[x-2, y-2, 44, 44], 0)
        pg.draw.rect(bg,green,[x, y, 40, 40], 0)
    elif a==1 : #yellow #1：紅色矩形，表示正在工作
        pg.draw.rect(bg,red,[x, y, 40, 40], 0)

#定義背景輸出函數
def output_bg(BG_B,BG_S): #將背景畫布bg繪製到主視窗screen上
#  根據BG_B（機台緩衝區）和BG_S（機台工作狀態）的數據，調用update_number和update_work函數更新各個機台的顯示
    screen.blit(bg, (0,0))
    update_number(100,100,BG_B[0])
    update_number(450, 50,BG_B[1])
    update_number(750, 50,BG_B[2])
    update_number(300,200,BG_B[3])
    update_number(600,200,BG_B[4])
    update_number(900,200,BG_B[5])
    update_number(300,400,BG_B[6])
    update_number(600,400,BG_B[7])
    update_number(900,400,BG_B[8])
    update_number(450,600,BG_B[9])
    update_number(750, 600,BG_B[10])
    update_number(600, 750,BG_B[11])
    update_work(450,135,BG_S[1])
    update_work(750,135,BG_S[2])
    update_work(300,285,BG_S[3])
    update_work(600,285,BG_S[4])
    update_work(900,285,BG_S[5])
    update_work(300,485,BG_S[6])
    update_work(600,485,BG_S[7])
    update_work(900,485,BG_S[8])
    update_work(450,685,BG_S[9])
    update_work(750,685,BG_S[10])
    pg.display.update() #最後，更新Pygame顯示
```
結果:
```
pygame 2.6.1 (SDL 2.28.4, Python 3.12.5)
Hello from the pygame community. https://www.pygame.org/contribute.html
```
```
import random #用於隨機選擇
import numpy as np #於數值計算，特別是矩陣操作
import pandas as pd #處理事件列表和數據框架
from random import choice
import copy #深拷貝
import warnings #用於忽略未來警告
warnings.simplefilter(action='ignore', category=FutureWarning)

#初始化全局變數和列表
loss=0 #用於存儲當前的損失值
Loss_plt=[] #於存儲每次訓練的損失值，便於後續繪圖
day_list=[] #存儲訓練過程中的各種狀態和動作信息
w_state_now=[] #存儲訓練過程中的各種狀態和動作信息
w_buffer_now=[]
w_action_set_now=[]
w_action_choose=[]
car_s_location=[]
train_count=0 #訓練計數器，用於追蹤已訓練的樣本數量
###################################讀取最佳 Check Point Model  (若要讀取最佳訓練成果，請取消註記，並註記掉下方Train函式)
#讀取最佳檢查點模型（註解部分）
#  功能：如果需要使用已訓練的最佳模型，可以取消註解這部分代碼，載入保存的模型並設置為評估模式
#  說明：目前這部分代碼被註解，表示不使用已保存的模型，而是從頭開始訓練
# net=torch.load('Best_Model.pt')
# net.eval()
########################################################
#定義狀態生成函數
#  state_generator 函數：將當前的緩衝區buf、閒置狀態idle和車輛位置loc轉換為神經網絡可接受的狀態格式
#  buf：機台緩衝區數量
#  idle：機台閒置狀態（0表示不閒置，1表示閒置）
#  loc：車輛當前位置（節點編號）
def state_generator(buf, idle, loc): #資料前處理 (將當前Buffer、 Idel、 車位置，轉換成State)
    input_state=[]
    car_loc=[0 for k in range(12)]
    car_loc[loc] = 1
    for i in range(12):
        input_state.append([buf[i],idle[i],car_loc[i]]) 

    return input_state #輸出：一個包含12個節點的狀態列表，每個節點包含緩衝區數量、閒置狀態和車輛位置標識

#定義動作位置函數
#  根據動作編號action，返回當前位置和目標位置的節點編號。
#  功能：定義了25個可能的動作，每個動作對應於從一個節點到另一個節點的移動
def V_all(action):  # 回傳 current and goal position
    if action == 0: return 0, 1
    if action == 1: return 0, 2
    if action == 2: return 1, 3
    if action == 3: return 1, 4
    if action == 4: return 1, 5
    if action == 5: return 2, 3
    if action == 6: return 2, 4
    if action == 7: return 2, 5
    if action == 8: return 3, 6
    if action == 9: return 3, 7
    if action == 10: return 3, 8
    if action == 11: return 4, 6
    if action == 12: return 4, 7
    if action == 13: return 4, 8
    if action == 14: return 5, 6
    if action == 15: return 5, 7
    if action == 16: return 5, 8
    if action == 17: return 6, 9
    if action == 18: return 6, 10
    if action == 19: return 7, 9
    if action == 20: return 7, 10
    if action == 21: return 8, 9
    if action == 22: return 8, 10
    if action == 23: return 9, 11
    if action == 24: return 10, 11

#定義工作時間函數
#  根據目標機台編號target，返回該機台的加工時間T_work。
#  功能：定義了不同機台的工作時間，用於計算事件發生的時間
def work_time(target):  #各機台加工時間(自行調整)
    if target == 1:
        T_work = 30
    elif target == 2:
        T_work = 30
    elif target == 3:
        T_work = 40
    elif target == 4:
        T_work = 40
    elif target == 5:
        T_work = 40
    elif target == 6:
        T_work = 50
    elif target == 7:
        T_work = 50
    elif target == 8:
        T_work = 50
    elif target == 9:
        T_work = 40
    elif target == 10:
        T_work = 40
    elif target == 11:
        T_work = 0
    return T_work

#定義環境類別 Environment
class Environment: 
    def __init__(self, seed,car_num,machine_num): #初始化環境的基本參數和狀態
        random.seed(seed)
        self.seed = seed #seed：隨機數種子，用於確保結果的可重現性
        self.car_num = car_num #car_num：車輛數量
        self.machine_num = machine_num #machine_num：機台數量
        self.car_state = [] #車輛的狀態（0表示閒置，1表示忙碌）
        self.car_location = [] #車輛的位置（節點編號）
        self.machine_state = [] #Wr 真實的，機台的狀態（0表示閒置，1表示忙碌）
        self.machine_buffer = [] #Br 機台的緩衝區數量
        self.machine_s = [] #判斷action_set用的 判斷動作集合和管理機台狀態
        self.machine_b = [] #判斷動作集合和管理機台狀態
        self.generation_rate = 50 #第一個機台生成率(自行調整) 機台的生成率，用於控制物品的生成速度
        self.event_list = pd.DataFrame(columns=['Time', 'type', 'car', 'cur', 'target', 'action', 'location'])
        #事件列表，用於記錄和管理各種事件（如出發、到達、卸貨等）

#定義事件生成和重置方法
    def generate(self): #增加第一個機台（索引0）的緩衝區數量，模擬物品的生成
        self.machine_b[0] += 1
        self.machine_buffer[0] += 1

    def reset(self): #重置環境狀態，初始化所有車輛和機台的狀態為閒置
        "定義list"

        # self.event_list = pd.DataFrame(columns=['Time', 'type', 'car', 'cur', 'target', 'action', 'location'])
        # self.event_list = self.event_list.append([{'Time': 0, 'type': 1, 'car': 1, 'cur': 7, 'target': 1, 'action': 1}],ignore_index=True)

        for car in range(self.car_num):
            self.car_state.append(0)
            self.car_location.append(0) #設起始位置均為0
        for machine in range(self.machine_num):
            self.machine_state.append(0)
            self.machine_buffer.append(0)
            self.machine_s.append(0)
            self.machine_b.append(0)
        # self.machine_buffer[0]=1
        # self.machine_b[0]=1
        #初始化事件列表，添加一個初始的「finish」事件
        self.event_list = pd.concat([self.event_list,pd.DataFrame([{'Time': 0, 'type': "finish", 'car': None, 'cur': None,
                                                   'target': None, 'action': None, 'location': 0}])],
                                                 ignore_index=True)
        self.add_event() #用add_event方法生成初始事件，並對事件列表按時間排序和重置索引
        self.event_list = self.event_list.sort_values(axis=0, ascending=True, by=['Time'])  # 對時間做排序
        self.event_list = self.event_list.reset_index()  # 調整index
        self.event_list = self.event_list.drop('index', axis=1)  # 把多於的刪除

#定義座標轉換和距離計算方法
    "location 座標轉換方便計算 Manhattan distance"
    "可自行定義機台座標"
    def Trans_cor(self,location): #將機台的位置編號location轉換為實際的座標，便於計算曼哈頓距離
        if location == 0 : return (3,6)
        if location == 1 : return (2,5)
        if location == 2 : return (4,5)
        if location == 3 : return (1,4)
        if location == 4 : return (3,4)
        if location == 5 : return (5,4)
        if location == 6 : return (1,3)
        if location == 7 : return (3,3)
        if location == 8 : return (5,3)
        if location == 9 : return (2,2)
        if location == 10: return (4,2)
        if location == 11: return (3,1)

    "Manhattan distance"
    "可自行定義機台距離(移動時間)"
    #計算兩個機台之間的曼哈頓距離，並乘以5（可能表示移動時間的轉換比例）
    #曼哈頓距離：兩點在一個網格上的距離，計算方法是各坐標軸差值的絕對值之和
    def Distance(self,start, end):
        return sum(map(lambda i, j: abs(i - j), self.Trans_cor(start), self.Trans_cor(end)))*5

#定義動作集合生成方法
    "建立action_set"
    "action定義:  (0)---->(1,2) ,(1,2)---->(3,4,5) ,(3,4,5)---->(6,7,8) ,(6,7,8)---->(9)"
    def actionset(self):  #生成當前可行動集合action_set，即可以選擇的動作
        action_set = []
#         if env.event_list.Time[0] > 345600: #在第四天時，機台1永遠忙碌(故障)  [此為考慮機台故障實驗，無須理會]
#             self.machine_s[4] = 1
#             self.machine_s[6] = 1
            
        for action in range(0, 25):  # 檢查可做工單，B[]andW[]預先改變避免重複工單
            if self.machine_b[V_all(action)[0]] >= 1 and V_all(action)[1] == 11:
            #起點機台的緩衝區數量machine_b大於等於1，且目標機台為11
            #起點機台的緩衝區數量大於等於1，且目標機台處於閒置狀態machine_s為0，且目標機台的緩衝區數量machine_b小於2
                action_set.append(action)
            elif self.machine_b[V_all(action)[0]] >= 1 and self.machine_s[V_all(action)[1]] == 0 and self.machine_b[V_all(action)[1]] < 2: #起點機台B>=1 & 終點機台B<2 & 終點機台W=0，V_all函數定義的動作編號，確定起點和終點機台
                action_set.append(action)
        return action_set

#定義選擇動作及車輛的函數
    "選擇action及car"
    def choose_action(self,action_set): #擇最佳動作和對應的車輛

       #  1. 複製機台緩衝區和閒置狀態
        w_buffer = copy.deepcopy(self.machine_buffer[0:12]) #複製當前的機台緩衝區數量
        w_idle = copy.deepcopy(self.machine_state[0:12]) #複製當前的機台閒置狀態

        if w_buffer[0] > 2:
            w_buffer[0] = 2
        w_buffer[11] = 0  #限制w_buffer[0]不超過2，並將w_buffer[11]設為0，根據業務邏輯調整緩衝區數量

        #  2. 構建鄰接矩陣:根據可行動集合action_set，構建當前的鄰接矩陣adjacency_w
        #Input edges 12*12
        adjacency_w=[]    
        temp=np.zeros((12,12))
        for i in range(len(action_set)):
            num=int(action_set[i])
            nodes=connect(num)
            temp[nodes[0]][nodes[1]]=1
        adjacency_w.append(temp)        

        #  3. 構建節點座標資訊
        #Input nodes 3*12
        coord_w=[] #每個節點的緩衝區數量、閒置狀態和車輛位置標識
        car_loc=[0 for k in range(12)]
        for i in range(12):
            coord_w.append([w_buffer[i],w_idle[i],car_loc[i]]) 

        original_coord=copy.deepcopy(coord_w) #保存原始的節點座標資訊

        #  4. 計算每輛車的Q值
        cars_Q=[]
        cars_action=[]

        for i in range(len(self.car_state)): #判斷car是否閒置  0:idle
            coord_w = copy.deepcopy(original_coord)
            if self.car_state[i]==0:  #遍歷所有車輛，檢查其狀態是否閒置（car_state[i] == 0
                coord_w[self.car_location[i]][2] = 1
                #對於閒置的車輛，更新其位置標識並調用test_one_data_w函數，獲取選擇的動作和對應的Q值
                car_q=test_one_data_w(coord_w,adjacency_w,action_set) #GCN選擇action
                cars_Q.append(car_q[1])
                cars_action.append(car_q[0])
            else:
                cars_Q.append(-1000) #忙碌的car，Q給-1000濾掉
                cars_action.append(-1000)

        #  5. 選擇最佳車輛和動作
        car_num=cars_Q.index(max(cars_Q)) #找出最大Q值對應的車輛car_num和動作action_index
        action_index=cars_action[car_num]
        car_locat=self.car_location[car_num] #獲取該車輛的當前位置car_locat

        return action_index,car_num,car_locat #返回選擇的動作編號、車輛編號和車輛位置

    #定義添加事件的方法
    def add_event(self): #根據當前狀態生成新的事件，並更新車輛和機台的狀態
        global train_count

        #  1. 檢查是否有閒置的車輛
        while sum(self.car_state) != self.car_num : #有車閒置
            #  2. 生成可行動集合
            action_set = self.actionset() #獲取當前可行動的動作集合action_set
            if len(action_set) == 0 : #如果action_set為空，則跳出循環
                break
            #  3. 選擇動作和車輛
            if len(action_set) > 0 :
                #WANTRED
                action_set_temp=copy.deepcopy(action_set) #深拷貝action_set並添加到w_action_set_now
                w_action_set_now.append(action_set_temp)
                action, car, car_locat = self.choose_action(action_set)
                # 使用choose_action方法選擇最佳動作action和對應的車輛car及其位置car_locat
                car_s_location.append(car_locat)
                w_action_choose.append(action) #將車輛位置和選擇的動作分別添加到car_s_location和w_action_choose
                state_temp=copy.deepcopy(self.machine_state)
                buffer_temp=copy.deepcopy(self.machine_buffer)
                #深拷貝當前的機台狀態和緩衝區，並添加到w_state_now和w_buffer_now
                w_state_now.append(state_temp)
                w_buffer_now.append(buffer_temp)                
                ##################################!!!!!下方這小段為訓練，若不訓練便全註解
                #  4. 訓練模型（可選）
                if env.event_list.Time[0] > 100:
                #當事件列表中的時間大於100秒且小於604800秒（7天）時，調用Train_One_Data方法進行模型訓練
                    input_state = state_generator(w_buffer_now[train_count], w_state_now[train_count], car_s_location[train_count]) #3*12
                    input_state_next = state_generator(w_buffer_now[train_count+1], w_state_now[train_count+1], car_s_location[train_count+1])
                    if env.event_list.Time[0] < 604800: ####若超過這段時間則不訓練 604800=7天
                        Train_One_Data(input_state, w_action_set_now[train_count], w_action_choose[train_count], input_state_next, w_action_set_now[train_count+1])
                    train_count += 1 #更新訓練計數器train_count
                
                #################################################################
                #  5. 更新車輛和機台狀態
                self.car_state[car] = 1  # 將選中的車輛狀態設為忙碌（self.car_state[car] = 1）
                     #根據選擇的動作action，更新對應機台的緩衝區machine_b和狀態machine_s
                "選擇完action更新環境"  # 類似以前B W 更新環境
                if action == 0:
                    self.machine_b[0] -= 1
                    self.machine_s[1] += 1
                elif action == 1:
                    self.machine_b[0] -= 1
                    self.machine_s[2] += 1
                elif action == 2:
                    self.machine_b[1] -= 1
                    self.machine_s[3] += 1
                elif action == 3:
                    self.machine_b[1] -= 1
                    self.machine_s[4] += 1
                elif action == 4:
                    self.machine_b[1] -= 1
                    self.machine_s[5] += 1
                elif action == 5:
                    self.machine_b[2] -= 1
                    self.machine_s[3] += 1
                elif action == 6:
                    self.machine_b[2] -= 1
                    self.machine_s[4] += 1
                elif action == 7:
                    self.machine_b[2] -= 1
                    self.machine_s[5] += 1
                elif action == 8:
                    self.machine_b[3] -= 1
                    self.machine_s[6] += 1
                elif action == 9:
                    self.machine_b[3] -= 1
                    self.machine_s[7] += 1
                elif action == 10:
                    self.machine_b[3] -= 1
                    self.machine_s[8] += 1
                elif action == 11:
                    self.machine_b[4] -= 1
                    self.machine_s[6] += 1
                elif action == 12:
                    self.machine_b[4] -= 1
                    self.machine_s[7] += 1
                elif action == 13:
                    self.machine_b[4] -= 1
                    self.machine_s[8] += 1
                elif action == 14:
                    self.machine_b[5] -= 1
                    self.machine_s[6] += 1
                elif action == 15:
                    self.machine_b[5] -= 1
                    self.machine_s[7] += 1
                elif action == 16:
                    self.machine_b[5] -= 1
                    self.machine_s[8] += 1
                elif action == 17:
                    self.machine_b[6] -= 1
                    self.machine_s[9] += 1
                elif action == 18:
                    self.machine_b[6] -= 1
                    self.machine_s[10] += 1
                elif action == 19:
                    self.machine_b[7] -= 1
                    self.machine_s[9] += 1
                elif action == 20:
                    self.machine_b[7] -= 1
                    self.machine_s[10] += 1
                elif action == 21:
                    self.machine_b[8] -= 1
                    self.machine_s[9] += 1
                elif action == 22:
                    self.machine_b[8] -= 1
                    self.machine_s[10] += 1
                elif action == 23:
                    self.machine_b[9] -= 1
                    self.machine_s[11] += 1
                elif action == 24:
                    self.machine_b[10] -= 1
                    self.machine_s[11] += 1
               #  6. 添加出發事件
               #     將一個新的「departure」事件添加到事件列表event_list，記錄車輛出發的信息
                self.event_list = pd.concat([self.event_list,pd.DataFrame(
                    [{'Time': self.event_list.Time[0],
                      'type': "departure",
                      'car': car,
                      'cur': self.car_location[car],
                      'target': V_all(action)[0],
                      'action': action}])],
                ignore_index=True)

#定義完成事件的方法
    "貨物加工完成"
    def finish(self): #功能：處理機台完成加工的事件，更新機台狀態和緩衝區數量
        #  1. 更新機台狀態和緩衝區
        #當貨物完成 Br B Wr W 一起更新

        self.machine_state[self.event_list.location[0]] = 0 #將機台狀態設為閒置
        self.machine_buffer[self.event_list.location[0]] += 1 #增加機台的緩衝區數量
        self.machine_s[self.event_list.location[0]] = 0           #重設機台的閒置狀態
        self.machine_b[self.event_list.location[0]] += 1         #增加機台的緩衝區數量 機台buffer數量+1
        #  2. 生成新的事件
        self.add_event()
        #  3.移除已處理的事件
        self.event_list = self.event_list.drop([0])

#定義出發事件的方法
#  departure功能：處理車輛出發的事件，生成到達載貨地點的「loading」事件
    #  1. 計算到達時間：根據車輛當前位置和目標位置，使用Distance方法計算移動所需的時間，並加到當前事件的時間上
    def departure(self):

        self.event_list=pd.concat([self.event_list,pd.DataFrame([{'Time': self.event_list.Time[0]+self.Distance(self.car_location[self.event_list.car[0]],V_all(self.event_list.action[0])[0]),
                                                 'type': "loading",
                                                 'car': self.event_list.car[0],
                                                 'cur': self.car_location[self.event_list.car[0]], #car_location[i] 從車子位置出發到載貨地點
                                                 'target': V_all(self.event_list.action[0])[0], 'action': self.event_list.action[0]}])],
                                                  ignore_index=True)
        #  2. 添加「loading」事件:在事件列表中添加一個新的「loading」事件，記錄車輛的載貨信息
        #  3. 移除已處理的事件：從事件列表event_list中刪除已處理的事件（索引0）
        self.event_list=self.event_list.drop([0])

#定義裝載事件的方法:loading:處理車輛的裝載事件，生成卸貨的「unload」事件
    #  1. 計算卸貨時間：根據從載貨地點到目標地點的距離，使用Distance方法計算移動所需的時間，並加到當前事件的時間上
    #  2. 添加「unload」事件：在事件列表中添加一個新的「unload」事件，記錄車輛的卸貨信息
    def loading(self):
        self.event_list = pd.concat([self.event_list,pd.DataFrame([{'Time': self.event_list.Time[0]+self.Distance(V_all(self.event_list.action[0])[0],V_all(self.event_list.action[0])[1]),
                                                   'type': "unload",
                                                   'car': self.event_list.car[0],'cur': V_all(self.event_list.action[0])[0],# car_location[i] 從車子位置出發到載貨地點
                                                   'target': V_all(self.event_list.action[0])[1],'action': self.event_list.action[0]}])],
                                                 ignore_index=True)
        self.car_location[self.event_list.car[0]]=V_all(self.event_list.action[0])[0] #更新車輛當前位置
        #  3. 更新車輛和機台狀態:更新車輛的位置為目標地點，減少載貨地點機台的緩衝區數量（machine_buffer）
        self.machine_buffer[V_all(self.event_list.action[0])[0]]-=1 #確實再到貨物 Br -=1
        #  4. 移除已處理的事件
        self.event_list = self.event_list.drop([0]) 

#定義卸貨事件的方法:unload:處理卸貨完成的事件，更新車輛和機台的狀態，並生成新的「finish」事件
    def unload(self):
        #  1. 更新車輛狀態和位置
        self.car_state[self.event_list.car[0]] = 0 #將車輛狀態設為閒置
        self.car_location[self.event_list.car[0]]=self.event_list.target[0] #更新車輛的位置為目標地點
        #  2. 更新機台狀態
        self.machine_state[self.event_list.target[0]]+=1 #增加目標機台的工作狀態
        #  3. 添加「finish」事件:計算加工完成的時間（使用work_time方法）並添加到事件列表中
        self.event_list = pd.concat([self.event_list,pd.DataFrame([{'Time': self.event_list.Time[0] + work_time(self.event_list.target[0]), 'type': "finish",
                                                   'car': None,
                                                   'cur': None,
                                                   'target': None,
                                                   'action': None,
                                                   'location':self.event_list.target[0]}])],
                                                   ignore_index=True)
        #  4. 移除已處理的事件
        self.event_list = self.event_list.drop([0])
```
