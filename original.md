# AGN 新環境互動式
```
notebook_mode = True 
viz_mode = True
```
# Import
```
import os
import json
import argparse
import time

import numpy as np
import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
from sklearn.utils.class_weight import compute_class_weight

from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar

# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *
```
```
if notebook_mode == True:
    %load_ext autoreload
    %autoreload 2
    %matplotlib inline
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('png')
```
# Load configurations (讀取超參數設定)
```
if notebook_mode==False:
    parser = argparse.ArgumentParser(description='gcn_tsp_parser')
    parser.add_argument('-c','--config', type=str, default="configs/default.json")
    args = parser.parse_args()
    config_path = args.config
elif viz_mode == True:
    config_path = "logs/tsp10/config.json"
else:
    config_path = "configs/default.json"

config = get_config(config_path)
print("Loaded {}:\n{}".format(config_path, config))
```
```
# Over-ride config params (for viz_mode)
if viz_mode==True:
    config.gpu_id = "0"
    config.batch_size = 1
    config.accumulation_steps = 1
    config.beam_size = 1280
    
    # Uncomment below to evaluate generalization to variable sizes in viz_mode
#     config.num_nodes = 50
#     config.num_neighbors = 20
#     config.train_filepath = f"./data/tsp{config.num_nodes}_train_concorde.txt"
#     config.val_filepath = f"./data/tsp{config.num_nodes}_val_concorde.txt"
#     config.test_filepath = f"./data/tsp{config.num_nodes}_test_concorde.txt"
```
# Configure GPU options
```
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id) 
```
```
if torch.cuda.is_available():
    print("CUDA available, using GPU ID {}".format(config.gpu_id))
    
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)
```
# Test data loading (測試讀取一筆資料)
```
if notebook_mode:
    num_nodes = config.num_nodes
    num_neighbors = config.num_neighbors
    batch_size = config.batch_size
    #train_filepath = config.train_filepath
    train_filepath = "./data/tsp10_train_concorde_data1.txt"
    dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size, train_filepath)
    print("Number of batches of size {}: {}".format(batch_size, dataset.max_iter))

    t = time.time()
    batch = next(iter(dataset))  # Generate a batch of TSPs
    print("Batch generation took: {:.3f} sec".format(time.time() - t))
    print(batch)
    print("edges:", batch.edges.shape)
    print("edges_values:", batch.edges_values.shape)
    print("edges_targets:", batch.edges_target.shape)
    print("nodes:", batch.nodes.shape)
    print("nodes_target:", batch.nodes_target.shape)
    print("nodes_coord:", batch.nodes_coord.shape)
    print("tour_nodes:", batch.tour_nodes.shape)
    print("tour_len:", batch.tour_len.shape)
    print("ET",batch.edges_target)
    #edges：圖的邊的資訊（即節點之間的連接）
    #edges_values：邊的權重或距離值。
    #edges_targets：目標邊，表示解決 TSP 時最優解中哪條邊應該被選擇。
    #nodes：圖中的節點。
    #nodes_target：節點的目標標籤（根據 TSP 最優解）。
    #nodes_coord：節點的座標（可能代表城市的地理位置）。
    #tour_nodes：旅行推銷員的完整行程路線（節點順序）。
    #tour_len：行程的總長度。
    
    #繪製 TSP 圖像
    idx = 0
    f = plt.figure(figsize=(5, 5))
    a = f.add_subplot(111)
    plot_tsp(a, batch.nodes_coord[idx], batch.edges[idx], batch.edges_values[idx], batch.edges_target[idx])
```
# Instantiate model (建立模組)
```
if notebook_mode == True:
    # Instantiate the network
    net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        net.cuda()
    print(net)

    # Compute number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)
    # Define optimizer
    learning_rate = config.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(optimizer)
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
```
```
Node=[0,1,2,3,4,5,6,7,8,9,10,11]
n = 12
batch.edges_values = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]
#batch.edges_values=np.zeros((12,12))
buffer=[2, 1, 2, 0, 1, 2, 1, 1, 2, 0, 2, 0]#buffer這裡自訂Node1=2，Node12=0
idle=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]#閒置idle這裡自訂Node12=2
car=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(Node)):
    for j in range(len(Node)):
        if i==j:
            batch.edges_values[i][j]=0
        else:
            first=Node[i]
            second=Node[j]
            batch.edges_values[i][j]=DS(first,second)
batch.edges_values=[batch.edges_values]
#print(batch.edges_values)
batch.nodes_coord=[]

for i in range(len(Node)):
    temp=[]
    temp.insert(0,buffer[i])
    temp.insert(1,idle[i])
    temp.insert(2,car[i])
    batch.nodes_coord.append(temp)
batch.nodes_coord=[batch.nodes_coord] 
print(batch.nodes_coord)
batch.edges=[[0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
batch.edges=[batch.edges]
batch.nodes=[[1,1,1,1,1,1,1,1,1,1,1,1]]
batch.edges_target=[[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
#batch.edges_target=[[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
batch.edges_target=[batch.edges_target]
#Input variables
x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
##

#print(batch)
# Compute class weights
edge_labels = y_edges.cpu().numpy().flatten()
edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

y_preds, q = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
#loss = loss.mean()
print("Output size: {}".format(y_preds.size()))
#x_edges:指示函數，x_edges_value:distance matrix，
```
```
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
def Find_reward(state,actionset,action):

    reward=[[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    
    return reward
```
# TEST ONE DATA (TEST Function)
```
global edges_value
Node=[0,1,2,3,4,5,6,7,8,9,10,11]
n=12
batch.edges_values = [[[0 for k in range(n)] for j in range(n)] for i in range(n)] #Distance matrix固定不變

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
def test_one_data_w(coord,adjacency_w,action_s):
    global edges_value
    # 資料前處理 (Encoding)
    batch.edges=[[0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
    #batch.edges=[[2,1,1,0,0,0,0,0,0,0,0,0],[0,2,0,1,1,1,0,0,0,0,0,0],[0,0,2,1,1,1,0,0,0,0,0,0],[0,0,0,2,0,0,1,1,1,0,0,0],[0,0,0,0,2,0,1,1,1,0,0,0],[0,0,0,0,0,2,1,1,1,0,0,0],[0,0,0,0,0,0,2,0,0,1,1,0],[0,0,0,0,0,0,0,2,0,1,1,0],[0,0,0,0,0,0,0,0,2,1,1,0],[0,0,0,0,0,0,0,0,0,2,0,1],[0,0,0,0,0,0,0,0,0,0,2,1],[0,0,0,0,0,0,0,0,0,0,0,2]]
    batch.edges=[batch.edges] #指示函數不變
    #batch.edges=adjacency_w  #目前的state_input_edges
    batch.edges_values=[edges_value] #Distance matrix
    batch.nodes=[[1,1,1,1,1,1,1,1,1,1,1,1]]
    batch.nodes_coord=[coord] #目前的state_input_nodes
    batch.edges_target=[adjacency_w]#目前的state_input_edges


    x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
    x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
    x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

    # Forward pass (輸入網路架構，輸出Q值)
    edge_cw=1
    y_preds, q = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)

    q=q.reshape(12,12)
    max_num=torch.argmax(q)
    max_num=max_num.item()
    
    #Filter (只採用action_set的Q值)
    in_action_set=[]
    for i in range(len(action_s)):
        position=connect(action_s[i])
        in_action_set.append(q[position[0]][position[1]])

    Max_q=max(in_action_set)
    Max_choose=action_s[in_action_set.index(max(in_action_set))]    

    return [Max_choose,Max_q]
```
# Train One Data (Train Function)
```
def Train_One_Data(state_now,action_set_now,choose_action,state_next,action_set_next):
    global edges_value, loss
    edge_cw = None
    if len(action_set_next)==0:#若下一個action_set為空，則不學此筆資料
        return;
    #讀取現在的state
    #重新定義機台0和機台11的值
    if state_now[0][0] > 2: 
        state_now[0][0]=2
    state_now[11][0]=0
    
    #將action_set_now轉成adjacency
    adjacency_now=[]    
    temp=np.zeros((12,12))
    for i in range(len(action_set_now)):
        num=int(action_set_now[i])
        nodes=connect(num)
        temp[nodes[0]][nodes[1]]=1
    adjacency_now.append(temp)
    
    #NEXT STATE
    #重新定義機台0和機台11的值
    if state_next[0][0] > 2: 
        state_next[0][0]=2
    state_next[11][0]=0

    #將action_set_next轉成adjacency
    adjacency_next=[]    
    temp=np.zeros((12,12))
    for i in range(len(action_set_next)):
        num=int(action_set_next[i])
        nodes=connect(num)
        temp[nodes[0]][nodes[1]]=1
    adjacency_next.append(temp)
##################Input Now #######################################################################################
    batch.edges=[[0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0]]
    batch.edges=[batch.edges] #指示函數不變
    #batch.edges=adjacency_now
    batch.edges_values=[edges_value] #Distance matrix
    batch.nodes=[[1,1,1,1,1,1,1,1,1,1,1,1]]
    batch.nodes_coord=[state_now] #目前的state_input_nodes
    batch.edges_target=[adjacency_now]#目前的state_input_edges
####################AGN############AGN###############AGN##################AGN#############AGN##################AGN#############
    # Convert batch to torch Variables
    x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
    x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
    x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
    # Compute class weights (if uncomputed)
    if type(edge_cw) != torch.Tensor:
        edge_labels = y_edges.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
    # Forward pass
    y_preds, q = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
############################################################################################################################    
#Next State Data
    #batch.edges=adjacency_next
    batch.nodes_coord=[state_next] #下一筆的state_input_nodes
    batch.edges_target=[adjacency_next]#下一筆的state_input_edges

    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    y_preds2, q2 = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw) #下一筆的 Q
    q2_original=q2.clone()
    q2=q2.reshape(12,12)
    q2_original=q2_original.reshape(12,12)

    #Filter 找出q2的 a'(q2max_choose)
    in_action_set=[]
    for i in range(len(action_set_next)):
        position=connect(action_set_next[i])
        in_action_set.append(q2[position[0]][position[1]])

    q2max_choose=action_set_next[in_action_set.index(max(in_action_set))] 
    
    reward2=Find_reward(state_next,action_set_next,q2max_choose) #根據目前State找出Reward
    
    #print("Q2",q2)


    #找出Q2的max值            
    max_num=torch.argmax(q2)
    max_num=max_num.item()
    max_q2_values=q2[max_num//12][max_num%12].item()
    
####Update Q#############Update Q############Update Q#############Update Q############Update Q##################Update Q###
    original_q=q.clone()
    original_q=original_q.reshape(12,12) #12*12  

    reward1=Find_reward(state_now,action_set_next,choose_action) #根據目前State找出Reward

    target_q=q.clone().detach()

    target_q=target_q.reshape(12,12) #12*12
    #print("target_q:",target_q)
    #print("action_choose:",action_choose[batch_num+epoch* batches_per_epoch])
    #print("action_choose:",action_choose[1377])
    action_ij=connect(choose_action) #找出action_ij

    #print("action_ij",action_ij)
    target_q_values=reward1[action_ij[0]][action_ij[1]]+ 0.9 * max_q2_values
    target_q[action_ij[0]][action_ij[1]] = target_q_values
###################################################################################################################################
#Compute Loss and backward net
    #print("QQ",original_q)
    loss = F.smooth_l1_loss(original_q, target_q)
    loss = loss.mean()
    Loss_plt.append(loss.item())
    optimizer.zero_grad()
    #print("loss:",loss)
    loss.backward()
    optimizer.step()
```
# 畫圖Initial
```
import pygame as pg
import pandas as pd
#定義顏色
white  = (255, 255, 255)
black  = (  0,   0,   0)
red    = (255,   0,   0)
green  = (  0, 255,   0)
blue   = (  0,   0, 255)
yellow = (255, 255,   0)
pink = (186,217,232)
#畫布字型初始
pg.init()
clock = pg.time.Clock() #TIME CLOCK
pg.display.set_caption("env") #標題名
screen = pg.display.set_mode((1280,900)) #設定視窗
bg = pg.Surface(screen.get_size()).convert() #建立畫布bg
#bg =pg.image.load("girl.jpg")
bg.fill(pink)
font = pg.font.SysFont("simhei", 24)#字樣
font1 = pg.font.SysFont("simhei", 40)#字樣
#定義暫存區圖樣
def rect(bg, color, x, y):
    pg.draw.rect(bg, color,[x, y, 100, 100], 0)
    pg.draw.rect(bg, (0,0,0),[x, y, 100, 100], 2)
    pg.draw.line(bg, (0,0,0),(x,y+50), (x+100, y+50), 3)
    pg.draw.line(bg, (0,0,0),(x+50,y), (x+50, y+100), 3)
def car(bg, color, x, y):
    pg.draw.rect(bg, color,[x, y, 25, 25], 0)
#畫暫存區 (背景)
text1 = font.render("Begin", True, (0,0,255), pink)
bg.blit(text1, (100,80))

text10 = font.render("Goal", True, (0,0,255), pink)
bg.blit(text10, (600,730))
#V1
truck1 = pg.image.load("car.png")
truck1 = pg.transform.scale(truck1,(100,100))
truck2 = pg.image.load("car.png")
truck2 = pg.transform.scale(truck2,(100,100))
truck3 = pg.image.load("car.png")
truck3 = pg.transform.scale(truck3,(100,100))
textA = font1.render("A", True, (0,0,255), pink)
textB = font1.render("B", True, (0,0,255), pink)
textC = font1.render("C", True, (0,0,255), pink)
#pg.draw.circle(truck1, (255,0,0), (25,25), 20, 0)  
truck1.blit(textA, (10,5))
truck2.blit(textB, (30,5))
truck3.blit(textC, (50,5))

rect = truck1.get_rect()         #取得球矩形區塊
rect = truck2.get_rect()         #取得球矩形區塊
rect = truck3.get_rect()         #取得球矩形區塊
rect.center = (50,150)        #球起始位置
x, y = rect.topleft            #球左上角坐標
def env_info(Time_info):
    Time_info = str(Time_info)
    text0 = font1.render("Time:"+Time_info+'s', True, (0,0,255), pink)
    bg.blit(text0, (10,10))

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
def update_number(x,y,a):
    if   a==0 : #blue
        pg.draw.rect(bg,black,[x-2, y-2, 84, 84], 0)
        pg.draw.rect(bg,green,[x, y, 80, 80], 0)
        text1 = font1.render("0", True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
    elif a==1 : #yellow
        pg.draw.rect(bg,yellow,[x, y, 80, 80], 0)
        text1 = font1.render("1", True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
    elif a==2 : #green
        pg.draw.rect(bg,red,[x, y, 80, 80], 0)
        text1 = font1.render("2", True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
    elif a>=3 : # red
        pg.draw.rect(bg,red,[x, y, 80, 80], 0)
        a=str(a)
        text1 = font1.render(a, True, (0,0,255), (255,255,255))
        bg.blit(text1, (x+30,y+28))
def update_work(x,y,a):
    if   a==0 : #blue
        pg.draw.rect(bg,black,[x-2, y-2, 44, 44], 0)
        pg.draw.rect(bg,green,[x, y, 40, 40], 0)
    elif a==1 : #yellow
        pg.draw.rect(bg,red,[x, y, 40, 40], 0)

def output_bg(BG_B,BG_S):
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
    pg.display.update()
```
```
import random
import numpy as np
import pandas as pd
from random import choice
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
loss=0
Loss_plt=[]
day_list=[]
w_state_now=[]
w_buffer_now=[]
w_action_set_now=[]
w_action_choose=[]
car_s_location=[]
train_count=0
###################################讀取最佳 Check Point Model  (若要讀取最佳訓練成果，請取消註記，並註記掉下方Train函式)
# net=torch.load('Best_Model.pt')
# net.eval()
########################################################
def state_generator(buf, idle, loc): #資料前處理 (將當前Buffer、 Idel、 車位置，轉換成State)
    input_state=[]
    car_loc=[0 for k in range(12)]
    car_loc[loc] = 1
    for i in range(12):
        input_state.append([buf[i],idle[i],car_loc[i]]) 

    return input_state


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
class Environment: 
    def __init__(self, seed,car_num,machine_num):
        random.seed(seed)
        self.seed = seed
        self.car_num = car_num
        self.machine_num = machine_num
        self.car_state = []
        self.car_location = []
        self.machine_state = [] #Wr 真實的
        self.machine_buffer = [] #Br
        self.machine_s = [] #判斷action_set用的
        self.machine_b = []
        self.generation_rate = 50 #第一個機台生成率(自行調整)
        self.event_list = pd.DataFrame(columns=['Time', 'type', 'car', 'cur', 'target', 'action', 'location'])


    def generate(self):
        self.machine_b[0] += 1
        self.machine_buffer[0] += 1

    def reset(self):
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
        self.event_list = pd.concat([self.event_list,pd.DataFrame([{'Time': 0, 'type': "finish", 'car': None, 'cur': None,
                                                   'target': None, 'action': None, 'location': 0}])],
                                                 ignore_index=True)
        self.add_event()
        self.event_list = self.event_list.sort_values(axis=0, ascending=True, by=['Time'])  # 對時間做排序
        self.event_list = self.event_list.reset_index()  # 調整index
        self.event_list = self.event_list.drop('index', axis=1)  # 把多於的刪除

    "location 座標轉換方便計算 Manhattan distance"
    "可自行定義機台座標"
    def Trans_cor(self,location):
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
    def Distance(self,start, end):
        return sum(map(lambda i, j: abs(i - j), self.Trans_cor(start), self.Trans_cor(end)))*5

    "建立action_set"
    "action定義:  (0)---->(1,2) ,(1,2)---->(3,4,5) ,(3,4,5)---->(6,7,8) ,(6,7,8)---->(9)"
    def actionset(self): 
        action_set = []
#         if env.event_list.Time[0] > 345600: #在第四天時，機台1永遠忙碌(故障)  [此為考慮機台故障實驗，無須理會]
#             self.machine_s[4] = 1
#             self.machine_s[6] = 1
            
        for action in range(0, 25):  # 檢查可做工單，B[]andW[]預先改變避免重複工單
            if self.machine_b[V_all(action)[0]] >= 1 and V_all(action)[1] == 11:
                action_set.append(action)
            elif self.machine_b[V_all(action)[0]] >= 1 and self.machine_s[V_all(action)[1]] == 0 and self.machine_b[V_all(action)[1]] < 2: #起點機台B>=1 & 終點機台B<2 & 終點機台W=0
                action_set.append(action)
        return action_set

    "選擇action及car"
    def choose_action(self,action_set):


        w_buffer = copy.deepcopy(self.machine_buffer[0:12])
        w_idle = copy.deepcopy(self.machine_state[0:12])

        if w_buffer[0] > 2:
            w_buffer[0] = 2
        w_buffer[11] = 0

        #Input edges 12*12
        adjacency_w=[]    
        temp=np.zeros((12,12))
        for i in range(len(action_set)):
            num=int(action_set[i])
            nodes=connect(num)
            temp[nodes[0]][nodes[1]]=1
        adjacency_w.append(temp)        
        
        #Input nodes 3*12
        coord_w=[]
        car_loc=[0 for k in range(12)]
        for i in range(12):
            coord_w.append([w_buffer[i],w_idle[i],car_loc[i]]) 

        original_coord=copy.deepcopy(coord_w)
        cars_Q=[]
        cars_action=[]

        for i in range(len(self.car_state)): #判斷car是否閒置  0:idle
            coord_w = copy.deepcopy(original_coord)
            if self.car_state[i]==0:
                coord_w[self.car_location[i]][2] = 1 
                car_q=test_one_data_w(coord_w,adjacency_w,action_set) #GCN選擇action
                cars_Q.append(car_q[1])
                cars_action.append(car_q[0])
            else:
                cars_Q.append(-1000) #忙碌的car，Q給-1000濾掉
                cars_action.append(-1000)
        car_num=cars_Q.index(max(cars_Q))
        action_index=cars_action[car_num]
        car_locat=self.car_location[car_num]

        return action_index,car_num,car_locat

    
    def add_event(self):
        global train_count

        while sum(self.car_state) != self.car_num : #有車閒置
            action_set = self.actionset()
            if len(action_set) == 0 :
                break
            if len(action_set) > 0 :
                #WANTRED
                action_set_temp=copy.deepcopy(action_set)
                w_action_set_now.append(action_set_temp)
                action, car, car_locat = self.choose_action(action_set)  # 挑懸action及car 
                car_s_location.append(car_locat)
                w_action_choose.append(action)
                state_temp=copy.deepcopy(self.machine_state)
                buffer_temp=copy.deepcopy(self.machine_buffer)
                w_state_now.append(state_temp)
                w_buffer_now.append(buffer_temp)                
                ##################################!!!!!下方這小段為訓練，若不訓練便全註解
                if env.event_list.Time[0] > 100:
                    input_state = state_generator(w_buffer_now[train_count], w_state_now[train_count], car_s_location[train_count]) #3*12
                    input_state_next = state_generator(w_buffer_now[train_count+1], w_state_now[train_count+1], car_s_location[train_count+1])
                    if env.event_list.Time[0] < 604800: ####若超過這段時間則不訓練 604800=7天
                        Train_One_Data(input_state, w_action_set_now[train_count], w_action_choose[train_count], input_state_next, w_action_set_now[train_count+1])
                    train_count += 1
                
                #################################################################
                self.car_state[car] = 1  # 挑到輛要變busy
              
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
                self.event_list = pd.concat([self.event_list,pd.DataFrame(
                    [{'Time': self.event_list.Time[0],
                      'type': "departure",
                      'car': car,
                      'cur': self.car_location[car],
                      'target': V_all(action)[0],
                      'action': action}])],
                ignore_index=True)

    "貨物加工完成"
    def finish(self):
        #當貨物完成 Br B Wr W 一起更新

        self.machine_state[self.event_list.location[0]] = 0
        self.machine_buffer[self.event_list.location[0]] += 1
        self.machine_s[self.event_list.location[0]] = 0           #機台變為閒置
        self.machine_b[self.event_list.location[0]] += 1         #機台buffer數量+1



        self.add_event()

        self.event_list = self.event_list.drop([0])
    def departure(self):

        self.event_list=pd.concat([self.event_list,pd.DataFrame([{'Time': self.event_list.Time[0]+self.Distance(self.car_location[self.event_list.car[0]],V_all(self.event_list.action[0])[0]),
                                                 'type': "loading",
                                                 'car': self.event_list.car[0],
                                                 'cur': self.car_location[self.event_list.car[0]], #car_location[i] 從車子位置出發到載貨地點
                                                 'target': V_all(self.event_list.action[0])[0], 'action': self.event_list.action[0]}])],
                                                  ignore_index=True)
        self.event_list=self.event_list.drop([0])

    def loading(self):
        self.event_list = pd.concat([self.event_list,pd.DataFrame([{'Time': self.event_list.Time[0]+self.Distance(V_all(self.event_list.action[0])[0],V_all(self.event_list.action[0])[1]),
                                                   'type': "unload",
                                                   'car': self.event_list.car[0],'cur': V_all(self.event_list.action[0])[0],# car_location[i] 從車子位置出發到載貨地點
                                                   'target': V_all(self.event_list.action[0])[1],'action': self.event_list.action[0]}])],
                                                 ignore_index=True)
        self.car_location[self.event_list.car[0]]=V_all(self.event_list.action[0])[0] #更新車輛當前位置
        self.machine_buffer[V_all(self.event_list.action[0])[0]]-=1 #確實再到貨物 Br -=1
        self.event_list = self.event_list.drop([0])

    def unload(self):
        self.car_state[self.event_list.car[0]] = 0
        self.car_location[self.event_list.car[0]]=self.event_list.target[0]
        self.machine_state[self.event_list.target[0]]+=1
        self.event_list = pd.concat([self.event_list,pd.DataFrame([{'Time': self.event_list.Time[0] + work_time(self.event_list.target[0]), 'type': "finish",
                                                   'car': None,
                                                   'cur': None,
                                                   'target': None,
                                                   'action': None,
                                                   'location':self.event_list.target[0]}])],
                                                   ignore_index=True)
        self.event_list = self.event_list.drop([0])
```
# 主程式
```
global loss
import time
if __name__ == '__main__':
    
    #f1:departure 車輛出發
    #f2:loading 裝貨               buffer-=1
    #f3:unload 卸貨加工            state+=1
    #f4:finish 加工完成            b,buffer+=1 s,state-=1       如果有挑到action 針對該貨物對 b-=1 s+=1
    env = Environment(40, 3, 12)
    env.reset()
    #print(env.event_list)
    k = day = 0
    total=604800*2 #決定環境執行總時間
    tStart=time.time()
    while env.event_list.Time[0] < total:
        #畫圖
        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit()    
        env.add_event()
        output_bg(env.machine_buffer,env.machine_state)
        env_info(env.event_list.Time[0])
        truckV1_load(env.car_location)
        truckV2_load(env.car_location)
        truckV3_load(env.car_location)
        print('\r' + '[Progress]:[%s%s]%.2f%%;' % (
        '\033[1;32;43m🐵\033[0m' * int(env.event_list.Time[0]*20/total), '  ' * (20-int(env.event_list.Time[0]*20/total)),
        float(env.event_list.Time[0]/total*100))+'loss:%.3f'%loss+' Br:%s'%env.machine_buffer, end='')  
        
        env.add_event()
        
        while env.event_list.Time[0] > (k * env.generation_rate):
            k += 1
            env.generate()
        if env.event_list.type[0] == "departure":
            env.departure()
        elif env.event_list.type[0] == "loading":
            env.loading()
        elif env.event_list.type[0] == "unload":
            env.unload()
        elif env.event_list.type[0] == "finish":
            env.finish()

        env.event_list = env.event_list.sort_values(axis=0, ascending=True, by=['Time'])  # 對時間做排序
        env.event_list = env.event_list.reset_index()  # 調整index
        env.event_list = env.event_list.drop('index', axis=1)  # 把多於的刪除
        
        #每一段時間記錄一次工單量，也會記錄最佳模型
        count_day = env.event_list.Time[0] / 43200  #每43200秒(12小時)記錄一次出貨量
        if count_day > day+1:        
            print(env.machine_buffer[11])
            day_list.append(env.machine_buffer[11])
            if day == 1 :
                print(day_list[day]-day_list[day-1])
                max_day = day_list[day]-day_list[day-1]
                torch.save(net,'Best_Model.pt')
                print("Saved Model..")
            if day > 1 :
                print(day_list[day]-day_list[day-1])
                compare = day_list[day]-day_list[day-1]
                if compare > max_day :
                    max_day = compare
                    torch.save(net,'Best_Model.pt')
                    print("Saved Model..")
            day += 1
        #clock.tick(10) #可調慢程式執行速度 (觀察畫圖用)  
    pg.quit()        

tEnd=time.time()
#torch.save(net,'Best_Model.pt')
print("Spent Time :",tEnd-tStart) #程式總執行時間(s)
```
```
DQN_result=0
# for i in range(len(action_choose)):
#     if action_choose[i]==24 or action_choose[i]==25:
#         DQN_result=DQN_result+1
gcn_result=0
for i in range(len(w_action_choose)):
    if w_action_choose[i]==23 or w_action_choose[i]==24:
        gcn_result=gcn_result+1

print("傳統演算法_result",DQN_result)
print("gcn_result",gcn_result)#5100 #5411  ##ATTENTION 5187 5328 #ATTENTION with/d_k 5589
print("BR",env.machine_buffer)
```
```
print(day_list)
for i in range(len(day_list)-1):
    print(day_list[i+1]-day_list[i])
```
# PLOT
```
#繪製損失隨著回合數變化的曲線，並將損失數據保存到文本文件，以便未來分析和比較: DQN 和 GCN
plt.plot(np.array(Loss_plt), c='purple', label='Loss')
plt.legend(loc='best')
plt.ylabel('Loss')
plt.xlabel('episodes')
plt.axis([0, 70000, 0, 100])
np.savetxt('Loss.txt',Loss_plt,fmt="%f" )
plt.grid()
plt.show()
```
