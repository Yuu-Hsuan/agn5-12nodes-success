# 導入所需的庫
1. `torch` 和 `torch.nn` 是 `PyTorch` 庫，用於構建和訓練神經網絡。
2. `torch.nn.functional` 包含各種神經網絡操作（例如激活函數 `relu`）。
3. `numpy` 是用於數值計算的庫，方便處理數據陣列（在此程式碼中未直接使用）
```
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
```
# BatchNormNode 類
1. 定義了一個名為 `BatchNormNode` 的類，繼承自 `nn.Module`，用於批次正規化節點特徵。
2.  `__init__ `方法接收 `hidden_dim`，初始化一個 `nn.BatchNorm1d` 層（只對 1 維進行正規化）
```
class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
```
##forward 方法是前向傳播函數
1.  `x_trans = x.transpose(1, 2).contiguous()`：

    將資料形狀從 (batch_size, num_nodes, hidden_dim) 轉換為 (batch_size, hidden_dim, num_nodes)，以便於進行 BatchNorm1d。
   
2.  `x_trans_bn = self.batch_norm(x_trans)`：

    對轉換後的資料進行批次正規化。
    
3.  `x_bn = x_trans_bn.transpose(1, 2).contiguous()`：

    將資料形狀轉回 (batch_size, num_nodes, hidden_dim)，以保持與輸入形狀一致
    
```
    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn
```
# BatchNormEdge 類
定義了一個名為 `BatchNormEdge` 的類，用於批次正規化邊的特徵，使用 BatchNorm2d
```
class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
```
`forward` 方法中，將輸入形狀轉換為 `(batch_size, hidden_dim, num_nodes, num_nodes)` 以便於 `BatchNorm2d` 處理，之後再轉回原始形狀
```
    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn
```
# NodeFeatures 類
* `NodeFeatures` 類用於更新節點特徵。
* 定義了兩個線性轉換層 `U` 和 `V`，分別對節點特徵進行不同處理。
* 聚合模式（`aggregation`）有 "sum" 和 "mean" 兩種。
```
class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
```
* 將節點特徵 `x` 分別通過線性層 `U` 和 `V`。
* `Vx = Vx.unsqueeze(1)` 將 `Vx` 維度擴展，以便與 `edge_gate` 匹配
```
    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        
        #print("W1",W1.size())
        gateVx = edge_gate * Vx # B x V x V x H
        #print("gateVx",gateVx.size())
        #print(gateVx)
```
根據聚合模式計算 `x_new`，若為 "mean" 模式，會將邊的門值進行平均
```
        if self.aggregation=="mean":
            x_new = Ux + (torch.sum(gateVx, dim=2)) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
            #x_new = (torch.sum(gateVx, dim=2)) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        
        return x_new
```
# EdgeFeatures 類
用於更新邊的特徵，定義了兩個線性層 `U` 和 `V`
```
class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
```
根據邊的特徵 `e` 和節點特徵 `x` 計算卷積後的邊特徵 `e_new`。
```        
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        #print("x:",x.size())
        #print("e:",e.size())
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        Vx = Vx.unsqueeze(2)  # Extend Vx from "B x V x H" to "B x V x 1 x H"  
        e_new = Ue * (Vx + Wx)
        #print("e_new:",e_new.size())
        return e_new
```
# ResidualGatedGCNLayer 類
此類包含節點與邊的卷積層、批次正規化層，以及殘差連接
```
class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)
        #改
        #self.U = nn.Linear(hidden_dim, hidden_dim, True)
```
將邊特徵 `e_tmp` 批次正規化，並通過 softmax 函數計算邊的門值
```
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        #print("E_TEP",e_tmp)
        # Compute edge gates
        e_tmp = self.bn_edge(e_tmp)
        edge_gate = F.softmax(e_tmp,dim=2)
        #edge_gate = F.sigmoid(e_tmp)
        #print("E_gate",edge_gate)
        #print("edge_gate",edge_gate.size())
        # Node convolution
```
* `x_tmp` 是通過 `NodeFeatures` 類卷積後的節點特徵，並經過批次正規化。
* `F.relu(x_tmp)`：將正規化後的結果通過 ReLU 激活。
* `x_new = x_in + x`：加入殘差連接，避免訊息丟失（將原始輸入的`x_in` 與更新後的`x` 相加）。
* `e_new = edge_gate * e_in + e_in`：對邊特徵也加入殘差連接
```
        x_tmp = self.node_feat(x_in,edge_gate)
        
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x
        e_new = edge_gate * e_in + e_in
        return x_new, e_new
```
# MLP 類
* 定義多層感知機（MLP）結構，用於預測輸出。
* `L` 是 MLP 的層數，`U` 是中間層的線性轉換，並用 `nn.ModuleList` 儲存。
* `V` 是最後一層，用於輸出結果
```
class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)
```
* Ux = x 將輸入的特徵傳遞給 Ux。
* 對每一層 U_i，執行線性轉換並使用 ReLU 激活。
* 最終輸出層 V 用於計算預測結果 y。
```
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
```
# MLP_Q 類
`MLP_Q` 類與 `MLP` 類相似，但在最後一層使用 `Conv2d` 卷積層（用於處理二維特徵）
```
class MLP_Q(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP_Q, self).__init__()
        self.L = L
        self.V = nn.Conv2d(hidden_dim, output_dim, 1, 1, 0)
```
`Ux = x.permute(0, 3, 1, 2)`：調整維度順序，使輸入形狀適合卷積層
最終將卷積輸出結果的維度調整回原始順序，以保持一致性
```
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x                #B x V x V x H
       
        Ux=Ux.permute(0,3,1,2) #B x H x V x V
        y = self.V(Ux)  # B x O
        #y = F.relu(y)
        y=y.permute(0,2,3,1)
        return y
```
