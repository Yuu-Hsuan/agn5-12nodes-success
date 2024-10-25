```
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from models.gcn_layers import ResidualGatedGCNLayer, MLP ,MLP_Q
from utils.model_utils import *


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong,sub_sample=False,dimension=2,inter_channels=None,ctx_dim=(1,300),bn_layer=True): 
        #ATTENTION
        # 建立卷積層
        super(ResidualGatedGCNModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(64, 512, 3, stride=1, padding=1)     #Out channels=512

        assert dimension in [1, 2, 3]           # 斷言dimension必定在[1,2,3]裡

        self.dimension = dimension              # 輸入維度
        self.sub_sample = sub_sample            # 是否啟用樣本擷取(maxpool)

        #self.in_channels = in_channels          #1
        in_channels = ctx_dim[1]                               # FIXME:強制給512
        self.in_channels = in_channels
        print("self.in_channels",self.in_channels)
        self.inter_channels = inter_channels    #None

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2          #256
            # 進行壓縮得到channel數
            if self.inter_channels == 0:
                self.inter_channels = 1
        #print("self.inter_channels:", self.inter_channels)

        # 輸入維度決定算圖單元
        if dimension == 3:
            conv_nd = nn.Conv3d                                     # 卷積單元
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))    # pool單元
            bn = nn.BatchNorm3d                                     # Batch Normalize單元
        elif dimension == 2:
            conv_nd = nn.Conv2d                                     # 卷積單元
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))       # pool單元
            bn = nn.BatchNorm2d                                     # Batch Normalize單元
        else:
            conv_nd = nn.Conv1d                                     # 卷積單元
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))          # pool單元
            bn = nn.BatchNorm1d                                     # Batch Normalize單元

        # g()
        self.g = conv_nd(in_channels=self.in_channels,out_channels=self.inter_channels,kernel_size=1,stride=1,padding=0)
        
        
        #print("self.inter_channels",self.inter_channels)
        # Batch Normalize Layer
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))

            # 初始化參數
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

            # 初始化參數
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # theta()
        self.theta = conv_nd(in_channels=self.in_channels,      #1
                             out_channels=self.inter_channels,  #1
                             kernel_size=1,
                             stride=1,
                             padding=0)
        # phi()
        self.phi = conv_nd(in_channels=self.in_channels,        #1
                           out_channels=self.inter_channels,    #1
                           kernel_size=1,
                           stride=1,
                           padding=0)


        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.q_edges_out=config['q_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        self.mlp_edges_Q = MLP_Q(self.hidden_dim, self.q_edges_out, self.mlp_layers)
        #self.F=nn.ReLU()
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss
            # y_nodes: Targets for nodes (batch_size, num_nodes, num_nodes)
            # node_cw: Class weights for nodes loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
        """
        # Node and edge embedding
        dimension=3
        x = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        #print("X",x)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H #Distance Matrix
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        #print("e_vals",e_vals.size())
        #print("e_tags",e_tags.size())
        e = torch.cat((e_vals, e_tags), dim=3)
        #print("E_CAT",e.size())
        #print("FIRST E",e)
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H
            #print("X",x)
            #print("E",e)
        #print("E",e)
        # MLP classifier
        #print("e_size",e.size())
        #y_pred_edges = self.mlp_edges(z)  # B x V x V x voc_edges_out
        #q = self.mlp_edges_Q(z)  # B x V x V x voc_edges_out
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        q = self.mlp_edges_Q(e)  # B x V x V x voc_edges_out
        #print("Q",q)
        # y_pred_nodes = self.mlp_nodes(x)  # B x V x voc_nodes_out
        #print("Y_predict",y_pred_edges)
        #print("y_edges=",y_edges)
        #print("edge_cw=",edge_cw)
        # Compute loss
        edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
        
        #loss = loss_edges(y_pred_edges, y_edges, edge_cw) #cross_entrophy
        
        #print("y_pred_edges_size=",y_pred_edges.size())
        #print("y_edge_size",y_edges.size())
        
        #return y_pred_edges, loss, q
        return y_pred_edges, q
```
