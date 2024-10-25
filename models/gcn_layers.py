import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

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


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

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
        if self.aggregation=="mean":
            x_new = Ux + (torch.sum(gateVx, dim=2)) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
            #x_new = (torch.sum(gateVx, dim=2)) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
        
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
        x_tmp = self.node_feat(x_in,edge_gate)
        # Batch normalization

        # Residual connection
        ##x_new = x_in + x
        
        #print("X_TMP",x_tmp)
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x
#         Ux = self.U(x_in) 
#         x_new = Ux + x
        e_new = edge_gate * e_in + e_in
        #print("X!!!!",x_new)
        #print("E!!!!",e_new)
        #print("e_new!",e_new.size())
        #print("x_new!",x_new.size())
        return x_new, e_new


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

class MLP_Q(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP_Q, self).__init__()
        self.L = L
#         U = []
#         for layer in range(self.L - 1):
#             U.append(nn.Linear(hidden_dim, hidden_dim, True))
#         self.U = nn.ModuleList(U)
#         self.V = nn.Linear(hidden_dim, output_dim, True)
        self.V = nn.Conv2d(hidden_dim, output_dim, 1, 1, 0)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x                #B x V x V x H
        #print("UX",Ux)
#         for U_i in self.U:
#             Ux = U_i(Ux)  # B x H
#             Ux = F.relu(Ux)  # B x H
        Ux=Ux.permute(0,3,1,2) #B x H x V x V
        y = self.V(Ux)  # B x O
        #y = F.relu(y)
        y=y.permute(0,2,3,1)
        return y
