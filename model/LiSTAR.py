import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.nn.functional import relu

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.05)
    def forward(self, x):
        x =  torch.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x

class SpatialBlock(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, output_size, dropout=0.05, dim=10):
        super(SpatialBlock, self).__init__()      
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, num_nodes), requires_grad=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        # self.post_ln = nn.LayerNorm(input_size)
        # self.skip_proj = nn.Linear(input_size, output_size)

    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        x_spatial = torch.einsum('blnc, mn -> blmc', x, adp)
        # x_spatial = self.post_ln(x_spatial + x)
        y = torch.relu(self.fc1(x_spatial))
        y = self.fc2(self.dropout(y))
        # y = self.post_ln(y + self.skip_proj(x_spatial))
        
        return y
# [新增结束]

class LiSTAR(nn.Module):
    def __init__(self, win_size,d_model=256, local_size=[3, 5, 7],global_size=[3,5,7], channel=55,dropout=0.05, output_attention=True, graph_dim=10):
        super(LiSTAR, self).__init__()
        self.output_attention = output_attention
        self.local_size = local_size
        self.channel = channel
        self.win_size = win_size
        self.spac_size = nn.ModuleList(
            SpatialBlock(channel, localsize, d_model, global_size[index], dropout, graph_dim) 
            for index, localsize in enumerate(self.local_size)
        )
        self.spac_num = nn.ModuleList(
            SpatialBlock(channel, global_size[index], d_model, localsize, dropout, graph_dim)
            for index, localsize in enumerate(self.local_size)
        )

        self.mlp_size = nn.ModuleList(
            MLP(localsize, d_model, global_size[index]) for index, localsize in enumerate(self.local_size))
        self.mlp_num = nn.ModuleList(
            MLP(global_size[index], d_model, localsize) for index, localsize in enumerate(self.local_size))
        self.d_model = d_model

    def forward(self, in_size,in_num):
        local_mean = []
        global_mean = []
        local_mean_2 = []
        global_mean_2 = []
        B, L, M, _ = in_size[0].shape

        #temporal
        for index, localsize in enumerate(self.local_size):
            x_local, x_global = in_size[index], in_num[index]  # B L M N
            x_local = self.mlp_size[index](x_local)
            x_global = self.mlp_num[index](x_global)

            local_mean.append(x_local), global_mean.append(x_global)
        
        #spatial
        for index, localsize in enumerate(self.local_size):
            x_local, x_global = in_size[index], in_num[index]
            x_local_s = self.spac_size[index](x_local)
            x_global_s = self.spac_num[index](x_global)
            
            local_mean_2.append(x_local_s)
            global_mean_2.append(x_global_s)
        if self.output_attention:
            return local_mean, global_mean, local_mean_2, global_mean_2
        else:
            return None