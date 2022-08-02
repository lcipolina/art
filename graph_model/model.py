from torch_geometric.nn import GATConv, to_hetero
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import HeteroConv, GATConv


class MultiGNNEncoder(torch.nn.Module):
    def __init__(self, data, hidden_channels, type_layer = GATConv, activation = 'relu', aggregation = 'sum',
                 obj = 'style', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), shared = False, drop_rate = 0.2):
        super().__init__()
        self.aggregation = aggregation#aggregation function (useful when aggregating nodes)
        self.type_layer = type_layer#type layer of graph convolution
        self.activation = activation#avtivation function plugged after aggregation
        self.hidden_channels = hidden_channels#dimension of encoded features
        self.obj = obj#object node
        self.shared = shared#if true, a single gnn encoder is privided, otherwise, there is a gnn encoder for each target node
        self.device = device
        self.drop_rate = drop_rate
        self._build_layer(data)
   
    def _build_layer(self, data):
        type_edge = [e for e in data.edge_types if e[0] == 'artwork' and e[2] == self.obj][0]
        if self.shared:
            #if the model is in "shared parameters", than a single gnn encoder is provided
            self.encoders = HeteroConv({type_edge: self.type_layer((-1,-1), self.hidden_channels, dropout=self.drop_rate)}).to(self.device)
            self.encoders = nn.ModuleDict({'0': self.encoders})
        else:
            #if teh model is not in "shared parameters", than there will be a gnn encoder for each target node
            self.encoders = [HeteroConv({type_edge: self.type_layer((-1,-1), self.hidden_channels, dropout=0.2)}).to(self.device)
                        for _ in range(data[self.obj].x.shape[0])]
            self.encoders = nn.ModuleDict({str(ix): v for ix, v in enumerate(self.encoders)})

            
    def forward(self, x, edge_index):
        if self.shared:
            #if the model is shared, than the features are all the encoded obj nodes
            features = self.encoders['0'](x, edge_index)[self.obj]
        else:
            #if the model is not shared, the i-th gnn encoder is responsible of the i-th target node
            features =  torch.vstack(tuple(enc(x, edge_index)[self.obj][int(ix)] for ix, enc in self.encoders.items()))
        return self.activation(features.flatten())
   
    
class Head(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=1, activation = nn.ReLU, sub = 'artwork', obj = 'style', drop_rate = .2,
                bnorm = False):
        super().__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.sub = sub
        self.obj = obj
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation
        self.drop_rate = drop_rate
        self.bnorm = bnorm
        
        modules = []
        for _ in range(1, self.num_layers):
            modules.append(nn.Linear(hidden_channels, hidden_channels // 2))
            if self.bnorm and _ == 1:
                modules.append(nn.BatchNorm1d(hidden_channels // 2))
            if self.activation == nn.ReLU:
                modules.append(self.activation(inplace = True))
            elif self.activation == nn.Tanh:
                modules.append(self.activation())
            else:
                modules.append(self.activation(negative_slope = 0.1, inplace = True))
            hidden_channels //= 2
            modules.append(nn.Dropout(self.drop_rate))
        modules.append(nn.Linear(hidden_channels, out_channels))
        self.head = nn.Sequential(*modules)
        
    def forward(self, x, z):
        #return self.head(torch.cat((x, z * torch.ones(x.shape[0], z.shape[0]).to('cuda:0')), dim = 1))
        return self.head(torch.cat((x, z * torch.ones(x.shape[0], z.shape[0])), dim = 1))
    
    
class ModelClassification(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels= None, sub = 'artwork', obj = 'style',
                 gnn_activation = 'relu', gnn_type_layer = GATConv, aggr = 'sum',
                 head_num_layers = 1, head_activation = nn.ReLU, shared = False, bnorm = True,
                 drop_rate = 0.2):
        super().__init__()
        self.sub = sub
        self.obj = obj
        self.encoder = MultiGNNEncoder(hidden_channels = hidden_channels,
                                 type_layer=gnn_type_layer,
                                 activation=gnn_activation,
                                 aggregation = aggr,
                                 data = data,
                                 obj = obj,
                                 shared = shared,
                                 drop_rate = drop_rate)
        
        head_hidden_channels = hidden_channels * (data[obj].x.shape[0] + 1)
        self.decoder = Head(hidden_channels = head_hidden_channels,
                                  out_channels = out_channels if out_channels else data[obj].x.shape[0],
                                  num_layers=head_num_layers,
                                  activation=head_activation,
                                  sub=sub,
                                  obj = obj,
                                  bnorm = bnorm,
                                  drop_rate = drop_rate)
        
    def forward(self, x_dict, edge_index_dict, x):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x, z_dict)
