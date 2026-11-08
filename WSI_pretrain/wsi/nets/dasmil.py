import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils._subgraph import subgraph



gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GATV2": geom_nn.GATv2Conv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, heads=3):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             heads=heads,
                             concat=False,
                             dropout=dp_rate)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        # GNN 对应的层
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


# Bag-level classifier class
class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            # Non-linear transformation
            self.lin = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            # Identity transformation
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class,
                             kernel_size=input_size)

    def forward(self, feats, c):
        # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # Handle multiple classes without for loop
        # Sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # Select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # Compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # Compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        # Normalize attention scores, A in shape N x C,
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        # Compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B



# MILNet module for multiple instance learning.
class MILNet(nn.Module):
    def __init__(self, 
                 i_classifier, # i_classifier (nn.Module): Instance-level classifier 
                 b_classifier # Bag-level classifier.
                 ):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B
    
# Randomly drops nodes from the adjacency matrix
def dropout_node(edge_index, 
                 p = 0.5,
                 num_nodes = None,
                 training = True):
    # Check if dropout probability is valid
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')
    # Get the number of nodes from edge_index or num_nodes argument
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # If not in training mode or p=0, return all nodes and edges
    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    # Generate random probabilities for each node
    prob = torch.rand(num_nodes, device=edge_index.device)

    # Create a node mask based on the dropout probability
    node_mask = prob > p
    # Subgraph the original edge_index and get the corresponding edge mask
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    # Return the retained edge_index, edge_mask, and node_mask
    return edge_index, edge_mask, node_mask





class DASMIL(nn.Module):
    def __init__(self,
                 c_in = 384,
                 c_hidden = 256,
                 classes = 1,
                 num_layers = 1,
                 layer_name = "GAT",
                 dropout = True,
                 dropout_rate = 0.2,
                 heads = 3
                ):
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.classes = classes
        self.dropout = True
        self.dropout_rate = 0.2
        self.kl = None
        self.target = "higher"
        
        super().__init__()
        self.GNNlast = GNNModel(c_in=self.c_in,
                            c_hidden=self.c_hidden,
                            c_out=self.c_hidden,
                            num_layers=num_layers, layer_name=layer_name, dp_rate=dropout_rate,heads=heads)
        self.gnn1 = GNNModel(c_in=self.c_in,
                            c_hidden=self.c_hidden,
                            c_out=self.c_in,
                            num_layers=num_layers, layer_name=layer_name, dp_rate=dropout_rate,heads=heads)
        self.gnn2 = GNNModel(c_in=self.c_in,
                            c_hidden=self.c_hidden,
                            c_out=self.c_in,
                            num_layers=num_layers, layer_name=layer_name, dp_rate=dropout_rate,heads=heads)

        mil2fc,mil2bag = FCLayer(self.c_hidden,self.classes),BClassifier(self.c_hidden,self.classes)
        self.mil2 = MILNet(mil2fc,mil2bag)
        mil3fc,mil3bag = FCLayer(self.c_hidden,self.classes),BClassifier(self.c_hidden,self.classes)
        self.mil3 = MILNet(mil3fc,mil3bag)

    # Forward pass at the scale level.
    def forward_scale(self,x: torch.Tensor,edge_index: torch.Tensor,gnnlayer: torch.nn.Module, glu=None)-> tuple[torch.Tensor,torch.Tensor]:
        r = x
        if self.training and self.dropout:
            edge_index, _, _ = dropout_node(edge_index=edge_index)
        x = gnnlayer(x, edge_index)
        return x, edge_index


    def forward_gnn(self,x,edge_index,levels,childof,edge_index2=None,edge_index3=None):
        results={}
        featsperlevel=[]
        indecesperlevel=[]
        # forward input for each scale gnn
        for i in levels.unique():
            #select scale
            indeces_feats=(levels==i).nonzero().view(-1)
            feats=x[indeces_feats]
            if i==levels.min():
                feats,_= self.forward_scale(feats,edge_index2,self.gnn1)
            else:
                feats,_= self.forward_scale(feats,edge_index3,self.gnn2)
            featsperlevel.append(feats)
            indecesperlevel.append(indeces_feats)
        #merge scales
        feats=torch.concat(featsperlevel) #[N+M,250]
        indeces= torch.concat(indecesperlevel).view(-1)#[N+M,]
        indeces= indeces.sort()[1]
        feats= feats[indeces]
        #forward togheter
        feats,edge_index2= self.forward_scale(feats,edge_index,self.GNNlast)
        results["childof"]=childof[indecesperlevel[1]]
        return feats,indecesperlevel,results

    def forward_mil(self,indecesperlevel,feats,results):
        #second step: MIL
        results["lower"]=self.mil2(feats[results["childof"].unique()].view(-1,self.c_hidden))#x5
        results["higher"]=self.mil3(feats[indecesperlevel[1]].view(-1,self.c_hidden))#x20
        return results
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor = None, edge_index3: torch.Tensor = None):
        feats, indecesperlevel, results = self.forward_gnn(x, edge_index, levels, childof, edge_index2, edge_index3)
        results = self.forward_mil(indecesperlevel, feats, results)
        return results

    def predict(self, results):
        prediction_patch_higher, prediction_bag_higher, _, _ = results[self.target]

        if self.classes > 1:
            # For multi-class classification
            higher_prediction = torch.sigmoid(prediction_bag_higher)
            if self.kl is not None:
                # If second scale predictions are available
                _, prediction_bag_lower, A, B = results[self.kl]
                lower_prediction = 0.5 * \
                    torch.sigmoid(prediction_bag_lower)+0.5 * \
                    torch.sigmoid(prediction_bag_higher)[:, -1]
            else:
                lower_prediction = None
        else:
            # For binary classification
            higher_prediction = 0.5 * \
                torch.sigmoid(prediction_bag_higher)[
                    :, -1]+0.5 * torch.sigmoid(prediction_bag_higher)[:, -1]
            if self.kl is not None:
                # If second scale predictions are available
                _, prediction_bag_lower, _, _ = results[self.kl]
                lower_prediction = torch.sigmoid(prediction_bag_lower)[:, -1]
            else:
                lower_prediction = None
        return higher_prediction, lower_prediction
