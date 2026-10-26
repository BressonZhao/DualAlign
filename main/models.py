# Neural components shared by FS pretraining and dual-alignment training.
# Node inputs are [N, F], edge_index is [2, E], and edge weights are [E].
# The active stage-two entry point builds GCN_reweight layers directly;
# the GNN wrapper classes below are separate reusable components.
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, fill_diag, mul
from typing import Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_sparse import matmul as torch_sparse_matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Identity in the forward pass; reverse and scale encoder gradients in backward.
# The domain head minimizes BCE while its input encoder opposes that objective.
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # alpha is a fixed coefficient for this call and receives no gradient.
        return grad_output.neg() * ctx.alpha, None

def spmm(src: Adj, other: Tensor, reduce: str = "sum") -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        src (Tensor or torch_sparse.SparseTensor]): The input sparse matrix,
            either a :class:`torch_sparse.SparseTensor` or a
            :class:`torch.sparse.Tensor`.
        other (Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    assert reduce in ['sum', 'add', 'mean', 'min', 'max']

    # torch_sparse supports all listed reductions; native sparse tensors use sum/add.
    if isinstance(src, SparseTensor):
        return torch_sparse_matmul(src, other, reduce)

    if reduce in ['sum', 'add']:
        return torch.sparse.mm(src, other)

    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"`torch.sparse.Tensor`.")

# GCN propagation with a separate structural normalization and adaptation weight.
# For tensor edge_index, each normalized message is scaled by (1-lmda)+lmda*w_e.
class GCN_reweight(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # Reset both trainable parameters and any normalization cached for a graph.
    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, lmda = 1) -> Tensor:
        """"""
        # Keep adaptation weights outside gcn_norm: degrees use the base adjacency.
        # edge_rw must follow the exact ordering of the propagated edges.
        edge_rw = edge_weight
        #print(edge_weight.shape)
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    # Enable caching only when this layer will reuse the same graph topology.
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # Newer PyG releases build propagate()'s signature from this comment, so list every kwarg.
        # propagate_type: (x: Tensor, edge_weight: OptTensor, lmda: float, edge_rw: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None, lmda = lmda, edge_rw = edge_rw)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor, lmda, edge_rw) -> Tensor:
        
        # x_j is [E, out_channels]; broadcast each scalar edge coefficient over channels.
        # lmda=0 keeps base GCN messages, and lmda=1 applies the full adaptation factor.
        x_j = (edge_weight.view(-1, 1) * x_j)
        x_j = (1-lmda) * x_j + (lmda) * (edge_rw.view(-1, 1) * x_j)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # The fused SparseTensor path uses adj_t values directly and bypasses message().
        return spmm(adj_t, x, reduce=self.aggr)

# GraphSAGE-style mean aggregation with weighted neighbor messages and a root input.
class GS_reweight(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, reducer='mean',
                 normalize_embedding=False):
        # Aggregation is fixed to mean here; the reducer argument is not consulted.
        super(GS_reweight, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.agg_lin = torch.nn.Linear(out_channels + in_channels, out_channels)

        self.normalize_emb = normalize_embedding

    def forward(self, x, edge_index, edge_weight, lmda):
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_weight = edge_weight, lmda = lmda)

    def message(self, x_j, edge_index, edge_weight, lmda):
        x_j = self.lin(x_j)
        x_j = (1-lmda) * x_j + (lmda) * (edge_weight.view(-1, 1) * x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # Combine the aggregated neighbor representation with each node's original input.
        aggr_out = torch.cat((aggr_out, x), dim=-1)
        aggr_out = self.agg_lin(aggr_out)
        aggr_out = F.relu(aggr_out)

        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


# Configurable GCN/GraphSAGE wrapper returning [N, conv_dim] embeddings.
# output_dim is retained in the signature but does not configure these layers.
class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GNN, self).__init__()
        if args.backbone == 'GCN':
            self.prop_input = GCN_reweight(input_dim, args.hidden_dim)
            self.prop_hidden = GCN_reweight(args.hidden_dim, args.hidden_dim)
            self.prop_out = GCN_reweight(args.hidden_dim, args.conv_dim)
        elif args.backbone == 'GS':
            self.prop_input = GS_reweight(input_dim, args.hidden_dim)
            self.prop_hidden = GS_reweight(args.hidden_dim, args.hidden_dim)
            self.prop_out = GS_reweight(args.hidden_dim, args.conv_dim)

        self.dropout = args.dropout
        self.K = args.K
        self.bn = args.bn
        self.lmda = args.rw_lmda
        self.hidden_dim = args.hidden_dim

        # conv layers
        self.conv = nn.ModuleList()
        self.conv.append(self.prop_input)
        # For K>3, intermediate entries reuse the same prop_hidden module and parameters.
        for i in range(args.K - 2):
            self.conv.append(self.prop_hidden)
        self.conv.append(self.prop_out)

        #bn layers
        self.bns = nn.ModuleList()
        for i in range(args.K - 1):
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

    def forward(self, x,data):
        edge_index, edge_weight = data.edge_index, data.edge_weight
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight=edge_weight, lmda = self.lmda)
            if self.bn and (i != len(self.conv) - 1):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# Optional feature projection followed by a gradient-reversal domain classifier.
# Returns (projected_features, domain_probability); domain labels are 0/1.
class Adv_DANN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, f_layer, dc_layer, args):
        super(Adv_DANN, self).__init__()
        self.f_layer = f_layer
        if f_layer > 0:
            self.f = nn.ModuleList()
            for l in range(f_layer):
                self.f.append(nn.Linear(input_dim, input_dim))

        self.dc = nn.ModuleList()
        if dc_layer == 1:
            self.dc.append(nn.Linear(input_dim, out_dim))
        else:
            self.dc.append(nn.Linear(input_dim, hidden_dim))
            for l in range(dc_layer - 2):
                self.dc.append(nn.Linear(hidden_dim, hidden_dim))
            self.dc.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = args.dropout
        self.resnet = args.resnet

    def forward(self, x, alpha):
        if self.f_layer > 0:
            x = self.f[0](x)
            if self.resnet:
                for i in range(1, len(self.f)):
                    if i != len(self.f) - 1:
                        x = self.f[i](x.clone()) + x.clone()
                    else:
                        x = self.f[i](x)
                    x = F.leaky_relu(x.clone())
                    x = F.dropout(x.clone(), p=self.dropout, training=self.training)
            else:
                for i in range(1, len(self.f)):
                    x = self.f[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

        # Reverse gradients only on the discriminator branch; returned x is unreversed.
        y = GradReverse.apply(x, alpha)

        for i in range(len(self.dc) - 1):
            y = self.dc[i](y)
            y = F.relu(y)
            # y = F.dropout(y, p=self.dropout)

        # Sigmoid produces probabilities for utils.BCE_loss, rather than logits.
        y = self.dc[len(self.dc) - 1](y)
        y = torch.sigmoid(y)
        return x, y


# FS feature encoder: [N, input_dim] -> [N, mlp_embed_dim], without graph edges.
class MLPs(torch.nn.Module):
    def __init__(self, input_dim, args):
        super(MLPs, self).__init__()
        self.mlp_encoder = nn.ModuleList()
        if args.mlp_layers == 1:
            self.mlp_encoder.append(nn.Linear(input_dim, args.mlp_embed_dim))
        else:
            self.mlp_encoder.append(nn.Linear(input_dim, args.hidden_dim))
            for i in range(args.mlp_layers - 2):
                self.mlp_encoder.append(nn.Linear(args.hidden_dim, args.hidden_dim))
            self.mlp_encoder.append(nn.Linear(args.hidden_dim, args.mlp_embed_dim))

    def forward(self, data):
        embed_x = data.x
        for i in range(len(self.mlp_encoder)):
            embed_x = self.mlp_encoder[i](embed_x)
            # Keep the final embedding linear; apply ReLU only between MLP layers.
            if i != (len(self.mlp_encoder) - 1):
                embed_x = F.relu(embed_x)
        return embed_x


# Node classifier returning [N, num_classes] logits for cross-entropy.
class Classifier(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim, args):
        super(Classifier, self).__init__()
        self.mlp_encoder = nn.ModuleList()
        if args.class_layers == 1:
            self.mlp_encoder.append(nn.Linear(input_dim, output_dim))
        else:
            self.mlp_encoder.append(nn.Linear(input_dim, hidden_dim))
            for i in range(args.class_layers - 2):
                self.mlp_encoder.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp_encoder.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, embed_x):
        for i in range(len(self.mlp_encoder)):
            embed_x = self.mlp_encoder[i](embed_x)
            if i != (len(self.mlp_encoder) - 1):
                embed_x = F.relu(embed_x)
        return embed_x


# GNN wrapper reading adaptation weights from data.edge_weight.
# As in GNN, repeated hidden entries (K>3) share parameters.
class GNN_simple(torch.nn.Module):
    def __init__(self, input_dim, args):
        super(GNN_simple, self).__init__()
        if args.backbone == 'GCN':
            self.prop_input = GCN_reweight(input_dim, args.gcn_dim)
            self.prop_hidden = GCN_reweight(args.gcn_dim, args.gcn_dim)
            self.prop_out = GCN_reweight(args.gcn_dim, args.conv_dim)
        elif args.backbone == 'GS':
            self.prop_input = GS_reweight(input_dim, args.hidden_dim)
            self.prop_hidden = GS_reweight(args.hidden_dim, args.hidden_dim)
            self.prop_out = GS_reweight(args.hidden_dim, args.conv_dim)

        self.dropout = args.dropout
        self.K = args.K
        self.bn = args.bn
        self.lmda = args.rw_lmda
        self.gcn_dim = args.gcn_dim

        # conv layers
        self.conv = nn.ModuleList()
        self.conv.append(self.prop_input)
        for i in range(args.K - 2):
            self.conv.append(self.prop_hidden)
        self.conv.append(self.prop_out)

        #bn layers
        self.bns = nn.ModuleList()
        for i in range(args.K - 1):
            self.bns.append(nn.BatchNorm1d(self.gcn_dim))

    def forward(self, x,data):
        edge_index, edge_weight = data.edge_index, data.edge_weight
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight=edge_weight, lmda = self.lmda)
            if self.bn and (i != len(self.conv) - 1):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x



# GNN wrapper accepting an explicit [E] adaptation-weight vector at forward time.
# The vector is reused at every layer; stage-two training instead tracks one per layer.
class GNN_reweight(torch.nn.Module):
    def __init__(self, input_dim, args):
        super(GNN_reweight, self).__init__()
        if args.backbone == 'GCN':
            self.prop_input = GCN_reweight(input_dim, args.gcn_dim)
            self.prop_hidden = GCN_reweight(args.gcn_dim, args.gcn_dim)
            self.prop_out = GCN_reweight(args.gcn_dim, args.conv_dim)
        elif args.backbone == 'GS':
            self.prop_input = GS_reweight(input_dim, args.hidden_dim)
            self.prop_hidden = GS_reweight(args.hidden_dim, args.hidden_dim)
            self.prop_out = GS_reweight(args.hidden_dim, args.conv_dim)

        self.dropout = args.dropout
        self.K = args.K
        self.bn = args.bn
        self.lmda = args.rw_lmda
        self.gcn_dim = args.gcn_dim

        # conv layers
        self.conv = nn.ModuleList()
        self.conv.append(self.prop_input)
        for i in range(args.K - 2):
            self.conv.append(self.prop_hidden)
        self.conv.append(self.prop_out)

        #bn layers
        self.bns = nn.ModuleList()
        for i in range(args.K - 1):
            self.bns.append(nn.BatchNorm1d(self.gcn_dim))

    def forward(self, x,data,edge_weight):
        for i, layer in enumerate(self.conv):
            x = layer(x, data.edge_index, edge_weight=edge_weight, lmda = self.lmda)
            if self.bn and (i != len(self.conv) - 1):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
