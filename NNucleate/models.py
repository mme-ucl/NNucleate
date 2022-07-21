from torch import nn
import torch
from .utils import unsorted_segment_sum


def initialise_weights(model: nn.Module):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)


# Linear Model
class NNCV(nn.Module):
    """Instantiates an NN for approximating CVs. Supported are architectures with up to 3 layers.

    :param insize: Size of the input layer.
    :type insize: int
    :param l1: Size of dense layer 1.
    :type l1: int
    :param l2: Size of dense layer 2, defaults to 0.
    :type l2: int, optional
    :param l3: Size of dense layer 3, defaults to 0.
    :type l3: int, optional
    """

    def __init__(self, insize: int, l1: int, l2=0, l3=0):
        super(NNCV, self).__init__()
        self.flatten = nn.Flatten()
        # defines the structure
        if l2 > 0:
            if l3 > 0:
                self.sig_stack = nn.Sequential(
                    nn.Linear(insize, l1),
                    nn.Sigmoid(),
                    nn.Linear(l1, l2),
                    nn.Sigmoid(),
                    nn.Linear(l2, l3),
                    nn.Sigmoid(),
                    nn.Linear(l3, 1),
                )
            else:
                self.sig_stack = nn.Sequential(
                    nn.Linear(insize, l1),
                    nn.Sigmoid(),
                    nn.Linear(l1, l2),
                    nn.Sigmoid(),
                    nn.Linear(l2, 1),
                )
        else:
            self.sig_stack = nn.Sequential(
                nn.Linear(insize, l1), nn.Sigmoid(), nn.Linear(l1, 1)
            )

    def forward(self, x):
        # defines the application of the network to data
        # NEVER call forward directly
        # Only say model(x)
        x = self.flatten(x)
        label = self.sig_stack(x)
        return label


# Graph model
class GCL(nn.Module):
    """The graph convolutional layer for the graph-based model. Do not instantiate this directly.

    :param hidden_nf: Hidden dimensionality of the latent node representation.
    :type hidden_nf: int
    :param act_fn: PyTorch activation function to be used in the multi-layer perceptrons, defaults to nn.ReLU()
    :type act_fn: torch.nn.modules.activation, optional
    """

    def __init__(self, hidden_nf: int, act_fn=nn.ReLU()):
        super(GCL, self).__init__()

        self.edge_mlp = nn.Sequential(
            # Only takes the neighbourhood node
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            # Maps to the same dimension
            nn.Linear(hidden_nf, hidden_nf),
        )

        self.node_mlp = nn.Sequential(
            # Node MLP just takes the current vector and the resulting neighbourhood vector
            nn.Linear(hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

    def edge_model(self, source, target):

        out = torch.cat([source - target], dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, _ = edge_index
        # Get the summed edge vectors for each node
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)

        return out

    def forward(self, h, edge_index):
        row, col = edge_index

        edge_feat = self.edge_model(h[row], h[col])
        h = self.node_model(h, edge_index, edge_feat)
        return h


class GNNCV(nn.Module):
    """_summary_

    :param in_node_nf: Dimensionality of the data in the graph nodes, defaults to 3.
    :type in_node_nf: int, optional
    :param hidden_nf: Hidden dimensionality of the latent node representation, defaults to 3.
    :type hidden_nf: int, optional
    :param device: Device the model should be stored on (For GPU support), defaults to "cpu".
    :type device: str, optional
    :param act_fn: PyTorch activation function to be used in the multi-layer perceptrons, defaults to nn.ReLU().
    :type act_fn: torch.nn.modules.activation, optional
    :param n_layers:  The number of graph convolutional layers, defaults to 1.
    :type n_layers: int, optional
    """

    def __init__(
        self, in_node_nf=3, hidden_nf=3, device="cpu", act_fn=nn.ReLU(), n_layers=1
    ):

        super(GNNCV, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, act_fn=act_fn))

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1),
        )
        self.to(self.device)

    def forward(self, x, edges, n_nodes):
        h = self.embedding(x)  # turn 1D h into internal h
        for i in range(0, self.n_layers):
            h = self._modules["gcl_%d" % i](h, edges)  # update h

        h = self.node_dec(h)  # pipe H through a MLP
        h = h.view(-1, n_nodes, self.hidden_nf)  # stacks all the hidden_nf
        h = torch.sum(h, dim=1)  # create one vector containinf the hidden h sums
        pred = self.graph_dec(h)  #
        return pred.squeeze(1)
