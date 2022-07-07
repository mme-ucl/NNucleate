from torch import nn
import torch
from .utils import unsorted_segment_sum

# Linear Model
class NNCV(nn.Module):
    def __init__(self, insize, l1, l2=0, l3=0):
        """Instantiates an NN for approximating CVs. Supported are architectures with up to 3 layers.

        Args:
            insize (int): Size of the input layer
            l1 (int): Size of dense layer 1
            l2 (int, optional): Size of dense layer 2. Defaults to 0.
            l3 (int, optional): Size of dense layer 3. Defaults to 0.
        """
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
    def __init__(self, hidden_nf, act_fn=nn.ReLU()):
        """The graph convolutional layer for the graph-based model. Do not instantiate this directly.
        
        Args:
            hidden_nf (int): Hidden dimensionality of the latent node representation.
            act_fn (torch.nn.modules.activation, optional): PyTorch activation function to be used in the multi-layer perceptrons.
        """
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
    def __init__(
        self, in_node_nf=3, hidden_nf=3, device="cpu", act_fn=nn.ReLU(), n_layers=1
    ):
        """Graph-based model for the approximation of collective variables.

        Args:
            in_node_nf (int, optional): Dimensionality of the data in the graph nodes. Defaults to 3.
            hidden_nf (int, optional): Hidden dimensionality of the latent node representation. Defaults to 3.
            device (str, optional): Device the model should be stored on (For GPU support). Defaults to 'cpu'.
            act_fn (torch.nn.modules.activation, optional): PyTorch activation function to be used in the multi-layer perceptrons. Defaults to nn.ReLU().
            n_layers (int, optional): The number of graph convolutional layers. Defaults to 1.
        """
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
