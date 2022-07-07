from torch import nn


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
                    nn.Linear(l3, 1)
                )
            else:
                self.sig_stack = nn.Sequential(
                    nn.Linear(insize, l1),
                    nn.Sigmoid(),
                    nn.Linear(l1, l2),
                    nn.Sigmoid(),
                    nn.Linear(l2, 1)
                )
        else:
            self.sig_stack = nn.Sequential(
                nn.Linear(insize, l1),
                nn.Sigmoid(),
                nn.Linear(l1, 1)
            )

    def forward(self, x):
        # defines the application of the network to data
        # NEVER call forward directly
        # Only say model(x)
        x = self.flatten(x)
        label = self.sig_stack(x)
        return label