from torch import nn


class MTMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dims):
        super(MTMLP, self).__init__()
        if type(hidden_dims) == int:
            print("Warning: you passed a single integer ({}) as argument "
                  "hidden_dims, but a list of integers specifying dimensions"
                  "for all hidden layers is expected. The model will now have"
                  "a single hidden layer with the dimensionality you "
                  "specified.".format(hidden_dims))
            hidden_dims = [hidden_dims]

        # Define input and hidden layers
        self.dimensionalities = [input_dim] + hidden_dims
        self.hidden = []
        i = 0
        for i in range(len(hidden_dims)):
            self.hidden.append(nn.Linear(self.dimensionalities[i],
                                         self.dimensionalities[i + 1]))
        # Define outputs
        self.outputs = []
        for output_dim in output_dims:
            self.outputs.append(nn.Linear(self.dimensionalities[i + 1],
                                          output_dim))

        # Define nonlinearity and dropout (used across all hidden
        # layers in self.forward())
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

        # Initialize all weights
        for layer in self.hidden + self.outputs:
            nn.init.xavier_normal(layer.weight)

    def forward(self, x):
        """
        Defines a forward pass of the model. Note we don't softmax the output
        here, this is done by the loss function
        :param x: fixed-size input
        :return: a list containing the linear class distribution
        for every model output
        """
        h = x
        for layer in self.hidden:
            h = self.dropout(self.tanh(layer(h)))
        return [output(h) for output in self.outputs]
