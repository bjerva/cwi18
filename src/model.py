from torch import nn


class MTMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dims, binary=False):
        self.binary = binary
        super(MTMLP, self).__init__()
        if type(hidden_dims) == int:
            print("Warning: you passed a single integer ({}) as argument "
                  "hidden_dims, but a list of integers specifying dimensions"
                  "for all hidden layers is expected. The model will now have"
                  "a single hidden layer with the dimensionality you "
                  "specified.".format(hidden_dims))
            hidden_dims = [hidden_dims]

        self.all_parameters = nn.ParameterList()
        # Define input and hidden layers
        self.dimensionalities = [input_dim] + hidden_dims
        self.hidden = []
        i = 0
        for i in range(len(hidden_dims)):
            layer = nn.Linear(self.dimensionalities[i],
                              self.dimensionalities[i+1])
            self.hidden.append(layer)
            self.all_parameters.append(layer.weight)

        # Define outputs
        self.outputs = []
        for output_dim in output_dims:
            layer = nn.Linear(self.dimensionalities[i+1], output_dim)
            self.outputs.append(layer)
            self.all_parameters.append(layer.weight)

        # Define nonlinearity and dropout (used across all hidden
        # layers in self.forward())
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Dropout(0.0)

        # Initialize all weights
        for weight_matrix in self.all_parameters:
            nn.init.xavier_normal(weight_matrix)

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
