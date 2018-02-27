from torch import nn


class MTMLP(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims, dropout=0.2,
                 binary=False, share_input=False):
        self.binary = binary
        self.share_input = share_input
        super(MTMLP, self).__init__()
        if type(hidden_dims) == int:
            print("Warning: you passed a single integer ({}) as argument "
                  "hidden_dims, but a list of integers specifying dimensions"
                  "for all hidden layers is expected. The model will now have"
                  "a single hidden layer with the dimensionality you "
                  "specified.".format(hidden_dims))
            hidden_dims = [hidden_dims]

        self.all_parameters = nn.ParameterList()

        # Define task inputs
        self.inputs = {}
        if share_input:
            layer = nn.Linear(input_dims[0], hidden_dims[0])
            self.all_parameters.append(layer.weight)
            for task_id in range(len(input_dims)):
                self.inputs[task_id] = layer
        else:
            for task_id, input_dim in enumerate(input_dims):
                layer = nn.Linear(input_dim, hidden_dims[0])
                self.inputs[task_id] = layer
                self.all_parameters.append(layer.weight)

        # Define shared hidden layers
        self.shared = []
        for i in range(len(hidden_dims)-1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.shared.append(layer)
            self.all_parameters.append(layer.weight)

        # Define outputs
        self.outputs = []
        for output_dim in output_dims:
            layer = nn.Linear(hidden_dims[-1], output_dim)
            self.outputs.append(layer)
            self.all_parameters.append(layer.weight)

        # predict at lowest hidden layer, output dimensionality is number
        # of languages (== number of tasks == number of outputs)
        self.lang_id_output = nn.Linear(hidden_dims[0], len(output_dims))
        self.all_parameters.append(self.lang_id_output.weight)

        # Define nonlinearity and dropout (used across all hidden
        # layers in self.forward())
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Dropout(0.0)

        # Initialize all weights
        for weight_matrix in self.all_parameters:
            nn.init.xavier_normal(weight_matrix)

    def forward(self, x, input_task_id=0, output_all=True, train_mode=False,
                output_lang_id=True):
        """
        Defines a forward pass of the model. Note we don't softmax the output
        here, this is done by the loss function
        :param x: fixed-size input
        :param input_task_id: which task ID to use for input
        :param output_all: whether to return outputs for all tasks
        :param train_mode: dropout yay or nay
        :return: a list containing the linear class distribution
        for every model output
        """
        # print(input_task_id)
        dropout = self.dropout if train_mode else lambda u: u
        x = dropout(self.tanh(self.inputs[input_task_id](x)))
        lang_id = self.lang_id_output(x)
        for layer in self.shared:
            x = dropout(self.tanh(layer(x)))

        output = [output(x) for output in self.outputs] if output_all \
            else self.outputs[input_task_id](x)
        if output_lang_id:
            return output, lang_id
        else:
            return output
