import torch
from torch.nn import Module, ModuleList, BatchNorm1d, Identity, Conv1d, ReLU, Tanh, Linear, Softmax, Sigmoid

# 2 CNN layers with 120 neurons

class ConvolutionalBlock(Module):

    def __init__(self, in_features, out_features, kernel = 1, activation_function = "relu", normalization = True):
        super(ConvolutionalBlock, self).__init__()
        self.convo = Conv1d(in_features, out_features, kernel_size=kernel)
        self.normalization = BatchNorm1d(out_features) if normalization else Identity()
        self.act = self._choose_activation(activation_function)

    def _choose_activation(self, activation_function):
        if activation_function == 'relu': return ReLU()
        else: return Tanh()

    def forward(self, x):
        x = self.convo(x)
        x = self.normalization(x)
        x = self.act(x)
        return x
    
# 2 MLP of 60 neurons

class FunctionMLP(Module):

    def __init__(self, in_features, neurons = 60, actions = 3, is_value_function = True):
        super(FunctionMLP, self).__init__()
        out_features = 1 if is_value_function else actions
        self.mlp1 = Linear(in_features, out_features=neurons)
        self.act1 = ReLU()
        self.mlp2 = Linear(neurons, out_features=out_features)
        self.act2 = Sigmoid()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act1(x)
        x = self.mlp2(x)
        x = self.act2(x)
        return x
    
class DDQNNetwork(Module):

    def __init__(self, num_of_features, conv_channels = (120,240,120), kernel_size = (1,1,1), value_neurons = 60, num_of_actions = 3, internal_activation_type = "relu", normalization = True):
        super(DDQNNetwork, self).__init__()

        assert len(conv_channels) == len(kernel_size)

        self.convs = ModuleList()
        curr_size = num_of_features

        for conv_size, kernel in zip(conv_channels, kernel_size):
            convo = ConvolutionalBlock(curr_size, conv_size, kernel=kernel, activation_function=internal_activation_type)
            self.convs.append(convo)
            curr_size = conv_size

        self.value_function_mlp = FunctionMLP(curr_size, value_neurons)
        self.advantage_function = FunctionMLP(curr_size, value_neurons, actions=num_of_actions, is_value_function=False)

    def forward(self, x: torch.Tensor):
        for convo in self.convs:
            x = convo(x)

        batch_x = x.shape[0]
        flatten_x = x.view(batch_x, -1)

        value_x = self.value_function_mlp(flatten_x)
        adv_x = self.advantage_function(flatten_x)
        adv_mean_x = torch.mean(adv_x, dim=1, keepdim=True)

        return value_x + adv_x - adv_mean_x
