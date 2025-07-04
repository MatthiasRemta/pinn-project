import torch

from torch import nn
from torch.nn import functional as F, init

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='tanh'):
        super(MLP, self).__init__()
        net = nn.ModuleList([])

        net.append(nn.Linear(input_dim, hidden_dim[0]))
        if activation == 'tanh':
            net.append(nn.Tanh())
        elif activation == 'relu':
            net.append(nn.ReLU())

        for i in range(len(hidden_dim) - 1):    
            net.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            if activation == 'tanh':
                net.append(nn.Tanh())
            elif activation == 'relu':
                net.append(nn.ReLU())

        net.append(nn.Linear(hidden_dim[-1], output_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """A general-purpose residual block."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.tanh,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.tanh,
        dropout_probability=0.0,
        use_batch_norm=False,
        preprocessing=None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.preprocessing = preprocessing
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if self.preprocessing is None:
            temps = inputs
        else:
            temps = self.preprocessing(inputs)
        if context is None:
            temps = self.initial_layer(temps)
        else:
            temps = self.initial_layer(torch.cat((temps, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super(DeepOnet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, x_branch, x_trunk):
        branch_output = self.branch_net(x_branch)
        trunk_output = self.trunk_net(x_trunk)
        output = torch.einsum('bi,bi->b', branch_output, trunk_output)
        return output
