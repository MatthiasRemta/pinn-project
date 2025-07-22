import torch

from torch import nn
from torch.nn import functional as F, init


class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, branch_dim=4, output_dim=4, hidden_dim=32):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.branch_dim = branch_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        qp = x[:, :self.branch_dim]
        time = x[:, self.branch_dim:]
        
        bases = self.branch_net(qp)
        coefficients = self.trunk_net(time)

        output = bases * coefficients
        return output.view(output.shape[0], self.output_dim, self.hidden_dim//self.output_dim).sum(dim=2)

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
        activate_output=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.preprocessing = preprocessing
        self.activate_output = activate_output
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
        if self.activate_output:
            outputs = F.tanh(outputs)
        return outputs


class ConditionalLALayer(nn.Module):
    """"
    Upper or lower triangular layer for LA-SympNet. The weights depend on conditional parameters.
    """
    def __init__(self, input_dim, context_net, upper=True):
        super(ConditionalLALayer, self).__init__()
        self.input_dim = input_dim
        self.context_net = context_net
        self.upper = upper


    def forward(self, x):
        """
        Forward pass through the layer.
        """
        batch_size = x.shape[0]
        z = x[:, :self.input_dim]
        context = x[:, self.input_dim:]
        weights = self.context_net(context).view(batch_size, self.input_dim//2, self.input_dim//2)

        matrix = torch.eye(self.input_dim).reshape((1, self.input_dim, self.input_dim)).repeat(batch_size, 1, 1)
        if self.upper:
            matrix[:, self.input_dim//2:, :self.input_dim//2] = weights + weights.permute(0, 2, 1)
        else:
            matrix[:, :self.input_dim//2, self.input_dim//2:] = weights + weights.permute(0, 2, 1)
        return torch.cat([torch.matmul(z.unsqueeze(1), matrix).squeeze(1), context], dim=1)
    

class ConditionalLAModule(nn.Module):
    """"
    Concatenation of LA-Layers.
    """
    def __init__(self, input_dim, context_nets, layers=2):
        super(ConditionalLAModule, self).__init__()
        net = nn.ModuleList([])
        self.bias = nn.Parameter(torch.randn(input_dim) * 0.01)

        for i in range(layers):
            if i % 2 == 0:
                net.append(ConditionalLALayer(input_dim, context_nets[i], upper=True))
            else:
                net.append(ConditionalLALayer(input_dim, context_nets[i], upper=False))

        self.net = nn.Sequential(*net)
        self.input_dim = input_dim
            

    def forward(self, x):
        """
        Forward pass through the layer.
        """
        output = self.net(x)
        output[:, :self.input_dim] += self.bias
        return output
    

class ConditionalActivationModule(nn.Module):
    """"
    Symplectic activation (nonlinear) layer.
    """
    def __init__(self, input_dim, context_net, upper=True):
        super(ConditionalActivationModule, self).__init__()
        self.context_net = context_net
        self.activation = nn.Sigmoid()
        self.input_dim = input_dim
        self.upper = upper


    def forward(self, x):
        """
        Forward pass through the layer.
        """
        batch_size = x.shape[0]
        z = x[:, :self.input_dim]
        context = x[:, self.input_dim:]

        p, q = torch.chunk(z, 2, dim=-1)
        qp = torch.cat([q, p], dim=1)

        weights = self.context_net(context)
        
        indicator = torch.zeros(batch_size, self.input_dim)
        if self.upper:
            indicator[:, :self.input_dim//2] = weights
        else:
            indicator[:, self.input_dim//2:] = weights
        return torch.cat([z + self.activation(qp) * indicator, context], dim=1)
    

class ConditionalGAModule(nn.Module):
    """"
    Symplectic gradient module.
    """
    def __init__(self, input_dim, width, context_net, upper=True):
        super(ConditionalGAModule, self).__init__()
        self.context_net = context_net
        self.activation = nn.Sigmoid()
        self.input_dim = input_dim
        self.width = width
        self.upper = upper


    def forward(self, x):
        """
        Forward pass through the layer.
        """
        batch_size = x.shape[0]
        z = x[:, :self.input_dim]
        context = x[:, self.input_dim:]
        p, q = torch.chunk(z, 2, dim=-1)

        parameters = self.context_net(context)
        weights = parameters[:, :self.width*self.input_dim//2].view(batch_size, self.width, self.input_dim//2)
        scale = parameters[:, self.width*self.input_dim//2:(self.width+1)*self.input_dim//2].view(batch_size, self.input_dim//2)
        bias = parameters[:, (self.width+1)*self.input_dim//2:].view(batch_size, self.input_dim//2)

        if self.upper:
            q_new = self.activation(torch.matmul(p.unsqueeze(1), weights.permute(0, 2, 1)) + bias.unsqueeze(1)) * scale.unsqueeze(1)

            return torch.cat([p, (torch.matmul(q_new, weights) + q.unsqueeze(1)).squeeze(1), context], dim=1)
        else:
            p_new = self.activation(torch.matmul(q.unsqueeze(1), weights.permute(0, 2, 1)) + bias.unsqueeze(1)) * scale.unsqueeze(1)

            return torch.cat([(torch.matmul(p_new, weights) + p.unsqueeze(1)).squeeze(1), q, context], dim=1)
        

class ConditionalLASympNet(nn.Module):
    """
    Symplectic neural network with conditional LA-Layers.
    """
    def __init__(self, input_dim, la_context_nets, act_context_nets, layers=3, sublayers=2):
        super(ConditionalLASympNet, self).__init__()
        net = nn.ModuleList([])
        
        for i in range(layers-1):
            net.append(ConditionalLAModule(input_dim, la_context_nets[i], sublayers))
            net.append(ConditionalActivationModule(input_dim, act_context_nets[i], upper=(i % 2 == 0)))
        net.append(ConditionalLAModule(input_dim, la_context_nets[-1], sublayers))

        self.net = nn.Sequential(*net)
        self.input_dim = input_dim

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.net(x)[:, :self.input_dim]
    
class ConditionalGASympNet(nn.Module):
    """
    Symplectic neural network with conditional GA-Layers.
    """
    def __init__(self, input_dim, context_nets, layers=4, width=32):
        super(ConditionalGASympNet, self).__init__()
        net = nn.ModuleList([])
        
        for i in range(layers):
            net.append(ConditionalGAModule(input_dim, width, context_nets[i], upper=(i % 2 == 0)))

        self.net = nn.Sequential(*net)
        self.input_dim = input_dim

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.net(x)[:, :self.input_dim]

