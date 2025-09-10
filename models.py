import torch

from torch import nn
from torch.nn import functional as F, init


class Fourier1d(nn.Module):
    def __init__(self, complex_input=False, channels=1, modes=12):
        super(Fourier1d, self).__init__()
        self.weights = nn.Parameter(torch.randn(channels, channels, modes, dtype=torch.cdouble))
        self.linear = nn.Conv1d(channels, channels, 1)
        self.fft = torch.fft.fft if complex_input else torch.fft.rfft
        self.ifft = torch.fft.ifft if complex_input else torch.fft.irfft
        self.complex_input = complex_input
        self.modes = modes

    def forward(self, x):
        batch_size, channels, gridpoints = x.shape
        tmp = self.fft(x)[:, :, :self.modes]   # truncate higher mode
        #tmp = torch.einsum('bcm,ocm->bom', tmp, self.weights)
        tmp = torch.einsum('abc,bdc->adc', tmp, self.weights)
        tmp = self.ifft(tmp, n=gridpoints)   # pad to original input size
        mixing = self.linear(x)
        output =  F.gelu(tmp + mixing)
        return output
    

class FNO1d(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, output_channels=1, complex_input=False, fourier_layers=2, modes=12):
        super(FNO1d, self).__init__()
        model = nn.ModuleList([])
        model.append(nn.Conv1d(input_channels, hidden_channels, 1))
        for _ in range(fourier_layers):
            model.append(Fourier1d(complex_input=complex_input, channels=hidden_channels, modes=modes))

        model.append(nn.Conv1d(hidden_channels, output_channels, 1))

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, branch_dim=2):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.branch_dim = branch_dim
    
    def forward(self, x):
        qp = x[:, :self.branch_dim]
        time = x[:, self.branch_dim:]
        
        bases = self.branch_net(qp)
        coefficients = self.trunk_net(time)

        return bases * coefficients


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
    def __init__(
        self,
        features,
        activation=nn.Tanh,
    ):
        super().__init__()
        net = nn.ModuleList([])
        
        for _ in range(2):
            net.append(activation())
            net.append(nn.Linear(features, features))

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        tmp = self.net(inputs)
        return inputs + tmp


class ResidualNet(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        num_blocks=2,
        activation=nn.Tanh,
        activate_output=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.activate_output = activate_output

        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs):
        tmp = self.initial_layer(inputs)
        
        for block in self.blocks:
            tmp = block(tmp)
        
        outputs = self.final_layer(tmp)
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

