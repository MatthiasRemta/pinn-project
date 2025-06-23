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

class LinearSympModule(nn.Module):
    '''Linear symplectic module.
    '''
    def __init__(self, dim, layers):
        super(LinearSympModule, self).__init__()
        self.dim = dim
        self.layers = layers
        
        self.params = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh[0], pqh[1], pqh[2]
        for i in range(self.layers):
            S = self.params['S{}'.format(i + 1)]
            if i % 2 == 0:
                p = p + q @ (S + S.t()) * h
            else:
                q = p @ (S + S.t()) * h + q
        return p + self.params['bp'] * h, q + self.params['bq'] * h
    
    def __init_params(self):
        '''Si is distributed N(0, 0.01), and b is set to zero.
        '''
        d = int(self.dim / 2)
        params = nn.ParameterDict()
        for i in range(self.layers):
            params['S{}'.format(i + 1)] = nn.Parameter((torch.randn([d, d]) * 0.01).requires_grad_(True))
        params['bp'] = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        params['bq'] = nn.Parameter(torch.zeros([d]).requires_grad_(True))
        return params
    
class ActivationModule(nn.Module):
    '''Activation symplectic module.
    '''
    def __init__(self, dim, activation, mode):
        super(ActivationModule, self).__init__()
        self.dim = dim
        self.activation = activation
        self.mode = mode
        
        self.params = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh[0], pqh[1], pqh[2]
        if self.mode == 'up':
            return p + self.act(q) * self.params['a'] * h, q
        elif self.mode == 'low':
            return p, self.act(p) * self.params['a'] * h + q
        else:
            raise ValueError
            
    def __init_params(self):
        d = int(self.dim / 2)
        params = nn.ParameterDict()
        params['a'] = nn.Parameter((torch.randn([d]) * 0.01).requires_grad_(True))
        return params
    
class GradientModule(nn.Module):
    '''Gradient symplectic module.
    '''
    def __init__(self, dim, width, activation, mode):
        super(GradientModule, self).__init__()
        self.dim = dim
        self.width = width
        self.activation = activation
        self.mode = mode
        
        self.params = self.__init_params()
        
    def forward(self, pqh):
        p, q, h = pqh[0], pqh[1], pqh[2]
        if self.mode == 'up':
            gradH = (self.act(q @ self.params['K'] + self.params['b']) * self.params['a']) @ self.params['K'].t()
            return p + gradH * h, q
        elif self.mode == 'low':
            gradH = (self.act(p @ self.params['K'] + self.params['b']) * self.params['a']) @ self.params['K'].t()
            return p, gradH * h + q
        else:
            raise ValueError
            
    def __init_params(self):
        d = int(self.dim / 2)
        params = nn.ParameterDict()
        params['K'] = nn.Parameter((torch.randn([d, self.width]) * 0.01).requires_grad_(True))
        params['a'] = nn.Parameter((torch.randn([self.width]) * 0.01).requires_grad_(True))
        params['b'] = nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        return params
    
class SympNet(nn.Module):
    def __init__(self):
        super(SympNet, self).__init__()
        self.dim = None
        
    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        dim = xh.size(-1)
        size = len(xh.size())
        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:] 
            pred = [x0]
            for _ in range(steps):
                pred.append(self(torch.cat([pred[-1], h], dim=-1)))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1).view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res
    
class GSympNet(SympNet):
    '''
    G-SympNet.
    Input: [num, dim] or [num, dim + 1]
    Output: [num, dim]
    '''
def __init__(self, dim, layers=3, width=20, activation='sigmoid'):
    super(GSympNet, self).__init__()
    self.dim = dim
    self.layers = layers
    self.width = width
    self.activation = activation
    
    self.modus = self.__init_modules()
    
def forward(self, pqh):
    d = int(self.dim / 2)
    if pqh.size(-1) == self.dim + 1:
        p, q, h = pqh[..., :d], pqh[..., d:-1], pqh[..., -1:]
    elif pqh.size(-1) == self.dim:
        p, q, h = pqh[..., :d], pqh[..., d:], torch.ones_like(pqh[..., -1:])
    else:
        raise ValueError
    for i in range(self.layers):
        GradM = self.modus['GradM{}'.format(i + 1)]
        p, q = GradM([p, q, h])
    return torch.cat([p, q], dim=-1)

def __init_modules(self):
    modules = nn.ModuleDict()
    for i in range(self.layers):
        mode = 'up' if i % 2 == 0 else 'low'
        modules['GradM{}'.format(i + 1)] = GradientModule(self.dim, self.width, self.activation, mode)
    return modules