import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers=[4096],p=0.5):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        layers_ = layers[:]
        for layer in range(num_layers - 1):
            if len(layers_) == 0:
                outgoing = incoming
            else:
                outgoing = layers_.pop(0)
            linear_layer = nn.Linear(incoming, outgoing, bias = bias)            
            mod.append(linear_layer)
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace = True))
            if dropout:
                mod.append(nn.Dropout(p = p))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
        self.mod = nn.Sequential(*mod)
        self.apply(weights_init)
    def forward(self, x):
        return self.mod(x)

class Delta_MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers=[4096], p=0.5):
        super(Delta_MLP, self).__init__()
        mod = []
        incoming = inp_dim
        layers_ = layers[:]
        for layer in range(num_layers - 1):
            if len(layers_) == 0:
                outgoing = incoming
            else:
                outgoing = layers_.pop(0)
            linear_layer = nn.Linear(incoming, outgoing, bias = bias)            
            mod.append(linear_layer)
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))

            mod.append(nn.LeakyReLU(0.2))
            if dropout:
                mod.append(nn.Dropout(p = p))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU())
        self.mod = nn.Sequential(*mod)
        self.apply(weights_init)
    def forward(self, x):

        return self.mod(x)
    
# class Delta_MLP(nn.Module):
#     def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers=[]):
#         super(Delta_MLP, self).__init__()
        
#         self.fc1 = nn.Linear(inp_dim, 512)
        
#         self.fc2 = nn.Linear(512, 512)
        
#         self.fc3 = nn.Linear(512, out_dim)

#         self.layernorm = nn.LayerNorm(512)
#         self.act = nn.ReLU()
#         self.dropout = nn.Dropout1d(0.5)

#         self.apply(weights_init)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.layernorm(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         res = x
#         x = self.fc2(x)
#         x = self.layernorm(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x+=res
#         x = self.fc3(x)
#         return x
    



