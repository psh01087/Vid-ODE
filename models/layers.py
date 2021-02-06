import torch.nn as nn

def create_net(n_inputs, n_outputs, n_layers=1,
               n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))
    
    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def create_convnet(n_inputs, n_outputs, n_layers=1, n_units=128, nonlinear='tanh'):
    
    if nonlinear == 'tanh':
        nonlinear = nn.Tanh()
    else:
        raise NotImplementedError('There is no named')
    
    layers = []
    layers.append(nn.Conv2d(n_inputs, n_units, 3, 1, 1, dilation=1))
    
    for i in range(n_layers):
        layers.append(nonlinear)
        layers.append(nn.Conv2d(n_units, n_units, 3, 1, 1, dilation=1))
    
    layers.append(nonlinear)
    layers.append(nn.Conv2d(n_units, n_outputs, 3, 1, 1, dilation=1))
    
    return nn.Sequential(*layers)