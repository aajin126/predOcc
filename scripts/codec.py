from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual blocks: 
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Encoder & Decoder Architecture:
# Encoder:
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU()
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU()
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x # predicted_map 

# Decoder:
class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.Sequential(*[
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=num_hiddens,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU()
                                        ])

        self._conv_trans_1 = nn.Sequential(*[
                                            nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(),                  
                                            nn.Conv2d(in_channels=num_hiddens//2,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.Sigmoid()
                                        ])

    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._conv_trans_1(x)
        return x
