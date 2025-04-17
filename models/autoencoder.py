import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._knc = dict()

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruct = self.decoder(embedding)
        return embedding, reconstruct

    def set_knc(self, knc, key="auto"):
        self._knc[key] = knc

    def get_knc(self, key="auto"):
        return self._knc[key]


class Encoder(nn.Module):
    def __init__(self, dim_list, act_list, dropout=0.5):
        super(Encoder, self).__init__()

        layers = create_layer_list(dim_list, act_list, dropout=dropout)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x_encoded = self.encoder(x)
        return x_encoded


class VariationalEncoder(nn.Module):
    def __init__(self, dim_list, act_list):
        super(VariationalEncoder, self).__init__()
        layers = create_layer_list(dim_list, act_list)

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(dim_list[-1], dim_list[-1])
        self.fc_var = nn.Linear(dim_list[-1], dim_list[-1])
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        x_encoded = nn.functional.normalize(x_encoded)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=std.size())).type(torch.FloatTensor).cuda()
        embedding = mu + std * Variable(std_z, requires_grad=False)  # Re-parameterization trick
        return embedding, mu, std


class Decoder(nn.Module):
    def __init__(self, dim_list, act_list, dropout=0.5):
        super(Decoder, self).__init__()

        layers = create_layer_list(dim_list, act_list, dropout=dropout)

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        embedding = self.decoder(x)
        return embedding


def create_layer_list(dim_list, act_list, dropout):
    layers = []
    # iterate over list of layer widths
    for i in range(len(dim_list) - 1):
        # add layers according to widths defined
        layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
        # add activation
        assert act_list[i] in ['gelu', 'sigmoid', 'softmax']
        if act_list[i] == 'gelu':
            layers.append(nn.GELU())
        elif act_list[i] == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act_list[i] == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        if i == 0:
            layers.append(nn.Dropout(dropout))
    return layers


if __name__ == '__main__':
    dim_lst = [1024, 512, 256, 128, 64, 32, 16]
    enc = Encoder(dim_lst)
    vnc = VariationalEncoder(dim_lst)
    dec = Decoder(dim_lst[::-1])
    print(enc, vnc, dec)
