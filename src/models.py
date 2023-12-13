import os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats

import torch
from torch import nn

class VanillaAutoEncoder(nn.Module):
    def __init__(self, input_shape):
        super(VanillaAutoEncoder, self).__init__()

        self.input_shape = input_shape

        self.encoder = nn.Sequential(
            nn.Linear(self.input_shape, 512),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, self.input_shape),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


class VariationalEncoder(nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(VariationalEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.fc1 = nn.Linear(self.input_shape, 256)
        self.fc2_mu = nn.Linear(256, self.latent_shape)
        self.fc2_logvar = nn.Linear(256, self.latent_shape)
        self.relu = nn.ReLU()

    def _reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, X):
        X = self.relu(self.fc1(X))
        mu = self.fc2_mu(X)
        logvar = self.fc2_logvar(X)
        z = self._reparameterise(mu, logvar)
        return mu, logvar, z

class VariationalDecoder(nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(VariationalDecoder, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(self.latent_shape, 256)
        self.fc4 = nn.Linear(256, self.input_shape)

    def forward(self, X):
        X = self.relu(self.fc3(X))
        X = self.sigmoid(self.fc4(X))
        return X

class Classifier(nn.Module):
    def __init__(self, input_shape, hidden_layers, output_shape):
        super(Classifier, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        for idx, layer_shape in enumerate(self.hidden_layers):
            if idx == 0:
                X = self.relu(nn.Linear(self.input_shape, layer_shape)(X))
            else:
                X = self.relu(nn.Linear(self.hidden_layers[idx-1], layer_shape)(X))
        X = self.sigmoid(nn.Linear(self.hidden_layers[-1], self.output_shape)(X))
        return X
    

class VariationalAutoEncoder(nn.Module):
    def __init__(self, 
                 input_shape, 
                 latent_shape, 
                 hidden_layers, 
                 output_shape):
        super(VariationalAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.hidden_layers = hidden_layers
        self.output_shape = output_shape
        # Encoding part
        self.encoder = VariationalEncoder(self.input_shape, self.latent_shape)
        # Decoding part
        self.decoder = VariationalDecoder(self.input_shape, self.latent_shape)

        # Classification part (Take latent shape as input)
        self.classifier = Classifier(self.latent_shape,
                                     self.hidden_layers,
                                     self.output_shape)

   
    def forward(self, X):
        mu, logvar, X = self.encoder(X)
        Y = self.classifier(X)
        X = self.decoder(X)
        return X, mu, logvar, Y
