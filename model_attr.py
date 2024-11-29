import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import FNHP_pytorch_attributes
import numpy as np
import os

torch.manual_seed(int(os.getenv('SEED_VAL')))

class TDG_FNHP(nn.Module):
    
                        
    def __init__(self, device, latent_dim, hidden_dim, fnn_dim, rnn_dim, lstm_dim, layers_fnn, layers_rnn, layers_lstm, learning_rate, noise_dim, step_size, dataset_name):

        super(TDG_FNHP, self).__init__()
        
        self.weights_fnn = None
        self.weights_rnn = None
        self.init_lin_h = nn.Linear(noise_dim, latent_dim)
        self.init_lin_c = nn.Linear(noise_dim, latent_dim)
        
        self.init_input = nn.Linear(noise_dim, latent_dim)
        self.num_rnn_layer = layers_rnn
        self.num_lstm_layer = layers_lstm
        self.step_size = step_size
        self.size_latent = latent_dim
        self.size_hidden = hidden_dim
        self.size_rnn = rnn_dim
        self.size_fnn = fnn_dim
        self.num_fnn_layer = layers_fnn
        self.learning_rate = learning_rate
        self.lstm_dim = lstm_dim
        
        self.FNHP_model = FNHP_pytorch_attributes.FNHP_pytorch(num_fnn_layer=self.num_fnn_layer, num_rnn_layer=self.num_rnn_layer, size_fnn=self.size_fnn,size_rnn=self.size_rnn,time_step=self.step_size,dataset=dataset_name)
        
        self.total_params_rnn = sum([torch.prod(torch.tensor(param.shape)).item() for param in self.FNHP_model.simple_rnn_layer[0].parameters()])

        self.total_params_fnn = 0
        
        for i in range(len(self.FNHP_model.dense_layers)):
            self.total_params_fnn += sum([torch.prod(torch.tensor(param.shape)).item() for param in self.FNHP_model.dense_layers[i].parameters()])
        self.LSTM = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,num_layers=self.num_lstm_layer)
        
        self.init_input = nn.Linear(noise_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.total_params_fnn + self.total_params_rnn, latent_dim),
            nn.ReLU()            
        )
        

        self.decoder_fnn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, self.total_params_fnn
            ),
            nn.Softplus()
        )
        self.decoder_rnn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, self.total_params_rnn
            ),
            nn.Tanh()
        )
        self.device = device

    def forward(self, X, y, only_attr, init_noise, z, E=None, H_C=None, newstep = False, epoch=0):
               
        if H_C == None and E == None:
            init_c, init_h = [], []            
            for _ in range(self.num_lstm_layer):
                init_c.append(torch.tanh(self.init_lin_h(init_noise)))
                init_h.append(torch.tanh(self.init_lin_c(init_noise)))
            # Initialize hidden inputs for the LSTM
            H_C = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
            
            inputs = torch.tanh(self.init_input(init_noise))
        else:
            inputs = self.encoder(E)
            
        old_H, old_C = H_C
        
        out, H_C = self.LSTM(inputs.unsqueeze(0))
        
        H_C = old_H, old_C      
        weights_fnn = self.decoder_fnn(out.squeeze(0)[-1].unsqueeze(0)).squeeze()
        weights_rnn = self.decoder_rnn(out.squeeze(0)[-1].unsqueeze(0)).squeeze()

        self.weights_rnn = weights_rnn
        self.weights_fnn = weights_fnn
        
        start_index = 0
        
        totalws = 0
        length = 0
        length = len(self.FNHP_model.dense_layers)
        for i in range(length):
            weight_shape = self.FNHP_model.dense_layers[i].weight.shape
            n_weights = torch.prod(torch.tensor(weight_shape)).item()
            totalws += n_weights
            
            weights = weights_fnn[start_index: start_index + n_weights]
            
            weights = torch.reshape(weights, shape=weight_shape)
            
            layer = self.FNHP_model.dense_layers[i]
            
            del layer.weight
            layer.weight = weights
            
            start_index += n_weights   

            if(self.FNHP_model.dense_layers[i].bias != None):
                bias_shape = self.FNHP_model.dense_layers[i].bias.shape
                n_bias = torch.prod(torch.tensor(bias_shape)).item()
                totalws+= n_bias

                bias = weights_fnn[start_index: start_index + n_bias]
                bias = torch.reshape(bias, shape=bias_shape)
                
                del layer.bias
                layer.bias = bias
                start_index += n_bias
        start_index = 0

        weight_hh_shape = self.FNHP_model.simple_rnn_layer[0].weight_hh_l0.shape
        n_weight_hh = torch.prod(torch.tensor(weight_hh_shape)).item()
        weight_hh = weights_rnn[start_index: start_index + n_weight_hh]
        rlayer = self.FNHP_model.simple_rnn_layer[0]

        del rlayer.weight_hh_l0
        rlayer.weight_hh_l0 = weight_hh

        start_index += n_weight_hh

        weight_ih_shape = self.FNHP_model.simple_rnn_layer[0].weight_ih_l0.shape
        n_weight_ih = torch.prod(torch.tensor(weight_ih_shape)).item()
        weight_ih = weights_rnn[start_index: start_index + n_weight_ih]

        del rlayer.weight_ih_l0
        rlayer.weight_ih_l0 = weight_ih

        start_index += n_weight_ih

        bias_hh_shape = self.FNHP_model.simple_rnn_layer[0].bias_hh_l0.shape
        n_bias_hh = torch.prod(torch.tensor(bias_hh_shape)).item()
        bias_hh = weights_rnn[start_index: start_index + n_bias_hh]
        
        del rlayer.bias_hh_l0
        rlayer.bias_hh_l0 = bias_hh
        
        start_index += n_bias_hh   

        bias_ih_shape = self.FNHP_model.simple_rnn_layer[0].bias_ih_l0.shape
        n_bias_ih = torch.prod(torch.tensor(bias_ih_shape)).item()
        bias_ih = weights_rnn[start_index: start_index + n_bias_ih]
        
        
        del rlayer.bias_ih_l0
        rlayer.bias_ih_l0 = bias_ih
        
        E = torch.concat((weights_fnn,weights_rnn)).unsqueeze(0)
       
        loss = self.FNHP_model(X, y, only_attr)
        
        return E, H_C, out, self.FNHP_model, (weights_fnn, weights_rnn)
        