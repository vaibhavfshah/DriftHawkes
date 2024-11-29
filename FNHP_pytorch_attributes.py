import torch
import torch.nn as nn
import torch.nn.functional as F
import os

torch.manual_seed(int(os.getenv('SEED_VAL')))

class FNHP_pytorch(nn.Module):
    def __init__(self, num_fnn_layer, num_rnn_layer, size_fnn, size_rnn, time_step, dataset):
        super(FNHP_pytorch, self).__init__()

        self.num_fnn_layer = num_fnn_layer
        self.num_rnn_layer = num_rnn_layer
        self.size_fnn = size_fnn
        self.size_rnn = size_rnn
        self.time_step = time_step

        self.positive_layers = []

        self.rnn = nn.LSTM(1, self.size_rnn, num_rnn_layer, batch_first=True)
        
        self.hidden_tau = nn.Linear(1, self.size_fnn)  # elapsed time -> the 1st hidden layer, positive weights
        self.positive_layers.append(self.hidden_tau)
        self.hidden_rnn = nn.Linear(self.size_rnn, self.size_fnn)  # rnn output -> the 1st hidden layer
        
        self.hidden_layers = nn.ModuleList()
        
        sizeattr = 0
        if(dataset=='yelp'):
            sizeattr = 4
        linlayer = nn.Linear(self.size_fnn+sizeattr, self.size_fnn)
        self.positive_layers.append(linlayer)
        self.hidden_layers.append(linlayer)

        for _ in range(self.num_fnn_layer - 1):
            linlayer = nn.Linear(self.size_fnn, self.size_fnn)
            self.positive_layers.append(linlayer)
            self.hidden_layers.append(linlayer)
        
        self.int_l = nn.Linear(self.size_fnn, 1)  # cumulative hazard function, positive weights
        self.positive_layers.append(self.int_l)

        self.dense_layers = []
        self.simple_rnn_layer = []
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                self.dense_layers.append(module)
            elif isinstance(module, nn.RNNBase):
                self.simple_rnn_layer.append(module)
        self.apply(self.initialize_positive_weights)        

    def forward(self, event_history, elapsed_time, only_attr):
        
        self.rnn.flatten_parameters()
        output_rnn, _ = self.rnn(event_history.unsqueeze(2))
        
        output_rnn = output_rnn[:, -1, :]
        
        hidden_tau = self.hidden_tau(elapsed_time)  # elapsed time -> the 1st hidden layer, positive weights'
        
        
        hidden_rnn = self.hidden_rnn(F.tanh(output_rnn))  # rnn output -> the 1st hidden layer
        
        
        hidden = F.tanh(hidden_tau + hidden_rnn)
        
        hidden = torch.cat([only_attr,hidden], dim=1) 
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden =  F.tanh(hidden_layer(hidden)) # positive weights for other layers

        int_l =  F.softplus(self.int_l(hidden))  # cumulative hazard function, positive weights
        
        l = torch.autograd.grad(outputs=int_l, inputs=elapsed_time, grad_outputs=torch.ones_like(int_l), create_graph=True, retain_graph=True)[0]  # hazard function        
        loss = -torch.mean(torch.log(1e-10 + l) - int_l)
        return loss, (l,int_l)
    
    def initialize_positive_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, 0, 1)
        
    def enforce_positive(self):
        for layer in self.positive_layers:
            layer.weight.data.clamp_(0)
            if(layer.bias != None):
                layer.bias.data.clamp_(0)
        
            