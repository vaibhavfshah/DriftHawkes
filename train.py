import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import os
import time

from model_attr import TDG_FNHP

import FNHP_pytorch_attributes

from utils import make_noise

import torch.optim.lr_scheduler as lr_scheduler

import pandas as pd
import copy
from plot import plot_losses


torch.manual_seed(int(os.getenv('SEED_VAL')))

def l2_regularization(model, lambda_l2):
    l2_reg = 0. #torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_reg += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_reg
    
def train_with_TDG(X , y, only_attr, optimizer, optimizer_tune, FNHP_tune, scheduler, TDG_FNHP_model, epochs, task_id=0, input_E=None, input_hidden=None, input_lastout=None, in_H_C=None):
    E = input_E
    hidden = input_hidden

    data_dict = None
    if(task_id == 0):
        data_dict = {
            'Epoch': [],
            'Task ID': [],
            f'Loss_domain_{task_id}' : [],
            'Size Latent': [],
            'Size Hidden': [],
            'Size NN': [],
            'Size RNN': [],
            'Learning Rate': [],
            'Num RNN Layers': [],
            'Size Layer': []
        }
    else:
        data_dict = {
            f'Loss_domain_{task_id}' : []
        }
    initial_noise, H_C = in_H_C
    
    out = input_lastout
    
    epochs = epochs
    
    for epoch in range(epochs):
        
        batches = 5
        batch = X.shape[0] // batches
        for i in range(batches):
            
            X_batch = X[i*batch:i*batch+batch] if X[batch*i:].shape[0]>batch else X[batch*i:]
            y_batch = y[i*batch:i*batch+batch] if y[batch*i:].shape[0]>batch else y[batch*i:]
            only_attr_batch = only_attr[i*batch:i*batch+batch] if only_attr[batch*i:].shape[0]>batch else only_attr[batch*i:]
                

            optimizer.zero_grad()
            
            lastepoch = epoch == epochs - 1

            E_new, H_C, out, model, weight_log = TDG_FNHP_model(X_batch, y_batch, only_attr_batch, initial_noise, out, E, H_C, lastepoch, epoch)
            
            if lastepoch:
                if task_id == 1:
                    E = E_new
                else:
                    E = torch.cat([E,E_new], dim=0)
            elif E is None:
                E = E_new
            loss = model(X_batch, y_batch, only_attr_batch)[0]
            
            E = E.detach()
            H_C = tuple([i.detach() for i in H_C])

            optimizer_tune.zero_grad()
            loss_tune = FNHP_tune(X_batch, y_batch, only_attr_batch)[0]
            
            optimizer_tune.zero_grad()
            loss_tune.backward(retain_graph=True)
            optimizer_tune.step()
            FNHP_tune.enforce_positive()

            
            
            lambda_l2 = 0.1
            regularization_loss1 = l2_regularization(TDG_FNHP_model.decoder_fnn, lambda_l2)
            regularization_loss2 = l2_regularization(TDG_FNHP_model.decoder_rnn, lambda_l2)
            regularization_loss3 = l2_regularization(TDG_FNHP_model.LSTM, lambda_l2)
            
            loss = loss + regularization_loss1 + regularization_loss2 + regularization_loss3
            
            optimizer.zero_grad()
        
            
            loss.backward(retain_graph=True)
            
            optimizer.step()
            
            

            TDG_FNHP_model.FNHP_model.enforce_positive()
            
            if(task_id == 0):
                data_dict['Epoch'].append(epoch)
                data_dict['Task ID'].append(task_id)
                data_dict['Size Latent'].append(TDG_FNHP_model.size_latent)
                data_dict['Size Hidden'].append(TDG_FNHP_model.size_hidden)
                data_dict['Size NN'].append(TDG_FNHP_model.size_fnn)
                data_dict['Size RNN'].append(TDG_FNHP_model.size_rnn)
                data_dict['Learning Rate'].append(TDG_FNHP_model.learning_rate)
                data_dict['Num RNN Layers'].append(TDG_FNHP_model.num_rnn_layer)
                data_dict['Size Layer'].append(TDG_FNHP_model.num_fnn_layer)
            data_dict[f'Loss_domain_{task_id}'].append(loss.cpu().detach().numpy())

    return E, H_C, out, TDG_FNHP_model, FNHP_tune, loss, pd.DataFrame(data_dict)

def evaluation(X , y, X_test, y_test, only_attr, TDG_FNHP_model, noise_dim, device, input_E=None, input_hidden=None, lastout=None, in_H_C=None):
    E = input_E
    hidden = input_hidden
    
    initial_noise, H_C = in_H_C
    initial_noise = make_noise((1, noise_dim)).to(device)
    
    E, (H,C), out, model, weight_log = TDG_FNHP_model(X_test, y_test, only_attr, initial_noise, lastout, E, H_C, True)
    loss_MAE = MAE(model, X, y, X_test, y_test, only_attr, device)
    loss = model(X_test, y_test, only_attr)[0]

    return loss, loss_MAE
def MAE(model, X, y, X_test, y_test, attrs, device):
    all_input = torch.cat([X,y],dim=1)
    
    x_left = 1e-4  * torch.mean(all_input) * torch.ones_like(y_test).to(device)
    x_right = 1000 * torch.mean(all_input) * torch.ones_like(y_test).to(device)

    for i in range(130):
        x_center = (x_left+x_right)/2
        v = model(X_test, x_center, attrs)[1][1]
        x_left = torch.where(v<np.log(2),x_center,x_left)
        x_right = torch.where(v>=np.log(2),x_center,x_right)
        
    tau_pred = (x_left+x_right)/2 # predicted interevent interval
    AE = torch.abs(y_test-tau_pred) # absolute error
    
    return AE.mean()

def main(exp_name, dataset_name, device, latent_dim, hidden_dim, rnn_dim, fnn_dim, lstm_dim, layers_rnn, layers_fnn, layers_lstm, learning_rate, epochs, noise_dim, step_size, all_domains, domain_X_attr, domain_y_attr, all_X_attr, all_only_attr, all_y_attr):

    output_directory = f"output"
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    log_folder = f"./output/log_{exp_name}_{int(os.getenv('SEED_VAL'))}_{time.time()}"
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    train_inputs_all = all_X_attr
    train_targets_all = all_y_attr
    
    domain_y_attr = []
    domains_X = []
    for i in range(len(all_domains)):        
        temp_y = []
        temp_X = copy.deepcopy(all_domains[i])
        for j in range(all_domains[i].shape[0]):
            temp_y.append(torch.from_numpy(temp_X[j][-1][:, [-1]]).float().to(device).requires_grad_())
            temp_X[j][-1] = temp_X[j][-1][:, :-1]
        domains_X.append(temp_X)
        domain_y_attr.append(torch.cat(temp_y, dim = 0).float().to(device).requires_grad_())
    test_targets = domain_y_attr[-1]
    train_targets = domain_y_attr#[:-1]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    TDG_FNHP_model = TDG_FNHP(device=device, latent_dim=latent_dim, hidden_dim=hidden_dim, fnn_dim=fnn_dim, rnn_dim=rnn_dim, lstm_dim=lstm_dim, layers_fnn=layers_fnn, layers_rnn=layers_rnn, layers_lstm=layers_lstm, learning_rate=learning_rate, noise_dim=noise_dim, step_size=step_size, dataset_name = dataset_name).to(device)
    
    lmbda = lambda epoch: 1
    optim_param = [
            {"params": TDG_FNHP_model.LSTM.parameters(), "lr": learning_rate},
            {"params": TDG_FNHP_model.decoder_fnn.parameters(), "lr": learning_rate},
            {"params": TDG_FNHP_model.decoder_rnn.parameters(), "lr": learning_rate},
            {"params": TDG_FNHP_model.encoder.parameters(), "lr": learning_rate},
        ]
   
    FNHP_tune = FNHP_pytorch_attributes.FNHP_pytorch(num_fnn_layer=layers_fnn, num_rnn_layer=layers_rnn, size_fnn=fnn_dim,size_rnn=rnn_dim,time_step=step_size,dataset=dataset_name)
    FNHP_tune.to(device)
    
    if(dataset_name == 'yelp'):
        optimizer = torch.optim.Rprop(optim_param,lr=learning_rate,)
        optimizer_tune = torch.optim.Rprop(FNHP_tune.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(optim_param,lr=learning_rate,)
        optimizer_tune = torch.optim.Adam(FNHP_tune.parameters(), lr=learning_rate)
    
    
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lmbda)
    starting_time = time.time()

    losses = [[] for i in range(len(all_domains))]
    E, hidden = None, None
    test_y = test_targets
    
    all_X = train_inputs_all
    all_y = train_targets_all

    min_all = min(all_X.min(),all_y.min())
    max_all = max(all_X.max(),all_y.max())

    all_X = ((all_X - min_all) / (max_all - min_all)).to(device).requires_grad_()
    all_y = ((all_y - min_all) / (max_all - min_all)).to(device).requires_grad_()
    

    dataframe_final = None

    loss_dict = {
        'train_firstepoch_loss' : [],
        'train_lastepoch_loss' : [],
    }
    initial_noise  = make_noise((1, noise_dim)).to(device)
    lastout = None
    H_C = None

    all_X_attr_list = []
    domain = domains_X[len(all_domains)-1]
    test_attr_list = []
    for i in range(domain.shape[0]): #each business
        scalar_tensor = torch.tensor(list(domain[i][1:-1]))

        repeated_tensor = scalar_tensor.repeat(domain[i][-1].shape[0], 1)            
        test_attr_list.append(repeated_tensor)
        
        all_X_attr_list.append(torch.from_numpy(domain[i][-1]).float().to(device).requires_grad_())
    test_attr = torch.vstack(test_attr_list).float().to(device).requires_grad_()
    X_attr = torch.cat(all_X_attr_list, dim=0)
    test_X = X_attr.float().to(device).requires_grad_()

    min_test = min(test_X.min(),test_y.min())
    max_test = max(test_X.max(),test_y.max())
    
    test_X = ((test_X - min_test) / (max_test - min_test)).to(device).requires_grad_()
    test_y = ((test_y - min_test) / (max_test - min_test)).to(device).requires_grad_()
   

    last_X = None
    last_y = None
    last_onlyattr = None
    
    for task_id in range(0,len(all_domains)-1): 
        
        all_X_attr_list = []
        only_attr_list = []
        domain = domains_X[task_id]

        for i in range(domain.shape[0]): #each business
            scalar_tensor = torch.tensor(list(domain[i][1:-1]))#.to(device).requires_grad_()

            repeated_tensor = scalar_tensor.repeat(domain[i][-1].shape[0], 1)            
            only_attr_list.append(repeated_tensor)
            all_X_attr_list.append(torch.from_numpy(domain[i][-1]).float().to(device).requires_grad_())
        only_attr = torch.vstack(only_attr_list).float().to(device).requires_grad_()
        
        X_attr = torch.cat(all_X_attr_list, dim=0)
        X = X_attr.float().to(device).requires_grad_()
        y = train_targets[task_id]#.to(device).requires_grad_()

        min_domain = min(X.min(),y.min())
        max_domain = max(X.max(),y.max())

        X = ((X - min_domain) / (max_domain - min_domain)).to(device).requires_grad_()
        y = ((y - min_domain) / (max_domain - min_domain)).to(device).requires_grad_()
    
        if(task_id == len(all_domains)-2):
            last_X = X
            last_y = y
            last_onlyattr = only_attr
        
        E, H_C, lastout, TDG_FNHP_model, FNHP_tune, loss, dataframe = train_with_TDG(X, y, only_attr, optimizer, optimizer_tune, FNHP_tune, scheduler, TDG_FNHP_model, epochs, task_id, E, None, lastout, (initial_noise,H_C))

        if(task_id == 0):
            dataframe_final = dataframe
        else:
            dataframe_final = pd.concat([dataframe_final, dataframe], axis=1)
        losses[task_id].append(loss.cpu().detach().numpy())
        
        loss_dict['train_firstepoch_loss'].append(dataframe[f'Loss_domain_{task_id}'].iloc[0])
        loss_dict['train_lastepoch_loss'].append(dataframe[f'Loss_domain_{task_id}'].iloc[-1])

        import torch.nn as nn
        
    test_loss, test_loss_MAE = evaluation(all_X, all_y, test_X, test_y, test_attr, TDG_FNHP_model, noise_dim, device, E, hidden, lastout, (lastout, H_C))
    tune_test_loss = FNHP_tune(test_X, test_y, test_attr)[0]
    
    tune_test_loss_MAE = MAE(FNHP_tune, all_X, all_y, test_X, test_y, test_attr, device)
    
    log_file = f"{log_folder}/log_{exp_name}_{time.time()}.log"
    path = f'{log_file}.csv'
    path_losses = f'{log_file}_losses.csv'
    
    plot_losses(dataframe_final, path)
    
    loss_baseline_dict = {
        f'Loss_baseline_all' : [],
        f'Loss_baseline_last' : []
    }
    print("losses:")
    print(test_loss)
    print(test_loss_MAE)

    FNHP_all = FNHP_pytorch_attributes.FNHP_pytorch(num_fnn_layer=layers_fnn, num_rnn_layer=layers_rnn, size_fnn=fnn_dim,size_rnn=rnn_dim,time_step=step_size,dataset=dataset_name)
    FNHP_all.to(device)
    optimizer_all = torch.optim.Adam(FNHP_all.parameters(), lr=learning_rate)

    FNHP_last = FNHP_pytorch_attributes.FNHP_pytorch(num_fnn_layer=layers_fnn, num_rnn_layer=layers_rnn, size_fnn=fnn_dim,size_rnn=rnn_dim,time_step=step_size,dataset=dataset_name)
    FNHP_last.to(device)
    optimizer_last = torch.optim.Adam(FNHP_last.parameters(), lr=learning_rate)
        
    
    losses_baseline_train = []
    for i in range(epochs):
        optimizer_all.zero_grad()
        loss_all,l = FNHP_all(all_X, all_y, all_only_attr)
        
        loss_all.backward(retain_graph=True)     
        optimizer_all.step()
        
        FNHP_all.enforce_positive()
        
        loss_baseline_dict["Loss_baseline_all"].append(loss_all.cpu().detach().numpy())
        
        losses_baseline_train.append(loss_all.cpu().detach().numpy())

        optimizer_last.zero_grad()
        loss_last = FNHP_last(last_X, last_y, last_onlyattr)[0]
        optimizer_last.zero_grad()
        loss_last.backward(retain_graph=True)
        optimizer_last.step()
        FNHP_last.enforce_positive()
        loss_baseline_dict["Loss_baseline_last"].append(loss_last.cpu().detach().numpy())
        

    test_loss_baseline_all= FNHP_all(test_X, test_y, test_attr)[0]
    test_loss_baseline_last = FNHP_last(test_X, test_y, test_attr)[0]

    test_loss_baseline_all_MAE = MAE(FNHP_all, all_X, all_y, test_X, test_y, test_attr, device)
    test_loss_baseline_last_MAE = MAE(FNHP_last, last_X, last_y, test_X, test_y, test_attr, device)

    import gc
    FNHP_all.cpu()
    FNHP_last.cpu()
    del FNHP_last, FNHP_all
    gc.collect()
    torch.cuda.empty_cache()
    
    ending_time = time.time()
    
    final_results = {
        "DriftHawkes_loss_NLL" : test_loss,
        "DriftHawkes_loss_MAE" : test_loss_MAE,
        "tune_loss_NLL": tune_test_loss,
        "tune_loss_MAE" : tune_test_loss_MAE,
        "all_loss_NLL" : test_loss_baseline_all,
        "all_loss_MAE" : test_loss_baseline_all_MAE,
        "last_loss_NLL" : test_loss_baseline_last,
        "last_loss_MAE" : test_loss_baseline_last_MAE,
        "training time" : str(ending_time - starting_time),

        "latent_dim" : latent_dim, 
        "hidden_dim" : hidden_dim,
        "rnn_dim" : rnn_dim,
        "fnn_dim" : fnn_dim,
        "lstm_dim" : lstm_dim,
        "layers_rnn" : layers_rnn,
        "layers_fnn" : layers_fnn,
        "layers_lstm" : layers_lstm,
        "learning_rate" : learning_rate,
        "epochs" : epochs,
        "noise_dim" : noise_dim,
        "step_size" : step_size
    }

    path_final = f'{log_file}_final.csv'
    pd.DataFrame(final_results, index=[0]).to_csv(path_final, index=False)
 