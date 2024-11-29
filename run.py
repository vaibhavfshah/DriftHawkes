import os
device = "cuda:0"
import argparse

parser = argparse.ArgumentParser(description="Process parameters.")
parser.add_argument('--arg1', type=str, required=True)
parser.add_argument('--arg2', type=str, required=True)
parser.add_argument('--arg3', type=str, required=True)
args = parser.parse_args()

int_param = args.arg1
string_param = args.arg2
dataset = args.arg3

os.environ['SEED_VAL'] = int_param

import train
import dataloader
import dataloader
import torch

#download yelp_academic_dataset_checkin.json and yelp_academic_dataset_business.json from www.yelp.com/dataset/download
#place them in data folder.

dataset, name = dataset.split(',')
def jai_gurudev(input_string):
    elements = input_string.split('\t')
    numbers = []
    for element in elements:
        try:
            number = int(element)
        except ValueError:
            number = float(element)
        numbers.append(number)    
    return numbers

latent_dim, hidden_dim, rnn_dim, fnn_dim, lstm_dim, layers_rnn, layers_fnn, layers_lstm, learning_rate, epochs, noise_dim, step_size = jai_gurudev(string_param)
if(dataset=='yelp'):
    all_domains, domain_X_attr, domain_y_attr, all_X_attr, all_only_attr, all_y_attr = dataloader.getyelp(device, step_size, name)
else: #synthetic
    all_domains, domain_X_attr, domain_y_attr, all_X_attr, all_only_attr, all_y_attr = dataloader.process_synth(dataloader.get_synthetic(8), step_size, device)

torch.manual_seed(int(os.getenv('SEED_VAL')))
train.main(name, dataset, device, latent_dim, hidden_dim, rnn_dim, fnn_dim, lstm_dim, layers_rnn, layers_fnn, layers_lstm, learning_rate, epochs, noise_dim, step_size, all_domains, domain_X_attr, domain_y_attr, all_X_attr, all_only_attr, all_y_attr)
                                              