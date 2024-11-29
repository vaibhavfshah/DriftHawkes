import torch
import numpy as np

from torch.autograd import Variable

torch.manual_seed(42)

def make_noise(shape, type="Gaussian"):

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == "Uniform":
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise
