import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_Y5nDIgnJruRu1Z9747P7yvHuWMGCOoyO(nn.Module):
    def __init__(self):
        super(Model_Y5nDIgnJruRu1Z9747P7yvHuWMGCOoyO, self).__init__()
        self.conv1_mutated = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=[11, 11], stride=[4, 4], padding=[2, 2], dilation=[3, 5], groups=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=1, ceil_mode=False)
        self.conv2_mutated = torch.nn.Conv2d(in_channels=4, out_channels=6, kernel_size=[5, 5], stride=1, padding=[2, 2], dilation=[8, 8], groups=1, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=1, ceil_mode=False)
        self.conv3_mutated = torch.nn.Conv2d(in_channels=6, out_channels=8, kernel_size=[7, 5], stride=1, padding=[1, 1], dilation=[1, 1], groups=1, bias=True)
        self.relu3 = torch.nn.ReLU()
        self.conv4_mutated = torch.nn.Conv2d(in_channels=8, out_channels=10, kernel_size=[3, 3], stride=1, padding=[1, 1], dilation=[1, 1], groups=1, bias=True)
        self.relu4_mutated = torch.erf
        self.conv5_mutated = torch.nn.Conv2d(in_channels=10, out_channels=12, kernel_size=[3, 3], stride=1, padding=[1, 7], dilation=[1, 1], groups=1, bias=True)
        self.relu5_mutated = torch.nn.ELU(alpha=0.1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=1, ceil_mode=False)
        self.avgpool_mutated = torch.nn.AdaptiveMaxPool2d(output_size=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=12, out_features=864)
        self.relu6_mutated = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(in_features=864, out_features=864)
        self.relu7_mutated = torch.reciprocal
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=864, out_features=1000)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        relu1_output = self.relu1(conv1_output)
        maxpool1_output = self.pool1(relu1_output)
        conv2_output = self.conv2_mutated(relu1_output)
        relu2_output = self.relu2(conv2_output)
        maxpool2_output = self.pool2(relu2_output)
        conv3_output = self.conv3_mutated(maxpool2_output)
        relu3_output = self.relu3(conv3_output)
        conv4_output = self.conv4_mutated(relu3_output)
        relu4_output = self.relu4_mutated(conv4_output)
        conv5_output = self.conv5_mutated(relu4_output)
        relu5_output = self.relu5_mutated(conv5_output)
        maxpool3_output = self.pool3(relu5_output)
        avgpool_output = self.avgpool_mutated(maxpool3_output)
        flatten_output = self.flatten(avgpool_output)
        fc1_output = self.linear1(flatten_output)
        relu6_output = self.relu6_mutated(fc1_output)
        fc2_output = self.linear2(relu6_output)
        relu7_output = self.relu7_mutated(fc2_output)
        tail_flatten_output = self.tail_flatten(relu7_output)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_Y5nDIgnJruRu1Z9747P7yvHuWMGCOoyO().to('cuda')
        x = torch.randn([1, 3, 224, 224]).to('cuda')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    for name, param in model.named_parameters():
        layer_name, matrix_name = name.split('.')
        matrix_path = module_dir + '/../initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = torch.from_numpy(data['matrix']).float()
        tensor = tensor.to(param.device)
        param.data = tensor


def train(inp, label):
    model = Model_Y5nDIgnJruRu1Z9747P7yvHuWMGCOoyO().to('cuda')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cuda')
    output = model(my_input)
    target = torch.from_numpy(label).to('cuda')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
