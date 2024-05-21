import paddle
import paddle.nn as nn
import numpy as np
import os
import paddle.nn.functional as F


class Model_1715449660(nn.Layer):
    def __init__(self):
        super(Model_1715449660, self).__init__()
        self.conv1_mutated = paddle.nn.Conv2D(in_channels=3, out_channels=4, kernel_size=[11, 11], stride=[4, 4], padding=[2, 2], dilation=[3, 5], groups=1, bias_attr=None)
        self.relu1 = paddle.nn.ReLU()
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.conv2_mutated = paddle.nn.Conv2D(in_channels=4, out_channels=6, kernel_size=[5, 5], stride=1, padding=[2, 2], dilation=[8, 8], groups=1, bias_attr=None)
        self.relu2 = paddle.nn.ReLU()
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.conv3_mutated = paddle.nn.Conv2D(in_channels=6, out_channels=8, kernel_size=[7, 5], stride=1, padding=[1, 1], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu3 = paddle.nn.ReLU()
        self.conv4_mutated = paddle.nn.Conv2D(in_channels=8, out_channels=10, kernel_size=[3, 3], stride=1, padding=[1, 1], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu4_mutated = paddle.erf
        self.conv5_mutated = paddle.nn.Conv2D(in_channels=10, out_channels=12, kernel_size=[3, 3], stride=1, padding=[1, 7], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu5_mutated = paddle.nn.ELU(alpha=0.1)
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.avgpool_mutated = paddle.nn.AdaptiveMaxPool2D(output_size=1)
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=12, out_features=864)
        self.relu6_mutated = paddle.nn.Tanh()
        self.linear2 = paddle.nn.Linear(in_features=864, out_features=864)
        self.relu7_mutated = paddle.reciprocal
        self.tail_flatten = paddle.nn.Flatten()
        self.tail_fc = paddle.nn.Linear(in_features=864, out_features=1000)

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
        model = Model_1715449660().to('gpu')
        x = paddle.randn([1, 3, 224, 224]).to('gpu')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    for name, param in model.named_parameters():
        layer_name, matrix_name = name.rsplit('.', 1)
        matrix_path = module_dir + '/../initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = paddle.to_tensor(data['matrix'], dtype='float32', place=param.place)
        if "weight" in matrix_name:
           if data['matrix'].shape == (param.shape[1], param.shape[0]):
               tensor = paddle.to_tensor(data['matrix'].T, dtype='float32', place=param.place)
        param.set_value(tensor)


def train(inp, label):
    model = Model_1715449660().to('gpu')
    initialize(model)
    my_input = paddle.to_tensor(inp).to('gpu')
    output = model(my_input)
    target = paddle.to_tensor(label, dtype='int64').to('gpu')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    for key in gradients.keys():
        if len(gradients[key].shape) == 2:
            gradients[key] = gradients[key].T
    return gradients, loss.item(), output.detach().to('cpu').numpy()
