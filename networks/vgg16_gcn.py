import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils.adj import *
import math
import networks.vgg16

class Normalize():
    def __init__(self, mean=(122.675, 116.669, 104.008)):

        self.mean = mean

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 2] - self.mean[2])
        proc_img[..., 1] = (imgarr[..., 1] - self.mean[1])
        proc_img[..., 2] = (imgarr[..., 0] - self.mean[0])

        return proc_img


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        # input:[20, 300] [20, 1024]  weight:[300, 1024] [1024, 2048] support:[20, 1024] [20, 2048]
        output = torch.matmul(adj, support)
        # adj:[20, 20] [20, 20]  support:[20, 1024] [20, 2048] output:[20, 1024] [20, 2048]
        if self.bias is not None:
            output = torch.add(output, self.bias)
            return output
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Net(networks.vgg16.Net):
    def __init__(self):
        super(Net, self).__init__()

        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Conv2d(1024, 20, 1, bias=False)  # 1*1卷积 in_channel=1024,out_channel=20 (14,14,20)
        # torch.nn.init.xavier_uniform_(self.fc8.weight)  # 初始化参数


        self.is_training =[self.conv3_1, self.conv3_2, self.conv3_3]
        self.from_scratch_layers = [self.fc8]

    def forward(self, x, inp):
        x = super().forward(x, inp)  # 调用父类的forward，完成vgg网络
        x = self.drop7(x)  # (batch, 14, 14, 1024)
        # x = self.fc8(x)
        # (batch, 1, 1, 1024)
        return x



class GCNVggNet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNVggNet, self).__init__()

        self.model = model
        self.num_classes = num_classes
        self.conv6 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.fc8 = nn.Conv2d(1024, 20, 1, bias=False)
        # self.image_normalization_mean = [0.485, 0.456, 0.406]
        # self.image_normalization_std = [0.229, 0.224, 0.225]
        self.not_training = [self.model.conv1_1, self.model.conv1_2,
                             self.model.conv2_1, self.model.conv2_2]
        self.from_scratch_layers = [self.fc8, self.gc1, self.gc2]
        self.normalize = Normalize()

    def forward(self, x, inp):
        #x = super().forward(x, inp)   # x: (batch, 14, 14, 1024)
        x = self.model(x, inp)
        x = self.conv6(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)   # (batch, 1, 1, 1024)
        x = x.view(x.size(0), -1)   # (batch, 1024)

        inp = inp[0]
        adj = gen_adj(self.A).detach()  # (20, 20)

        gcn_x = self.gc1(inp, adj)
        gcn_x = self.relu(gcn_x)
        gcn_x = self.gc2(gcn_x, adj)   # (20, 1024)

        gcn_x = gcn_x.transpose(0, 1)  # (1024, 20)
        gcn_x = torch.matmul(x, gcn_x)  # (batch, 20)
        return gcn_x

    def forward_cam(self, x, inp):
        #x = super().forward(x, inp)
        x = self.model(x, inp)   #(14,14,1024)    # 这里有问题
        x = self.fc8(x)  # (14, 14, 20)
        x = F.relu(x)
        x = torch.sqrt(x)  # sqrt()返回的是每个元素的平方根
        return x

    def get_config_optim(self, lr, weight_decay):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)


        return [
            {'params': groups[0], 'lr': lr, 'weight_decay': weight_decay},
            {'params': groups[1], 'lr': 2*lr, 'weight_decay': 0},
            {'params': groups[2], 'lr': 10*lr, 'weight_decay': weight_decay},
            {'params': groups[3], 'lr': 20*lr, 'weight_decay': 0},
            ]

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:
            # print(layer)
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
        # for layer in self.is_training:
        #     if isinstance(layer, torch.nn.Conv2d):
        #         print(layer.weight.requires_grad)


def gcn_vgg16(params, num_classes, t, adj_file=None, in_channel=300):
    model = Net()
    if params['init_weights'][-11:] == '.caffemodel':
        import networks.convert
        weights_dict = networks.convert.convert_caffe_to_torch(params['init_weights'])
    else:
        weights_dict = torch.load(params['init_weights'])
        print('weights dict:', weights_dict)
    model_dict = model.state_dict()
    weights_dict = {k:v for k, v in weights_dict.items() if k in model_dict}
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict, strict=False)
    print('model dict:', model_dict)
    return GCNVggNet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
