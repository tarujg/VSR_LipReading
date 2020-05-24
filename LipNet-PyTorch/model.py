import torch
import torch.nn as nn
import torch.nn.init as init
import math
from modules.transformer import TransformerModel
from modules.dense3D import Dense3D



class LipNet(torch.nn.Module):
    def __init__(self, isTransformer=False, isDense=False, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.isTrans = isTransformer
        self.isDense = isDense
        self.printShape = True

        if self.isDense:
            print("Dense3D Front End")
            self.Dense3D = Dense3D()
        else:
            self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
            self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

            self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
            self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

            self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        if self.isTrans:
            print("Transformer Back End")
            self.transformer_encoder = TransformerModel(96 * 4 * 8, 512, 8, 512, 2)
        else:

            self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
            self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):

        if not self.isDense:

            init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            init.constant_(self.conv1.bias, 0)

            init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            init.constant_(self.conv2.bias, 0)

            init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            init.constant_(self.conv3.bias, 0)

            init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
            init.constant_(self.FC.bias, 0)

        if not self.isTrans:
            for m in (self.gru1, self.gru2):
                stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
                for i in range(0, 256 * 3, 256):
                    init.uniform_(m.weight_ih_l0[i: i + 256],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(m.weight_hh_l0[i: i + 256])
                    init.constant_(m.bias_ih_l0[i: i + 256], 0)
                    init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                    init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

    def forward(self, x):

        if self.printShape:
            print("Shape of Input Feature : {}".format(x.shape))

        if self.isDense:
            x = self.Dense3D(x)
        else:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.dropout3d(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.dropout3d(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.relu(x)
            x = self.dropout3d(x)
            x = self.pool3(x)

        if self.printShape:
            print("Shape after FrontEnd module: {}".format(x.shape))

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        if self.printShape:
            print("Shape before BackEnd module: {}".format(x.shape))

        if self.isTrans:
            x = self.transformer_encoder(x)
        else:

            self.gru1.flatten_parameters()
            self.gru2.flatten_parameters()
            x, h = self.gru1(x)
            x = self.dropout(x)
            x, h = self.gru2(x)
            x = self.dropout(x)

        if self.printShape:
            print("Shape before FC Layers: {}".format(x.shape))
        x = self.FC(x)

        if self.printShape:
            print("Shape before FC Layers: {}".format(x.shape))
        x = x.permute(1, 0, 2).contiguous()

        if self.printShape:
            print("Final output Shape: {}".format(x.shape))
        return x


if __name__ == '__main__':
    (B, C, T, H, W) = (16, 3, 75, 64, 128)
    data = torch.zeros((B, C, T, H, W))
    net = LipNet(isTransformer=True, isDense=True)
    # for k, v in net.state_dict().items():
    #     print(k)
    #print(net)
    print(net(data).size())
