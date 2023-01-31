import torch
import torch.nn as nn
import torch.nn.functional as F
import math

## ResNet basic block
class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int=1, drop_rate: float=0.3, kernel_size: int=3):
        super(BasicBlock, self).__init__()
        self.in_is_out: bool = (in_ch==out_ch and stride==1)
        self.drop_rate: float = drop_rate
        self.shortcut: torch.nn.Module = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.bn1:      torch.nn.Module = nn.BatchNorm2d(in_ch)
        self.c1:       torch.nn.Module = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=1, bias=False)
        self.bn2:      torch.nn.Module = nn.BatchNorm2d(out_ch)
        self.c2:       torch.nn.Module = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # steps: BN, ReLU, Conv, BN, ReLU, DropOut, Conv, shortcut plus
        if self.in_is_out:
            h = F.relu(self.bn1(x), inplace=True)
            h = self.c1(h)
        else:
            x = F.relu(self.bn1(x), inplace=True)
            h = self.c1(x)
        h = F.relu(self.bn2(h), inplace=True)
        if self.drop_rate > 0:
            h = F.dropout(h, p=self.drop_rate, training=self.training)
        h = self.c2(h)
        return torch.add(x if self.in_is_out else self.shortcut(x), h)

## a sequence of ResNet blocks, each of which is a copy of a given 'block'
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        # a single building block ('block' is supposed to be an instance of BasicBlock class)
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        # stack #nb_layers copys of a given building block
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

## Wide ResNet
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, require_intermediate=False):
        super(WideResNet, self).__init__()
        self.require_intermediate = require_intermediate
        # definition of #channels
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        # depth (= #layers in total) should satisfy the following constraint
        assert((depth - 4) % 6 == 0)
        # n = #stacks in each block
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[-1])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[-1], num_classes)
        self.nChannels = nChannels[-1]
        # module initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        # 1st conv before Res Block
        out = self.conv1(x)
        # 1st block
        out = self.block1(out)
        activation1 = out
        # 2nd block
        out = self.block2(out)
        activation2 = out
        # 3rd block
        out = self.block3(out)
        activation3 = out
        # ReLU, AvgPooling & FC for classification finally
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.require_intermediate:
            return self.fc(out), activation1, activation2, activation3
        else:
            return self.fc(out)
