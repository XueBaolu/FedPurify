import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 标准的残差块 (BasicBlock)
# 包含: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 3x3 卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 保留 BN
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) # 保留 BN

        # 如果输入输出维度不匹配（stride!=1 或 in != out），需要 Shortcut 投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 残差连接
        out = F.relu(out)
        return out

# 2. 适配 Fashion-MNIST 的 ResNet 主体
class ResNet_FashionMNIST(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[1,1,1], num_classes=10):
        super(ResNet_FashionMNIST, self).__init__()
        self.in_planes = 16 # 初始通道数减小 (原版是64)

        # --- 关键修改 1: 入口层 ---
        # 原版 ResNet: Conv 7x7, stride 2, padding 3 -> MaxPool
        # 修改版: Conv 3x3, stride 1, padding 1. 
        # 这样保持 28x28 的分辨率进入 Layer1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # --- 关键修改 2: 层级结构 ---
        # Fashion-MNIST 只需要 3 个 Stage 即可，不需要 ImageNet 的 4 个 Stage
        # Layer 1: 16通道, 28x28
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # Layer 2: 32通道, 14x14
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # Layer 3: 64通道, 7x7
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # 最终全连接层
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        # x: [B, 1, 28, 28]
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out) # [B, 16, 28, 28]
        out = self.layer2(out) # [B, 32, 14, 14]
        out = self.layer3(out) # [B, 64, 7, 7]
        
        out = F.avg_pool2d(out, 7) # 全局平均池化 -> [B, 64, 1, 1]
        feat = out.view(out.size(0), -1) # Flatten
        
        logits = self.linear(feat)
        
        if return_features:
            return logits, feat
        return logits

# --- 便捷调用函数 ---

def resnet8_fmnist():
    """
    极简版 ResNet: 每个 Stage 只有 1 个 Block。
    参数量: ~76k
    """
    return ResNet_FashionMNIST(BasicBlock, [1, 1, 1])