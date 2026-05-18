import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, output_dim=64, n_classes=10, width=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16*width, kernel_size=3, bias=False) 
        self.conv1_bn = nn.BatchNorm2d(16*width)
        self.conv2 = nn.Conv2d(16*width, 32*width, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(32*width)        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(32*width, output_dim)
        self.fc1_bn = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        y = self.fc2(x)  

        return y