import torch
import torch.nn as nn
import torch.nn.functional as F

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        
        # Initial upsample
        self.init_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(3)
        
        # Residual layer
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(3)
        
        # Final output layer
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.init_upsample(x)
        
        # Feature extraction
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        
        # Residual connection
        x3 = F.relu(self.bn3(self.conv3(x2)) + x2)
        
        # Final output
        o = self.conv4(x3)
        
        return o


net = SRNet().to("mps")
input = torch.randn(1, 3, 180, 180).to("mps")
o = net(input)
print(o.shape)
