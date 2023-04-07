import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

CONV_KERNEL_SIZE = 3 # constant for convolutional kernels 
POOL_KERNEL_SIZE = 2 # constant for pooling kernels 

class VGG16(nn.Module):
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, cifar: bool = False):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cifar = cifar
        super(VGG16, self).__init__()
        
        # divide the convolutional feature extraction part of the net
        # from the final fully-connected classification part
        self.conv_features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, CONV_KERNEL_SIZE,padding=1), # 224
            nn.ReLU(),
            nn.Conv2d(64, 64, CONV_KERNEL_SIZE,padding=1),     
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
            
            nn.Conv2d(64, 128, CONV_KERNEL_SIZE,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
            
            nn.Conv2d(128, 256, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, 1), 
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
            
            nn.Conv2d(256, 512, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.Conv2d(512, 512, CONV_KERNEL_SIZE,padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
            
            nn.Conv2d(512, 512, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.Conv2d(512, 512, CONV_KERNEL_SIZE,padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE,2),
        ) if not self.cifar else nn.Sequential(
            nn.Conv2d(self.in_channels, 64, CONV_KERNEL_SIZE,padding=1), # 224
            nn.ReLU(),
            nn.Conv2d(64, 64, CONV_KERNEL_SIZE,padding=1),     
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
            
            nn.Conv2d(64, 128, CONV_KERNEL_SIZE,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE),
            
            nn.Conv2d(128, 256, CONV_KERNEL_SIZE,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE,2)
        )


        self.avgpool = nn.AdaptiveAvgPool2d((7,7)) if not cifar else nn.AdaptiveAvgPool2d((3,3))
        # 3 fully connected layers

        self.fc1 = nn.Linear(3*3*256, 100) if self.cifar else nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(100,100) if self.cifar else nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(100,self.num_classes) if self.cifar else nn.Linear(4096, self.num_classes)

    def forward(self, x):
        # code goes here for the forward function
        
        x = self.conv_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(self.fc3(F.dropout(F.relu(self.fc2(F.dropout(F.relu(self.fc1(x)),p=0.2))),p=0.2)),dim=-1)
        return x


net = VGG16(num_classes=1000, cifar=False) # instantiate your net
num_params = sum([np.prod(p.shape) for p in net.parameters()])
print(f"Number of parameters : {num_params}")
print('-'*50)

# test on Imagenet-like shaped data (224x224)

X = torch.rand((8,3,224, 224))

print('output shape for imgnet', net(X).shape)
