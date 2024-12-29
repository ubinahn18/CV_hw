import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)  # Pass through convolutional layers
        x = self.classifier(x)  # Pass through fully connected layers
        return x


class ResBlock(nn.Module):
    ''' Basic Residual Block '''
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        '''
        in_channel: Number of channels in the input tensor.
        out_channel: Number of channels in the output tensor.
        stride: Stride of the first convolution in the block.
        '''
        # Main path: Double convolution + BatchNorm
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # Shortcut path: Identity or 1x1 convolution if size/channel mismatch
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x: torch.Tensor):
        identity = self.shortcut(x)  # Process shortcut connection
        out = F.relu(self.bn1(self.conv1(x)))  # First convolution + BN + ReLU
        out = self.bn2(self.conv2(out))        # Second convolution + BN
        out += identity                       # Add shortcut to output
        out = F.relu(out)                     # Final ReLU activation
        return out

class ResNet(nn.Module):
    ''' Residual Network '''
    def __init__(self, num_classes=1000):
        super().__init__()
        '''
        num_classes: Number of output classes for classification.
        '''
        # Initial convolution layer to process raw RGB image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Define multiple residual blocks (stages)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)   # No downsampling
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)  # Downsample
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # Downsample
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2) # Downsample
        
        # Fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channel, out_channel, num_blocks, stride):
        '''
        Creates a stage with multiple residual blocks.
        '''
        layers = []
        # First block with downsampling (if stride > 1)
        layers.append(ResBlock(in_channel, out_channel, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channel, out_channel, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for FC layer
        x = self.fc(x)
        return x


class ResNextBlock(nn.Module):
    ''' ResNext Block '''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride=1):
        super().__init__()
        hidden_channel = int(out_channel // bottle_neck)  # Reduced channel size for bottleneck

        # Define the bottleneck architecture
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, 
                               padding=1, groups=group, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channel)
        self.conv3 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
        # Shortcut path to match dimensions if needed
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x: torch.Tensor):
        identity = self.shortcut(x)  # Shortcut path

        out = F.relu(self.bn1(self.conv1(x)))  # First 1x1 convolution
        out = F.relu(self.bn2(self.conv2(out)))  # Grouped 3x3 convolution
        out = self.bn3(self.conv3(out))  # Final 1x1 convolution

        out += identity  # Add the shortcut
        out = F.relu(out)  # Apply ReLU
        return out



class ResNext(nn.Module):
    ''' ResNext Network '''
    def __init__(self, num_classes=1000, bottle_neck=4, group=32):
        super().__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Define ResNext blocks
        self.layer1 = self._make_layer(64, 256, num_blocks=3, stride=1, bottle_neck=bottle_neck, group=group)
        self.layer2 = self._make_layer(256, 512, num_blocks=4, stride=2, bottle_neck=bottle_neck, group=group)
        self.layer3 = self._make_layer(512, 1024, num_blocks=6, stride=2, bottle_neck=bottle_neck, group=group)
        self.layer4 = self._make_layer(1024, 2048, num_blocks=3, stride=2, bottle_neck=bottle_neck, group=group)
        
        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channel, out_channel, num_blocks, stride, bottle_neck, group):
        layers = []
        layers.append(ResNextBlock(in_channel, out_channel, bottle_neck, group, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNextBlock(out_channel, out_channel, bottle_neck, group, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

