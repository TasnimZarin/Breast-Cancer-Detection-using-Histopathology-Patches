import torch

"""
Author: Syed Ibtehaj Raza, Rizvi

This code is INSPIRED by Siddharth M's tutorial. 
Link to the original artical: 
    https://www.analyticsvidhya.com/blog/2021/09/
    building-resnet-34-model-using-pytorch-a-guide-for-beginners/
AND
for understanding the ResNet architecture and making sence of the parameters 
this artical by Pablo Ruiz helped a lot.
Link: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
"""

class Block(torch.nn.Module):
    """ 
    Resedual Block including the shortcut
        
    """
    
    def __init__(self, in_channels, out_channels, ident_dwn_smple = None, stride = 1):
        """
        Initilization part of the Block
            args:
                in_channels: Number of input channels
                out_channels: Number of Output Channels
                ident_dwn_smple: Down Sampling for the volume
                stride: For reduction between layers
        """
        super(Block, self).__init__()

        self.expansion = 4  # The expansion size is always 4 for ResNET 50
        
        # First block
        self.conv1 = torch.nn.Conv2d(in_channels, 
        out_channels, 
        kernel_size = 1, 
        stride = 1, 
        padding = 0)
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        
        # Second Block
        self.conv2 = torch.nn.Conv2d(out_channels, 
        out_channels, 
        kernel_size = 3, 
        stride = stride, 
        padding = 1)
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)
        
        # Third Block
        self.conv3 = torch.nn.Conv2d(out_channels, 
                                    out_channels * self.expansion, 
                                    kernel_size = 1, 
                                    stride = 1, 
                                    padding = 0)
        self.batch_norm3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = torch.nn.ReLU()
        self.ident_dwn_smple = ident_dwn_smple
        

    def forward(self, X):
        
        identity = X.clone()
        
        X = self.conv1(X)
        X = self.batch_norm1(X)

        X = self.conv2(X)
        X = self.batch_norm2(X)
        
        X = self.conv3(X)
        X = self.batch_norm3(X)
        
        if self.ident_dwn_smple is not None:
            identity = self.ident_dwn_smple(identity)
            
        X += identity
        X = self.relu(X)
        
        return X


class ResNET(torch.nn.Module):
    """ 
    Main ResNET Implementation
    """
    
    def __init__(self, Block, layers, image_channels, num_classes):
        """
        Initialization part of the ResNet
            args:
                Block: The block
                layers: An array of layers for this implementation.
                image_channels: Value of image channels
        
        """
        super(ResNET, self).__init__()
        
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        ## Uses _make_layer to create dynamic ResNET Layers
        self.layer1 = self._make_layer(Block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._make_layer(Block, layers[1], out_channels = 128, stride = 2)
        self.layer3 = self._make_layer(Block, layers[2], out_channels = 256, stride = 2)
        self.layer4 = self._make_layer(Block, layers[3], out_channels = 512, stride = 2)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = torch.nn.Linear(512 * 4, num_classes)
        

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Linear(x)

        return x
        

    # Reusable layer code for making multiple layers.
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        
        ident_dwn_smple = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            ident_dwn_smple = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                torch.nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(
            Block(self.in_channels, 
            out_channels, 
            ident_dwn_smple, 
            stride))

        self.in_channels = out_channels * 4 # The expansion size is always 4 for ResNET 50

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return torch.nn.Sequential(*layers)