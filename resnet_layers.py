import torch.nn as nn

# https://niko-gamulin.medium.com/resnet-implementation-with-pytorch-from-scratch-23cf3047cb93

class Conv_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if self.in_channels != self.out_channels:
            self.downsample = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride,
                                padding=1, bias=False)
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out
    
class Encoder(nn.Module):
    def __init__(self, image_channels, embedding_channels):
        super().__init__()
        self.expansion = 1
        layers = [2, 2, 2, 2]
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(2, intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(2, intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(2, intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(2, intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(2048 * self.expansion, embedding_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

    def make_layers(self, num_residual_blocks, intermediate_channels, stride):
        layers = []

        layers.append(Conv_Block(self.in_channels, intermediate_channels, stride))
        self.in_channels = intermediate_channels * self.expansion # change to intermediate_channels
        for i in range(num_residual_blocks - 1):
            layers.append(Conv_Block(self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

# model = Encoder(image_channels=1, embedding_channels=64)
# print(model)
# out = model(torch.randn(2, 1, 250, 400))
# print(out[0].shape, out[1].shape)

class Transpose_Conv_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.transpose_conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                                  padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.transpose_conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1,
                                                  padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        
        if self.in_channels != self.out_channels:
            self.downsample = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride,
                                                padding=1, bias=False)
        else:
            self.downsample = nn.Identity()
    def forward(self, x):
        out = self.transpose_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.transpose_conv2(out)
        out = self.bn2(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, image_channels, embedding_channels, image_shape):
        super().__init__()
        self.expansion = 1
        layers = [2, 2, 2, 2]
        
        self.in_channels = 512
        self.fc = nn.Linear(embedding_channels, 2048 * self.expansion)
        
        self.layer4 = self.make_layers(2, intermediate_channels=256, stride=2)
        self.layer3 = self.make_layers(2, intermediate_channels=128, stride=2)
        self.layer2 = self.make_layers(2, intermediate_channels=64, stride=2)
        self.layer1 = self.make_layers(2, intermediate_channels=64, stride=1)

        self.transpose_conv1 = nn.ConvTranspose2d(64, image_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(image_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample((image_shape[0], image_shape[1]), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], x.shape[1]//4, 2, 2)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.transpose_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.upsample(x)
        return x

    def make_layers(self, num_residual_blocks, intermediate_channels, stride):
        layers = []

        layers.append(Transpose_Conv_Block(self.in_channels, intermediate_channels, stride))
        self.in_channels = intermediate_channels * self.expansion
        for i in range(num_residual_blocks - 1):
            layers.append(Transpose_Conv_Block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)
