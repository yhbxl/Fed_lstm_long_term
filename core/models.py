import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):
    """MLP模型用于联邦学习"""
    
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.2
    ):
        """初始化MLP模型
        
        Args:
            dim_in: 输入维度
            dim_hidden: 隐藏层维度
            dim_out: 输出维度
            dropout: Dropout比率
        """
        super().__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出预测
        """
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class CNN(nn.Module):
    """CNN模型，用于MNIST数据集"""
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    """CNN模型用于CIFAR数据集"""
    
    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 10
    ):
        """初始化CNN模型
        
        Args:
            num_channels: 输入通道数
            num_classes: 输出类别数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出预测
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNetBlock(nn.Module):
    """ResNet基本块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet模型，用于CIFAR10数据集"""
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return F.log_softmax(out, dim=1)

def get_model(model_name: str, num_classes: int = 10):
    """获取模型实例"""
    if model_name == "cnn":
        return CNN(num_classes=num_classes)
    elif model_name == "cnncifar":
        return CNNCifar(num_classes=num_classes)
    elif model_name == "mlp":
        return MLP(
            dim_in=784,  # 28x28 for MNIST
            dim_hidden=200,
            dim_out=num_classes
        )
    elif model_name == "resnet":
        return ResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_gradients(model: nn.Module) -> List[torch.Tensor]:
    """获取模型参数的梯度
    
    Args:
        model: 模型实例
        
    Returns:
        List[torch.Tensor]: 梯度列表
    """
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone())
    return gradients

def set_gradients(model: nn.Module, gradients: List[torch.Tensor]):
    """设置模型参数的梯度
    
    Args:
        model: 模型实例
        gradients: 梯度列表
    """
    for param, grad in zip(model.parameters(), gradients):
        if param.grad is not None:
            param.grad.copy_(grad) 