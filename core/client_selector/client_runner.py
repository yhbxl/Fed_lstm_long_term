#本地客户端训练逻辑

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any
import copy

class ClientModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 添加序列长度维度
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

def train_on_client(
    client: Dict[str, Any],
    model: nn.Module,
    config: Dict[str, Any]
) -> Tuple[nn.Module, int]:
    """在客户端上训练模型
    
    Args:
        client: 客户端信息，包含训练数据加载器和设备
        model: 全局模型
        config: 训练配置
        
    Returns:
        Tuple[nn.Module, int]: (训练后的模型, 样本数量)
    """
    # 创建模型副本
    client_model = copy.deepcopy(model)
    client_model.to(client['device'])
    client_model.train()
    
    # 获取训练配置
    optimizer = torch.optim.SGD(
        client_model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum']
    )
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = config['training']['local_epochs']
    train_loader = client['train_loader']
    
    # 训练模型
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(client['device'])
            targets = targets.to(client['device'])
            
            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return client_model, len(train_loader.dataset)

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """评估模型性能
    
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 设备
        
    Returns:
        Tuple[float, Dict[str, float]]: (损失, 指标字典)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, {
        'accuracy': accuracy,
        'loss': avg_loss
    }

def aggregate_models(
    models: List[nn.Module],
    weights: List[float]
) -> nn.Module:
    """聚合多个模型
    
    Args:
        models: 模型列表
        weights: 权重列表
        
    Returns:
        nn.Module: 聚合后的模型
    """
    if not models:
        raise ValueError("模型列表不能为空")
    
    # 创建聚合模型
    aggregated_model = copy.deepcopy(models[0])
    
    # 获取所有模型的状态字典
    state_dicts = [model.state_dict() for model in models]
    
    # 聚合参数
    aggregated_state_dict = {}
    for key in state_dicts[0].keys():
        aggregated_state_dict[key] = sum(
            state_dict[key] * weight 
            for state_dict, weight in zip(state_dicts, weights)
        )
    
    # 更新聚合模型
    aggregated_model.load_state_dict(aggregated_state_dict)
    
    return aggregated_model