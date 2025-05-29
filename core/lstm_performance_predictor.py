import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双层LSTM结构，使用dropout防止过拟合
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        elif x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 只使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions
        
    def predict_performance(self, client_id):
        """预测客户端的未来性能
        
        Args:
            client_id: 客户端ID
            
        Returns:
            float或None: 预测的性能值
        """
        # 检查历史记录
        if not hasattr(self, 'client_history') or client_id not in self.client_history:
            return None
            
        history = self.client_history[client_id]
        if len(history) < 2:  # 至少需要2个历史记录才能预测
            return None
            
        # 准备输入数据
        sequence = torch.tensor(history).float().unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # 使用模型预测
        self.eval()
        with torch.no_grad():
            prediction = self(sequence)
            
        return prediction.item()

class PerformancePredictor:
    def __init__(self, config):
        self.config = config
        self.model = LSTMPredictor(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        self.criterion = nn.MSELoss()
        
        # 早停相关参数
        self.patience = config.get('patience', 5)
        self.min_delta = config.get('min_delta', 0.001)
        self.best_loss = float('inf')
        self.counter = 0
        
    def predict(self, history_sequence):
        """
        预测客户端未来性能
        Args:
            history_sequence: 历史性能序列
        Returns:
            预测的未来性能
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(history_sequence).unsqueeze(-1)
            predictions = self.model(x)
        return predictions.numpy()
        
    def update(self, history_sequence, target):
        """
        更新预测模型
        Args:
            history_sequence: 历史性能序列
            target: 实际性能值
        Returns:
            float: 训练损失
            bool: 是否触发早停
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 直接传入序列，让模型处理维度
        x = torch.FloatTensor(history_sequence)
        y = torch.FloatTensor([target]).unsqueeze(-1)  # 修改目标张量维度为 [1, 1]
        
        predictions = self.model(x)
        loss = self.criterion(predictions, y)
        
        loss.backward()
        self.optimizer.step()
        
        # 早停检查
        if loss.item() < self.best_loss - self.min_delta:
            self.best_loss = loss.item()
            self.counter = 0
        else:
            self.counter += 1
            
        return loss.item(), self.counter >= self.patience

class PerformancePredictorManager:
    """性能预测管理器，负责训练和使用LSTM预测模型"""
    
    def __init__(self, config):
        """初始化性能预测管理器
        
        Args:
            config: 配置字典，包含所有必要的参数
        """
        self.config = config
        self.window_size = config['window_size']
        self.learning_rate = config['learning_rate']
        self.client_history = {}  # 客户端历史性能记录
        self.client_predictors = {}  # 客户端预测模型
        
    def update_client_history(self, client_id, performance_metrics):
        """更新客户端历史记录
        
        Args:
            client_id: 客户端ID
            performance_metrics: 性能指标，如 {'accuracy': 0.85, 'loss': 0.1}
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = []
            
        # 添加新的性能记录
        self.client_history[client_id].append(performance_metrics)
        
        # 保持窗口大小固定
        if len(self.client_history[client_id]) > self.window_size:
            self.client_history[client_id].pop(0)
    
    def _prepare_sequence_data(self, client_id, metric_name='accuracy'):
        """准备序列数据用于LSTM训练或预测
        
        Args:
            client_id: 客户端ID
            metric_name: 要使用的指标名称
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]或None: 输入序列和目标值
        """
        if client_id not in self.client_history:
            return None
            
        history = self.client_history[client_id]
        if len(history) < self.window_size:
            return None
            
        # 提取序列数据
        sequence = [record.get(metric_name, 0.0) for record in history]
        
        # 直接返回序列和目标值，让模型处理维度
        x = torch.tensor(sequence[:-1]).float()
        y = torch.tensor([sequence[-1]]).float()
        
        return x, y
    
    def train_predictor(self, client_id):
        """训练客户端的性能预测模型
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: 是否成功训练
        """
        # 准备数据
        data = self._prepare_sequence_data(client_id)
        if data is None:
            return False
            
        x, y = data
        
        # 创建或获取预测模型
        if client_id not in self.client_predictors:
            self.client_predictors[client_id] = PerformancePredictor(self.config)
        
        # 训练模型
        predictor = self.client_predictors[client_id]
        loss, should_stop = predictor.update(x, y)
        
        return True
    
    def predict_performance(self, client_id):
        """预测客户端的未来性能
        
        Args:
            client_id: 客户端ID
            
        Returns:
            float或None: 预测的性能值
        """
        # 检查历史记录
        if client_id not in self.client_history:
            return None
            
        history = self.client_history[client_id]
        if len(history) < self.window_size - 1:
            # 历史记录不足，无法预测
            return None
            
        # 准备输入数据
        sequence = [record.get('accuracy', 0.0) for record in history]
        x = torch.tensor(sequence).float().unsqueeze(0).unsqueeze(-1)  # [1, len(sequence), 1]
        
        # 检查是否有预测模型
        if client_id not in self.client_predictors:
            # 没有预测模型，创建一个
            self.client_predictors[client_id] = PerformancePredictor(self.config)
            # 新模型未经训练，返回最近的性能值作为预测结果
            return sequence[-1]
        
        # 使用模型预测
        predictor = self.client_predictors[client_id]
        
        predictor.model.eval()
        with torch.no_grad():
            prediction = predictor.model(x)
            
        return prediction.item() 