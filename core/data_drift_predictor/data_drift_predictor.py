import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import deque

class DataDriftPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(DataDriftPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)
        return torch.sigmoid(prediction)  # 输出0-1之间的漂移概率

class DataDriftAnalyzer:
    def __init__(self, feature_size, window_size=10):
        """初始化数据漂移分析器
        
        Args:
            feature_size: 特征维度
            window_size: 观测窗口大小
        """
        self.feature_size = feature_size
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # 存储每个客户端的特征历史
        self.client_features = {}
        self.client_metrics = {}
        
    def extract_features(self, data_loader):
        """提取数据特征
        
        Args:
            data_loader: 数据加载器，如果为None则返回模拟特征
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (特征, 标签)
        """
        if data_loader is None:
            # 生成模拟特征
            features = torch.randn(10, self.feature_size)  # 模拟10个样本
            labels = torch.randint(0, 10, (10,))  # 模拟10个标签
            return features, labels
            
        features_list = []
        labels_list = []
        
        for batch_data, batch_labels in data_loader:
            features_list.append(batch_data.view(batch_data.size(0), -1))
            labels_list.append(batch_labels)
            
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels
        
    def update_client_features(self, client_id, data_loader):
        """更新客户端特征
        
        Args:
            client_id: 客户端ID
            data_loader: 数据加载器
        """
        features, labels = self.extract_features(data_loader)
        
        if client_id not in self.client_features:
            self.client_features[client_id] = deque(maxlen=self.window_size)
            self.client_metrics[client_id] = {
                'drift_score': deque(maxlen=self.window_size),
                'drift_probability': deque(maxlen=self.window_size),
                'data_quality': deque(maxlen=self.window_size)
            }
            
        # 计算当前特征的统计量
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        
        # 如果是第一个特征，初始化scaler
        if len(self.client_features[client_id]) == 0:
            self.scaler.fit(features.numpy())
            
        # 计算漂移分数
        normalized_features = self.scaler.transform(features.numpy())
        drift_score = np.mean(np.abs(normalized_features))
        
        # 计算漂移概率（基于漂移分数的sigmoid变换）
        drift_probability = 1 / (1 + np.exp(-drift_score + 5))  # 5是阈值
        
        # 计算数据质量分数（基于特征的方差和均值）
        quality_score = 1.0 / (1.0 + np.mean(std.numpy()))
        
        # 更新历史记录
        self.client_features[client_id].append((mean, std))
        self.client_metrics[client_id]['drift_score'].append(drift_score)
        self.client_metrics[client_id]['drift_probability'].append(drift_probability)
        self.client_metrics[client_id]['data_quality'].append(quality_score)
        
    def get_client_metrics(self, client_id):
        """获取客户端指标
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Dict: 包含drift_score, drift_probability, data_quality的字典
        """
        if client_id not in self.client_metrics:
            return {
                'drift_score': 0.0,
                'drift_probability': 0.0,
                'data_quality': 0.5  # 默认中等质量
            }
            
        metrics = self.client_metrics[client_id]
        return {
            'drift_score': float(np.mean(metrics['drift_score'])),
            'drift_probability': float(np.mean(metrics['drift_probability'])),
            'data_quality': float(np.mean(metrics['data_quality']))
        }
        
    def analyze_drift_trend(self, client_id):
        """分析漂移趋势
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Dict: 包含trend, confidence的字典
        """
        if client_id not in self.client_metrics or len(self.client_metrics[client_id]['drift_score']) < 3:
            return {
                'trend': 'unknown',
                'confidence': 0.0
            }
            
        drift_scores = list(self.client_metrics[client_id]['drift_score'])
        
        # 计算趋势
        trend_coefficient = np.polyfit(range(len(drift_scores)), drift_scores, 1)[0]
        
        if trend_coefficient > 0.1:
            trend = 'increasing'
        elif trend_coefficient < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
            
        # 计算趋势置信度
        confidence = min(1.0, abs(trend_coefficient) * 10)  # 将趋势系数转换为0-1的置信度
        
        return {
            'trend': trend,
            'confidence': float(confidence)
        } 